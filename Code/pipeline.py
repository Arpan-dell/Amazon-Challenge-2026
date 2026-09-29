"""Pipeline stages. Each stage reads the previous stage's cache, so stages can be
re-run independently (e.g. retrain without recomputing features).

  learn      learn native-script dictionaries from train labels
  normalize  raw TSV -> cache/{split}_{country}/part-*.parquet
  featurize  blocking + features -> cache/{split}_{country}_feats.parquet
  train      stage 1 (pair model) and stage 2 (pair judged against its competitors, see
             stage2.py): grouped CV, threshold/policy search, final models -> models/matcher.pkl
  predict    test features -> output/matching_results.tsv + candidate_pairs.tsv
"""
import pickle
import warnings
import zlib

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

import blocking
import features
import normalize
import stage2
import translit
import config
from config import COUNTRIES, N_FOLDS, RANDOM_STATE, THRESHOLDS, P, log
from io_utils import (load_ground_truth_pairs, read_source, write_candidates,
                      write_matching)
from metric import decide, fmt, score

META = ["s1", "m", "src", "country"]
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def learn():
    translit.learn(P.data / "train")


def normalize_split(split):
    normalize.normalise_split(split)


def _feats_path(split, country):
    return P.cache / f"{split}_{country}_feats.parquet"


def featurize(split):
    truth = load_ground_truth_pairs(P.data / split) if split == "train" else None
    for country in COUNTRIES:
        rec = normalize.load_normalised(split, country)
        if rec is None:
            continue
        rec = rec.sort("src", maintain_order=True).with_row_index("i")
        n1 = int((rec["src"] == 1).sum())
        log(f"{split}/{country}")
        union, pruned = blocking.build(rec, n1)
        if truth is not None:
            _report_recall(rec, truth, union, pruned)
        X = features.make_features(rec, pruned)
        ids = rec["id"]
        meta = pl.DataFrame({"s1": ids.gather(pruned["a"]), "m": ids.gather(pruned["b"]),
                             "src": rec["src"].gather(pruned["b"]), "country": country})
        pl.concat([meta, X], how="horizontal").write_parquet(_feats_path(split, country))
        log(f"  wrote {_feats_path(split, country).name}: {X.height:,} pairs x {X.width} features")


def _report_recall(rec, truth, union, pruned):
    idx = rec.select("id", "i")
    t = (truth.join(idx.rename({"id": "s1", "i": "a"}), on="s1")
         .join(idx.rename({"id": "m", "i": "b"}), on="m").select("a", "b"))
    if t.height == 0:
        return
    hit = union.join(t, on=["a", "b"])
    r_union = hit.height / t.height
    r_pruned = pruned.join(t, on=["a", "b"]).height / t.height
    per_key = []
    keys = hit["keys"].to_numpy()
    for bit, name in enumerate(blocking.KEY_NAMES):
        on = (keys >> bit) & 1
        only = keys == (1 << bit)
        per_key.append(f"{name} {on.sum() / t.height:.3f}/{only.sum() / t.height:.3f}")
    log("    recall per key (any/only): " + "  ".join(per_key))
    log(f"    BLOCKING RECALL  keys: {r_union:.4f}   after pruning: {r_pruned:.4f}   ({t.height:,} true pairs)")


def _load_feats(split):
    parts = [pl.read_parquet(_feats_path(split, c)) for c in COUNTRIES if _feats_path(split, c).exists()]
    return pl.concat(parts, how="diagonal")


def new_model(stage=1):
    """stage 1: the pair model; stage 2: a smaller model on stage-1 output + group context."""
    try:
        import lightgbm as lgb
        if stage == 1:
            return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=127,
                                      min_child_samples=50, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1,
                                      random_state=RANDOM_STATE)
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                  min_child_samples=100, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1,
                                  random_state=RANDOM_STATE)
    except Exception:   # lightgbm missing, or libomp missing on macOS
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400 if stage == 1 else 250, learning_rate=0.08,
                                              max_leaf_nodes=127 if stage == 1 else 63,
                                              min_samples_leaf=50, l2_regularization=1.0,
                                              early_stopping=False, random_state=RANDOM_STATE)


def _cv(stage, X, y, folds):
    oof = np.zeros(len(y), np.float32)
    for k, (tr, va) in enumerate(folds):
        oof[va] = new_model(stage).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        log(f"  stage {stage} fold {k + 1}/{len(folds)} done")
    return oof


def _search(scored, truth, s1_ids, label, policies=("all", "excl")):
    """Best (macro F0.5, threshold, policy) on out-of-fold probabilities."""
    best = None
    for policy in policies:
        res = [(score(decide(scored, t, policy), truth, s1_ids)["macro_f05"], t) for t in THRESHOLDS]
        s, t = max(res)
        log(f"  {label} policy {policy:>4}: best CV macro F0.5 = {s:.4f} at threshold {t}")
        if best is None or s > best[0]:
            best = (s, t, policy)
    return best


def _in_fraction(col, frac):
    return pl.col(col).map_elements(lambda x: zlib.crc32(x.encode()) % 10_000 < frac * 10_000,
                                    return_dtype=pl.Boolean)


def train():
    df = _load_feats("train")
    truth = load_ground_truth_pairs(P.data / "train")
    s1_ids = read_source(P.data / "train", 1)["entity_id"]
    if config.TRAIN_S1_FRACTION < 1:
        keep = pl.DataFrame({"s1": s1_ids}).filter(_in_fraction("s1", config.TRAIN_S1_FRACTION))["s1"]
        s1_ids = keep
        df = df.filter(pl.col("s1").is_in(keep.implode()))
        truth = truth.filter(pl.col("s1").is_in(keep.implode()))
    df = (df.join(truth.with_columns(label=pl.lit(1, pl.Int8)), on=["s1", "m"], how="left")
          .with_columns(pl.col("label").fill_null(0)))
    feat_cols = [c for c in df.columns if c not in META + ["label"]]
    y = df["label"].to_numpy()
    log(f"training on {df.height:,} pairs, {y.sum():,} positives, {len(feat_cols)} features, "
        f"{len(s1_ids):,} S1 entities; candidate recall {y.sum() / truth.height:.4f}")
    log(f"model: {type(new_model()).__name__}")

    X = df.select(feat_cols).to_numpy()
    groups = df["s1"].hash().to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    oof1 = _cv(1, X, y, folds)
    scored1 = df.select("s1", "m").with_columns(prob=oof1)
    s1_best = _search(scored1, truth, s1_ids, "stage 1", policies=("excl",))
    model1 = new_model(1).fit(X, y)
    log("fitted final stage-1 model")

    G = stage2.group_features(df.select("s1", "m", "src").with_columns(prob=oof1), "train")
    X = np.hstack([X, G.to_numpy()])
    oof = _cv(2, X, y, folds)
    scored = df.select("s1", "m").with_columns(prob=oof)
    best = _search(scored, truth, s1_ids, "stage 2")
    _, t, policy = best
    log(f"CHOSEN policy={policy} threshold={t}: " + fmt(score(decide(scored, t, policy), truth, s1_ids)))
    log(f"  (stage 1 alone: {s1_best[0]:.4f}; predicting nothing would score "
        f"{score(scored.head(0).select('s1', 'm'), truth, s1_ids)['macro_f05']:.4f})")

    model2 = new_model(2).fit(X, y)
    with open(P.model, "wb") as fh:
        pickle.dump({"model": model1, "features": feat_cols, "model2": model2, "features2": G.columns,
                     "threshold": t, "policy": policy, "cv_score": best[0], "cv_score_stage1": s1_best[0]}, fh)
    log(f"saved {P.model}")
    pred = decide(scored, t, policy).with_columns(pred=pl.lit(1, pl.Int8))
    (scored.with_columns(p1=oof1, label=y).join(pred, on=["s1", "m"], how="left")
     .with_columns(pl.col("pred").fill_null(0)).write_parquet(P.dev / "oof.parquet"))
    truth.join(df.select("s1", "m"), on=["s1", "m"], how="anti").write_parquet(P.dev / "blocking_misses.parquet")
    _importance(model1, feat_cols)
    _importance(model2, feat_cols + G.columns)


def _importance(model, cols):
    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        top = sorted(zip(imp, cols), reverse=True)[:15]
        log("top features: " + ", ".join(f"{c}" for _, c in top))


def predict():
    with open(P.model, "rb") as fh:
        bundle = pickle.load(fh)
    parts = []
    for country in COUNTRIES:   # one country at a time: groups never span countries
        if not _feats_path("test", country).exists():
            continue
        df = pl.read_parquet(_feats_path("test", country))
        for c in bundle["features"]:
            if c not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(c))
        X = df.select(bundle["features"]).to_numpy()
        p1 = bundle["model"].predict_proba(X)[:, 1]
        prob = p1
        if bundle.get("model2") is not None:
            G = stage2.group_features(df.select("s1", "m", "src").with_columns(prob=p1), "test", [country])
            prob = bundle["model2"].predict_proba(np.hstack([X, G.select(bundle["features2"]).to_numpy()]))[:, 1]
        parts.append(df.select("s1", "m").with_columns(prob=prob, p1=p1))
        log(f"  scored test/{country}: {df.height:,} pairs")
        del df, X
    scored = pl.concat(parts)
    scored.write_parquet(P.dev / "test_scores.parquet")
    pred = decide(scored, bundle["threshold"], bundle["policy"])
    s1_ids = read_source(P.data / "test", 1)["entity_id"]
    write_matching(pred, s1_ids, P.output / "matching_results.tsv")
    write_candidates(scored.select("s1", "m"), s1_ids, P.output / "candidate_pairs.tsv")


def score_file(labels_path, pred_path=None):
    """Score a matching_results.tsv against held-out labels (local sample only)."""
    pred_path = pred_path or P.output / "matching_results.tsv"
    read = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
    explode = lambda df: (df.select(s1=pl.col(df.columns[0]), m=pl.col(df.columns[1]).fill_null("").str.split(","))
                          .explode("m").filter(pl.col("m") != ""))
    pred_df = read(pred_path)
    res = score(explode(pred_df), explode(read(labels_path)), pred_df[pred_df.columns[0]])
    log("HELD-OUT SCORE: " + fmt(res))
    return res
