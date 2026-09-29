"""Macro F0.5 over Source 1 entities (the challenge metric), vectorised with polars.

pred / truth: DataFrames with one row per (s1, m) pair.
"""
import polars as pl


def decide(scored, threshold, policy):
    """scored: [s1, m, prob] -> predicted pairs [s1, m].

    all   keep every candidate with prob >= threshold
    excl  additionally give each S2/S3 record to at most one S1 entity (its best);
          safe because no training record belongs to two S1 entities
    """
    d = scored.filter(pl.col("prob") >= threshold)
    if policy == "excl":
        d = d.sort("prob", descending=True).unique("m", keep="first")
    return d.select("s1", "m")


def score(pred, truth, s1_ids):
    base = pl.DataFrame({"s1": s1_ids}).unique()
    tp = pred.join(truth, on=["s1", "m"]).group_by("s1").agg(tp=pl.len())
    d = (base.join(tp, on="s1", how="left")
         .join(pred.group_by("s1").agg(npred=pl.len()), on="s1", how="left")
         .join(truth.group_by("s1").agg(ntrue=pl.len()), on="s1", how="left")
         .fill_null(0))
    p = pl.col("tp") / pl.col("npred")
    r = pl.col("tp") / pl.col("ntrue")
    d = d.with_columns(f=pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0)
                       .when(pl.col("tp") == 0).then(0.0)
                       .otherwise(1.25 * p * r / (0.25 * p + r)))
    single = d.filter(pl.col("ntrue") == 0)
    matched = d.filter(pl.col("ntrue") > 0)
    return {
        "macro_f05": d["f"].mean(),
        "singleton_acc": single["f"].mean() if single.height else float("nan"),
        "matched_f05": matched["f"].mean() if matched.height else float("nan"),
        "pair_precision": d["tp"].sum() / max(d["npred"].sum(), 1),
        "pair_recall": d["tp"].sum() / max(d["ntrue"].sum(), 1),
        "n_s1": d.height,
    }


def fmt(s):
    return "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items())
