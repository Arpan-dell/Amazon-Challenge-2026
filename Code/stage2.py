"""Second-stage features: judge each candidate against the other candidates.

The first model scores every (S1, Sx) pair on its own. An entity usually has
several records across Source 2 and 3, and every Sx record belongs to at most
one S1 entity, so a pair is easier to judge next to its competitors:

  own group   how this pair's first-stage probability ranks among the S1 entity's
              candidates, the best probability, how many look like matches
              (overall and within the same source)
  record      how the S1 entity ranks among the Sx record's candidate S1 entities
              and the best competing probability
  cross       how similar this record is to the S1 entity's likely matches
              (prob >= ANCHOR_MIN): a genuine extra record usually resembles the
              others, a look-alike differs from all of them the same way
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import normalize
from config import COUNTRIES

ANCHOR_MIN = 0.5
N_PARTS = 8           # compute cross similarities in this many slices of S1 entities (bounded memory)


def _texts(split, countries):
    parts = [normalize.load_normalised(split, c, columns=["id", "name_key", "addr_key"]) for c in countries]
    return pl.concat([p for p in parts if p is not None])


def _cross(scored, texts):
    """For every candidate: similarity to the S1 entity's other likely matches."""
    out = []
    ids = texts["id"]
    names, addrs = texts["name_key"], texts["addr_key"]
    pos = pl.DataFrame({"m": ids, "k": pl.int_range(0, len(ids), dtype=pl.UInt32, eager=True)})
    s = scored.select("s1", "m", "prob").join(pos, on="m", how="left")
    part = s["s1"].hash() % N_PARTS
    for p in range(N_PARTS):
        c = s.filter(part == p)
        anchors = c.filter(pl.col("prob") >= ANCHOR_MIN).select("s1", m2="m", k2="k", p2="prob")
        x = c.select("s1", "m", "k").join(anchors, on="s1").filter(pl.col("m") != pl.col("m2"))
        if x.height == 0:
            continue
        k, k2 = x["k"], x["k2"]
        xn = cpdist(names.gather(k).to_list(), names.gather(k2).to_list(), scorer=fuzz.token_set_ratio,
                    workers=-1, dtype=np.float32) / 100
        xa = cpdist(addrs.gather(k).to_list(), addrs.gather(k2).to_list(), scorer=fuzz.token_set_ratio,
                    workers=-1, dtype=np.float32) / 100
        x = x.with_columns(xn=xn, xa=xa).with_columns(xb=(pl.col("xn") + pl.col("xa")) / 2)
        out.append(x.group_by("s1", "m").agg(
            x_nm_max=pl.col("xn").max(), x_ad_max=pl.col("xa").max(), x_both_max=pl.col("xb").max(),
            x_support=(pl.col("p2") * pl.col("xb")).sum(), x_n=pl.len().cast(pl.Float32)))
    if not out:
        return scored.select("s1", "m").head(0)
    return pl.concat(out)


def group_features(scored, split, countries=COUNTRIES):
    """scored: [s1, m, src, prob] (first-stage) -> stage-2 features, same row order."""
    d = scored.select("s1", "m", "src", "prob").with_row_index("_r")
    top_b = d.group_by("m").agg(t1=pl.col("prob").max(), t2=pl.col("prob").top_k(2).min(), n_b=pl.len())
    d = d.join(top_b, on="m", how="left").with_columns(
        p_rank_a=pl.col("prob").rank("min", descending=True).over("s1"),
        p_max_a=pl.col("prob").max().over("s1"),
        p_sum_a=pl.col("prob").sum().over("s1"),
        n_hi_a=(pl.col("prob") >= ANCHOR_MIN).sum().over("s1"),
        n_hi_a_src=(pl.col("prob") >= ANCHOR_MIN).sum().over("s1", "src"),
        p_rank_b=pl.col("prob").rank("min", descending=True).over("m"),
        p_other_b=pl.when(pl.col("n_b") == 1).then(0.0)
        .when(pl.col("prob") >= pl.col("t1")).then(pl.col("t2")).otherwise(pl.col("t1")),
    ).with_columns(p_gap_a=pl.col("p_max_a") - pl.col("prob"),
                   p_margin_b=pl.col("prob") - pl.col("p_other_b"))
    cross = _cross(d, _texts(split, countries))
    d = d.join(cross, on=["s1", "m"], how="left").sort("_r")
    cols = ["prob", "p_rank_a", "p_max_a", "p_sum_a", "n_hi_a", "n_hi_a_src", "p_rank_b", "n_b",
            "p_other_b", "p_gap_a", "p_margin_b", "x_nm_max", "x_ad_max", "x_both_max", "x_support", "x_n"]
    return d.select(cols).rename({"prob": "p1"}).fill_null(-1).cast(pl.Float32)
