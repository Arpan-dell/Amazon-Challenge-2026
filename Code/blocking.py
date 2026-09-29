"""Blocking: cheap candidate generation with hash joins, then cheap-score pruning.

Every record gets several keys; an (S1, Sx) pair becomes a candidate if they
share any key. Keys are built from each record's *rarest* tokens (document
frequency within the country), which survive typos in other tokens,
reordering, and legal-suffix changes:

  name        all core name tokens, sorted      (reordered / suffix-changed names)
  compact     core tokens glued together        (domain names: acme.com ~ Acme Co)
  name_pair   two rare name tokens              (records with empty addresses)
  num_addr    house number x rare addr token    (same address, name heavily altered)
  name_loc    whole name x any address word, number or state
  rare_loc    rare name token x any address word, number or state
              (common names such as "Tirupati Exports" recur all over India, so
              name-only buckets overflow the cap; a place narrows them down)
  addr_full   all address tokens                (same address, different brand name)
  addr_pair   two rare address tokens           (same, with a partly rewritten address)

Buckets with more than BUCKET_CAP (or KEY_CAPS[key]) S1 x Sx pairs are skipped
(too generic). Candidates are then scored with two fast fuzzy scores and pruned
to the top KEEP_PER_S1 per S1 entity that are also top KEEP_PER_SX for the Sx
record, plus the top KEEP_ADDR by address similarity on both sides.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from config import (ADDR_KEEP_MIN, BUCKET_CAP, KEEP_ADDR, KEEP_PER_S1, KEEP_PER_SX, KEY_CAPS,
                    N_RARE_ADDR, N_RARE_ADDR_PAIR, N_RARE_NAME, log)

KEY_NAMES = ["name", "compact", "name_pair", "num_addr", "name_loc", "rare_loc", "addr_full", "addr_pair"]


def rare_tokens(rec, col, n):
    """(i, t, df) for the n lowest-document-frequency tokens of each record."""
    ex = rec.select("i", t=pl.col(col)).explode("t").drop_nulls("t").filter(pl.col("t") != "").unique()
    df = ex.group_by("t").agg(df=pl.len())
    rare = (ex.join(df, on="t").sort("i", "df", "t")
            .group_by("i", maintain_order=True).head(n))
    return rare, df


def _h(*cols):
    return pl.concat_str([pl.col(c) for c in cols], separator="|").hash()


def key_frames(rec):
    rn, _ = rare_tokens(rec, "name_toks", N_RARE_NAME)
    ra, _ = rare_tokens(rec, "addr_toks", N_RARE_ADDR)
    ra_pair, _ = rare_tokens(rec, "addr_toks", N_RARE_ADDR_PAIR)
    places = pl.concat([rec.select("i", t=pl.col("addr_toks")).explode("t"),
                        rec.select("i", t=pl.col("nums")).explode("t"),
                        rec.select("i", t=pl.col("state"))]).drop_nulls("t").filter(pl.col("t") != "").unique()
    yield "name", rec.filter(pl.col("name_key") != "").select("i", key=pl.col("name_key").hash())
    yield "compact", rec.filter(pl.col("compact").str.len_chars() >= 5).select("i", key=pl.col("compact").hash())
    yield "name_pair", (rn.join(rn, on="i", suffix="_2").filter(pl.col("t") < pl.col("t_2"))
                        .select("i", key=_h("t", "t_2")))
    nums = (rec.select("i", n=pl.col("nums")).explode("n").drop_nulls("n")
            .filter(pl.col("n").str.len_chars() <= 6))
    yield "num_addr", nums.join(ra, on="i").select("i", key=_h("n", "t"))
    yield "name_loc", (rec.filter(pl.col("name_key") != "").select("i", nk="name_key")
                       .join(places, on="i").select("i", key=_h("nk", "t")))
    yield "rare_loc", rn.join(places, on="i", suffix="_p").select("i", key=_h("t", "t_p"))
    yield "addr_full", rec.filter(pl.col("addr_key").str.len_chars() >= 8).select("i", key=pl.col("addr_key").hash())
    yield "addr_pair", (ra_pair.join(ra_pair, on="i", suffix="_2").filter(pl.col("t") < pl.col("t_2"))
                        .select("i", key=_h("t", "t_2")))


def join_keys(keys, n1, cap=BUCKET_CAP):
    k = keys.unique()
    k1, kx = k.filter(pl.col("i") < n1), k.filter(pl.col("i") >= n1)
    ok = (k1.group_by("key").agg(n1=pl.len()).join(kx.group_by("key").agg(nx=pl.len()), on="key")
          .filter(pl.col("n1").cast(pl.Int64) * pl.col("nx").cast(pl.Int64) <= cap).select("key"))
    return (k1.join(ok, on="key").join(kx, on="key", suffix="_b")
            .select(a="i", b="i_b").unique())


def candidate_pairs(rec, n1):
    """All key-sharing (a, b) pairs; `keys` is a bitmask of which key types matched."""
    frames = []
    for bit, (name, keys) in enumerate(key_frames(rec)):
        p = join_keys(keys, n1, KEY_CAPS.get(name, BUCKET_CAP)).with_columns(bit=pl.lit(1 << bit, pl.UInt16))
        log(f"    key {name:>10}: {p.height:>10,} pairs")
        frames.append(p)
    return pl.concat(frames).group_by("a", "b").agg(keys=pl.col("bit").sum())


def fuzzy(rec_col, a, b, scorer, chunk=2_000_000):
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), chunk):
        x = rec_col.gather(a[s:s + chunk]).to_list()
        y = rec_col.gather(b[s:s + chunk]).to_list()
        out[s:s + chunk] = cpdist(x, y, scorer=scorer, workers=-1, dtype=np.float32) / 100
    return out


def prune(rec, pairs, keep_a=KEEP_PER_S1, keep_b=KEEP_PER_SX, keep_addr=KEEP_ADDR):
    a, b = pairs["a"], pairs["b"]
    nm = fuzzy(rec["name_key"], a, b, fuzz.token_set_ratio)
    ad = fuzzy(rec["addr_key"], a, b, fuzz.token_set_ratio)
    both_addr = ((rec["addr_key"].gather(a) != "") & (rec["addr_key"].gather(b) != "")).to_numpy()
    cheap = np.where(both_addr, 0.6 * nm + 0.4 * ad, 0.6 * nm + 0.2)
    rank = lambda col, side: pl.col(col).rank("ordinal", descending=True).over(side)
    pairs = pairs.with_columns(nm_tset=nm, ad_tset=ad, cheap=cheap,
                               ad_known=pl.Series(np.where(both_addr, ad, 0), dtype=pl.Float32))
    by_cheap = (rank("cheap", "a") <= keep_a) & (rank("cheap", "b") <= keep_b)
    by_addr = ((pl.col("ad_known") >= ADDR_KEEP_MIN)
               & (rank("ad_known", "a") <= keep_addr) & (rank("ad_known", "b") <= keep_addr))
    return pairs.filter(by_cheap | by_addr).drop("ad_known")


def build(rec, n1):
    log(f"  blocking {n1:,} S1 vs {rec.height - n1:,} S2/S3 records")
    pairs = candidate_pairs(rec, n1)
    log(f"    union: {pairs.height:,} pairs ({pairs.height / max(n1, 1):.1f} per S1)")
    pruned = prune(rec, pairs)
    log(f"    pruned: {pruned.height:,} pairs ({pruned.height / max(n1, 1):.1f} per S1)")
    return pairs.select("a", "b", "keys"), pruned
