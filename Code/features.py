"""Pairwise features for pruned candidate pairs, computed in chunks.

  name      fuzzy scores on sorted / ordered / glued names, IDF-weighted token overlap,
            rarity of the rarest shared token, exact-key flags, legal form agreement,
            and whether the words found on one side only are typos of each other
            ("pharma" / "praa") or different words ("nelson" / "white": a look-alike)
  address   fuzzy scores, IDF-weighted overlap, numbers (house no.) agreement, state
  context   how common the name is (chains), which blocking keys fired, and how the
            candidate ranks against its competitors for the same S1 entity and for
            the same S2/S3 record
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from blocking import KEY_NAMES, fuzzy
from config import FEATURE_CHUNK


def _eq(x, y):
    """1 equal, 0 different, -1 either missing."""
    return (pl.when((x == "") | (y == "")).then(-1).when(x == y).then(1).otherwise(0)).cast(pl.Int8)


def _idf_overlap(ta, tb, idf, prefix):
    """IDF-weighted Jaccard and the rarity of the rarest shared token."""
    d = pl.DataFrame({"ta": ta, "tb": tb}).with_row_index("r").with_columns(
        inter=pl.col("ta").list.set_intersection("tb"), union=pl.col("ta").list.set_union("tb"))
    def wsum(col, agg):
        e = d.select("r", t=pl.col(col)).explode("t").join(idf, on="t", how="left")
        return e.group_by("r").agg(agg).sort("r")
    inter = wsum("inter", [pl.col("idf").sum().alias("wi"), pl.col("idf").max().alias("maxi")])
    union = wsum("union", [pl.col("idf").sum().alias("wu")])
    out = (d.select("r", n_inter=pl.col("inter").list.len(), n_union=pl.col("union").list.len())
           .join(inter, on="r", how="left").join(union, on="r", how="left").sort("r").fill_null(0))
    return {
        f"{prefix}_jacc": (out["n_inter"] / out["n_union"].clip(1)).to_numpy(),
        f"{prefix}_wjacc": (out["wi"] / out["wu"].clip(1e-6)).to_numpy(),
        f"{prefix}_rarest_shared": out["maxi"].to_numpy(),
        f"{prefix}_n_inter": out["n_inter"].to_numpy(),
    }


def _one_sided(ta, tb, idf, prefix, typo=0.8):
    """Tokens found on one side only: each is matched to its most similar token on the other
    side's one-sided list. Similar (>= typo) means a spelling change; dissimilar a different word."""
    d = pl.DataFrame({"ta": ta, "tb": tb}).with_row_index("r").select(
        "r", oa=pl.col("ta").list.set_difference("tb"), ob=pl.col("tb").list.set_difference("ta"))
    ea = d.select("r", x=pl.col("oa")).explode("x").drop_nulls("x")
    eb = d.select("r", x=pl.col("ob")).explode("x").drop_nulls("x")
    j = ea.join(eb, on="r", suffix="_o")
    j = j.with_columns(sim=cpdist(j["x"].to_list(), j["x_o"].to_list(), workers=-1, dtype=np.float32,
                                  scorer=JaroWinkler.normalized_similarity))
    best = pl.concat([ea.join(j.group_by("r", "x").agg(best=pl.col("sim").max()), on=["r", "x"], how="left"),
                      eb.join(j.group_by("r", "x_o").agg(best=pl.col("sim").max()).rename({"x_o": "x"}),
                              on=["r", "x"], how="left")]).with_columns(pl.col("best").fill_null(0.0))
    best = best.join(idf, left_on="x", right_on="t", how="left").with_columns(pl.col("idf").fill_null(0.0))
    agg = best.group_by("r").agg(
        n_typo=(pl.col("best") >= typo).sum(), n_sub=(pl.col("best") < typo).sum(),
        sub_idf=pl.when(pl.col("best") < typo).then(pl.col("idf")).otherwise(0.0).sum(),
        soft_min=pl.col("best").min())
    out = (d.select("r").join(agg, on="r", how="left").sort("r")
           .with_columns(pl.col("n_typo", "n_sub", "sub_idf").fill_null(0), pl.col("soft_min").fill_null(1.0)))
    return {f"{prefix}_{c}": out[c].to_numpy() for c in ("n_typo", "n_sub", "sub_idf", "soft_min")}


def _chunk_features(rec, p, idf_name, idf_addr):
    a, b = p["a"], p["b"]
    A = rec[a.to_numpy()]
    B = rec[b.to_numpy()]
    f = {
        "nm_tset": p["nm_tset"], "ad_tset": p["ad_tset"], "cheap": p["cheap"],
        "nm_ratio": fuzzy(rec["name_key"], a, b, fuzz.ratio),
        "nm_jw": 100 * fuzzy(rec["name_key"], a, b, JaroWinkler.normalized_similarity),
        "nm_partial": fuzzy(rec["name_seq"], a, b, fuzz.partial_ratio),
        "nm_seq_ratio": fuzzy(rec["name_seq"], a, b, fuzz.ratio),
        "cp_ratio": fuzzy(rec["compact"], a, b, fuzz.ratio),
        "cp_partial": fuzzy(rec["compact"], a, b, fuzz.partial_ratio),
        "ad_ratio": fuzzy(rec["addr_key"], a, b, fuzz.ratio),
        "ad_partial": fuzzy(rec["addr_key"], a, b, fuzz.partial_ratio),
    }
    f.update(_idf_overlap(A["name_toks"], B["name_toks"], idf_name, "nm"))
    f.update(_idf_overlap(A["addr_toks"], B["addr_toks"], idf_addr, "ad"))
    f.update(_one_sided(A["name_toks"], B["name_toks"], idf_name, "nm"))
    f.update(_one_sided(A["addr_toks"], B["addr_toks"], idf_addr, "ad"))
    f.update(_idf_overlap(A["nums"], B["nums"], pl.DataFrame({"t": [], "idf": []},
                                                            schema={"t": pl.String, "idf": pl.Float64}), "num"))
    eq = pl.DataFrame({"na": A["name_key"], "nb": B["name_key"], "ca": A["compact"], "cb": B["compact"],
                       "la": A["legal"], "lb": B["legal"], "sa": A["state"], "sb": B["state"],
                       "aa": A["addr_key"], "ab": B["addr_key"]}).select(
        name_exact=(pl.col("na") == pl.col("nb")).cast(pl.Int8),
        compact_exact=(pl.col("ca") == pl.col("cb")).cast(pl.Int8),
        legal_eq=_eq(pl.col("la"), pl.col("lb")),
        state_eq=_eq(pl.col("sa"), pl.col("sb")),
        addr_empty_a=(pl.col("aa") == "").cast(pl.Int8),
        addr_empty_b=(pl.col("ab") == "").cast(pl.Int8))
    f.update({c: eq[c].to_numpy() for c in eq.columns})
    f["num_conflict"] = ((A["nums"].list.len().to_numpy() > 0) & (B["nums"].list.len().to_numpy() > 0)
                         & (f["num_n_inter"] == 0)).astype(np.int8)
    f["ntok_a"] = A["name_toks"].list.len().to_numpy()
    f["ntok_b"] = B["name_toks"].list.len().to_numpy()
    f["domain_b"] = B["is_domain"].to_numpy().astype(np.int8)
    f["src3"] = (B["src"] == 3).to_numpy().astype(np.int8)
    f["name_freq_a"] = A["name_freq"].to_numpy()
    f["name_freq_b"] = B["name_freq"].to_numpy()
    keys = p["keys"].to_numpy()
    for bit, name in enumerate(KEY_NAMES):
        f[f"key_{name}"] = ((keys >> bit) & 1).astype(np.int8)
    f["n_keys"] = sum(f[f"key_{n}"] for n in KEY_NAMES)
    return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


def _context(p):
    """Competition features: rank/gap of this candidate among its S1's and its record's candidates."""
    out = {}
    for col in ("cheap", "nm_tset", "ad_tset"):
        for side, grp in (("a", "a"), ("b", "b")):
            out[f"rank_{side}_{col}"] = pl.col(col).rank("min", descending=True).over(grp)
            out[f"gap_{side}_{col}"] = pl.col(col).max().over(grp) - pl.col(col)
    out["n_cand_a"] = pl.len().over("a")
    out["n_cand_b"] = pl.len().over("b")
    return p.select(**out).cast(pl.Float32)


def make_features(rec, pairs):
    """rec: normalised records with row index `i`; pairs: pruned [a, b, keys, nm_tset, ad_tset, cheap]."""
    n = rec.height
    idf = lambda col: (rec.select(t=pl.col(col)).explode("t").drop_nulls().group_by("t").len()
                       .select("t", idf=(n / pl.col("len")).log()))
    idf_name, idf_addr = idf("name_toks"), idf("addr_toks")
    rec = rec.with_columns(name_freq=pl.len().over("name_key").cast(pl.Float32))
    chunks = [_chunk_features(rec, pairs.slice(s, FEATURE_CHUNK), idf_name, idf_addr)
              for s in range(0, pairs.height, FEATURE_CHUNK)]
    return pl.concat([pl.concat(chunks), _context(pairs)], how="horizontal")
