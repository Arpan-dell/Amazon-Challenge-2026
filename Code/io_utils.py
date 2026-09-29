"""Reading the challenge TSVs and writing submission files in the validator's format."""
from pathlib import Path

import polars as pl

from config import log

MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CAND_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def _read(path):
    # quote_char=None: names contain stray quotes; everything is read as text.
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def source_path(split_dir, k):
    split_dir = Path(split_dir)
    hits = sorted(split_dir.glob(f"*source{k}.tsv"))
    if not hits:
        raise FileNotFoundError(f"no *source{k}.tsv in {split_dir}")
    return hits[0]


def read_source(split_dir, k):
    df = _read(source_path(split_dir, k)).with_columns(
        pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""))
    return df


def load_ground_truth_pairs(split_dir):
    """Ground truth as one row per true pair: [s1, m]."""
    gt = _read(next(Path(split_dir).glob("*ground_truth.tsv")))
    return (gt.select(s1=pl.col("source1_entity_id"),
                      m=pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("m").filter(pl.col("m") != ""))


def _write_lists(pairs, s1_ids, col, header, path):
    """pairs: DataFrame [s1, m]; one output row per S1 id, comma-separated list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = pairs.group_by("s1").agg(pl.col(col).unique().sort().str.join(","))
    out = (pl.DataFrame({"s1": s1_ids}).join(grouped, on="s1", how="left", maintain_order="left")
           .with_columns(pl.col(col).fill_null(""))
           .rename({"s1": header[0], col: header[1]}))
    out.write_csv(path, separator="\t", quote_style="never")   # empty list = empty field, not ""
    log(f"wrote {path}: {out.height} rows, {(out[header[1]] != '').sum()} non-empty")


def write_matching(pairs, s1_ids, path):
    _write_lists(pairs.rename({"m": "v"}), s1_ids, "v", MATCH_HEADER, path)


def write_candidates(pairs, s1_ids, path):
    _write_lists(pairs.rename({"m": "v"}), s1_ids, "v", CAND_HEADER, path)
