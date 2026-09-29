"""Show the model's mistakes from the last `train` run, with the original text.

  python code/errors.py --data data/sample --work work/sample
  python code/errors.py --data data/sample --work work/sample --n 40 --country India

Three kinds of error, most costly first under F0.5:
  FALSE MERGES     predicted a match that is wrong (hurts precision)
  MISSED MATCHES   true pair was a candidate but the model said no
  BLOCKING MISSES  true pair never became a candidate (fix in blocking.py)
"""
import argparse

import polars as pl

import config
from config import P
from io_utils import read_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--country")
    args = ap.parse_args()
    config.setup(args.data, args.work)

    text = pl.concat([read_source(P.data / "train", k) for k in (1, 2, 3)]).select(
        id="entity_id", name="business_name", addr="business_address", country="country")
    def attach(df):
        df = (df.join(text.rename(lambda c: c + "_1"), left_on="s1", right_on="id_1")
              .join(text.rename(lambda c: c + "_x"), left_on="m", right_on="id_x"))
        return df.filter(pl.col("country_1") == args.country) if args.country else df

    oof = attach(pl.read_parquet(P.dev / "oof.parquet"))
    misses = attach(pl.read_parquet(P.dev / "blocking_misses.parquet"))
    sections = [
        ("FALSE MERGES", oof.filter((pl.col("pred") == 1) & (pl.col("label") == 0)).sort("prob", descending=True)),
        ("MISSED MATCHES", oof.filter((pl.col("pred") == 0) & (pl.col("label") == 1)).sort("prob")),
        ("BLOCKING MISSES", misses.sample(min(args.n, misses.height), seed=0) if misses.height else misses),
    ]
    for title, df in sections:
        print(f"\n===== {title}: {df.height:,} =====")
        for r in df.head(args.n).iter_rows(named=True):
            p = f"p={r['prob']:.2f}" if "prob" in r else "      "
            print(f"{p}  S1 {r['name_1'][:45]:45} | {r['addr_1'][:70]}")
            print(f"        {r['m'][:2]} {r['name_x'][:45]:45} | {(r['addr_x'] or '')[:70]}\n")


if __name__ == "__main__":
    main()
