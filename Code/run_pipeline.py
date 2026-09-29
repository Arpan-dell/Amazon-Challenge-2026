"""Command-line entry point.

Local sample (fast, for development):
  python code/run_pipeline.py --data data/sample --work work/sample all
  python code/run_pipeline.py --data data/sample --work work/sample score --labels data/sample/test_labels.tsv

Full data (Kaggle):
  python code/run_pipeline.py --data /kaggle/input/<dataset>/dataset --work /kaggle/working all

Stages (run in this order; "all" runs every one):
  learn  normalize-train  featurize-train  train  normalize-test  featurize-test  predict
"""
import argparse

import config
import pipeline

STAGES = {
    "learn": pipeline.learn,
    "normalize-train": lambda: pipeline.normalize_split("train"),
    "featurize-train": lambda: pipeline.featurize("train"),
    "train": pipeline.train,
    "normalize-test": lambda: pipeline.normalize_split("test"),
    "featurize-test": lambda: pipeline.featurize("test"),
    "predict": pipeline.predict,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder containing train/ and test/")
    ap.add_argument("--work", required=True, help="folder for caches, models and outputs")
    ap.add_argument("stages", nargs="+", choices=list(STAGES) + ["all", "score"])
    ap.add_argument("--labels", help="held-out labels for the score stage")
    ap.add_argument("--train-fraction", type=float, help="share of train S1 entities to fit on (saves RAM)")
    args = ap.parse_args()

    config.setup(args.data, args.work)
    if args.train_fraction:
        config.TRAIN_S1_FRACTION = args.train_fraction
    for stage in args.stages:
        if stage == "all":
            for fn in STAGES.values():
                fn()
        elif stage == "score":
            pipeline.score_file(args.labels)
        else:
            STAGES[stage]()


if __name__ == "__main__":
    main()
