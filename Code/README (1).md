# Business entity resolution pipeline

This pipeline links each Source 1 business to its Source 2 and Source 3 records using only the name
and address. The method is described in `documentation_template.md` at the top of the submission.

## Requirements
Python 3.10+ and a CPU machine. The full dataset needs about 30 GB of RAM, for example a Kaggle CPU
notebook.
```bash
pip install -r requirements.txt
```

## Run
`--data` is the folder that contains `train/` and `test/` with the challenge TSVs. `--work` is where
caches, models and outputs are written.
```bash
python run_pipeline.py --data /path/to/dataset --work /path/to/work learn normalize-train featurize-train
python run_pipeline.py --data /path/to/dataset --work /path/to/work train --train-fraction 0.4
python run_pipeline.py --data /path/to/dataset --work /path/to/work normalize-test featurize-test predict
```
To run every stage in one go, use `all`; it fits on 100% of training entities unless you add
`--train-fraction`.

Outputs:
- `/path/to/work/output/matching_results.tsv`: final matches, one row per test S1 entity.
- `/path/to/work/output/candidate_pairs.tsv`: the blocking candidate set that was fed to the model.

`kaggle_run.ipynb` runs the same commands on Kaggle, and was used for the submitted run (about 2 hours).

## Files
| File | Role |
|---|---|
| `run_pipeline.py` | command-line entry point (stages below) |
| `pipeline.py` | stages: learn → normalize → featurize → train → predict |
| `translit.py` | learns native-script → Latin dictionaries from the training labels |
| `lexicon.py` | legal forms, address abbreviations, state names, city aliases |
| `normalize.py` | name and address normalisation, cached per split and country |
| `blocking.py` | hash-key candidate generation and cheap-score pruning |
| `features.py` | 57 pair features |
| `metric.py` | macro F0.5 (challenge metric), threshold and assignment policy |
| `io_utils.py` | reading TSVs and writing the two output files |
| `config.py` | paths and tunable settings |
| `errors.py` | error analysis on out-of-fold predictions (optional) |
