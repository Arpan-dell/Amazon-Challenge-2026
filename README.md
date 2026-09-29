# Amazon ML Challenge 2026 — Business Entity Resolution

Links each **Source 1 (S1)** business to its matching records in **Source 2 / Source 3** using only
the business **name and address**. Built for the Amazon ML Challenge 2026 (macro F0.5 metric, which
penalises false merges more than misses).

- **No external data, APIs, geocoders or company registries.** The only fixed knowledge is the
  hand-written dictionaries in `Code/lexicon.py` (legal forms, street abbreviations, state names).
- **CPU only**, memory-safe: everything is processed with Polars, per country and in chunks.

## Features

- **Multilingual normalisation** — Indic native-script text (Devanagari, Gurmukhi, Tamil, Kannada,
  Gujarati, Bengali, Odia) is converted to Latin using a dictionary *learned from the training
  labels*, with `unidecode` as fallback.
- **Robust name cleaning** — mojibake, dotted abbreviations (`L.L.C.`), leetspeak, domain-style
  names, legal forms (US / Indian / French) separated from the core name, reorder-proof name keys.
- **Address handling** — shuffled components, per-country abbreviation expansion, state detection
  (names, codes, aliases), city aliases, zero-padded house numbers.
- **Blocking, not all-pairs** — hash-join candidate generation on each record's *rarest* tokens
  (8 key types: `name`, `compact`, `name_pair`, `num_addr`, `name_loc`, `rare_loc`, `addr_full`,
  `addr_pair`), with bucket caps and cheap-score pruning (top-k per S1 entity and per S2/S3 record).
- **Pair features** — fuzzy string scores (RapidFuzz), IDF-weighted token overlap, number
  agreement/conflict, state and legal-form agreement, name-frequency ("chain") signal, which keys
  fired, and rank/gap versus competing candidates.
- **Two-stage gradient-boosted model** — stage 1 scores each pair on its own; stage 2 (`stage2.py`)
  re-scores each pair against its competing candidates. LightGBM, with a scikit-learn
  `HistGradientBoostingClassifier` fallback if LightGBM isn't available.
- **Metric-driven decisions** — GroupKFold by S1 entity; the probability threshold and the
  exclusive-assignment policy (each S2/S3 record goes to at most one S1 entity) are tuned directly
  on macro F0.5. S1 entities with no confident match are predicted empty.
- **Country coverage** — US, India and France (France has no training labels).
- **Stage caching** — every stage writes Parquet caches, so stages can be re-run independently.

## Repository layout

| Path | Purpose |
|---|---|
| `Code/run_pipeline.py` | Command-line entry point |
| `Code/pipeline.py` | Stages: learn → normalize → featurize → train → predict |
| `Code/normalize.py`, `translit.py`, `lexicon.py` | Name/address normalisation and dictionaries |
| `Code/blocking.py` | Candidate generation and pruning |
| `Code/features.py`, `stage2.py` | Pair features and second-stage features |
| `Code/metric.py` | Macro F0.5, threshold and assignment policy |
| `Code/errors.py` | Optional error analysis on out-of-fold predictions |
| `Code/io_utils.py`, `config.py` | I/O and settings |
| `Code/kaggle_run.ipynb` | Notebook used to run the pipeline on Kaggle |
| `Code/README (1).md` | Detailed run notes for the pipeline |
| `entity_resolution_pipeline.py` | Simple Polars string-matching baseline |
| `documentation_template.md` | Methodology write-up |
| `output/matching_results.tsv` | Final predictions, one row per test S1 entity |

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

The full dataset needs roughly 30 GB of RAM (a Kaggle CPU notebook works). The dataset is **not**
included in this repository.

## Run

`--data` is the folder containing `train/` and `test/`; `--work` is where caches, models and outputs go.

```bash
cd Code
python run_pipeline.py --data /path/to/dataset --work /path/to/work learn normalize-train featurize-train
python run_pipeline.py --data /path/to/dataset --work /path/to/work train --train-fraction 0.4
python run_pipeline.py --data /path/to/dataset --work /path/to/work normalize-test featurize-test predict
```

Use `all` to run every stage in one go. Results are written to `<work>/output/matching_results.tsv`.

## Output format

Tab-separated, one row per test S1 entity; empty second column means no match was predicted.

```
source1_entity_id	matched_entity_ids
S1-714132312	S2-187020300,S2-637340732,S3-625880872,S3-867809779
S1-106407869	S3-585937637,S3-613056593
```

## Author

**Arpan Ailawadi** B.Tech, Electronics and Communication Engineering, Delhi Technological University [LinkedIn](https://www.linkedin.com/in/arpan-ailawadi/)
