"""Paths and tunable settings shared by every stage of the pipeline.

Paths are set once per run with setup(data_root, work_dir) so the same code
runs locally (data/sample, data/raw/...) and on Kaggle (/kaggle/input/...).
  data_root: folder containing train/ and test/ with the challenge TSVs
  work_dir:  where caches, models and outputs go
"""
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class P:
    data = ROOT / "data" / "raw" / "archive" / "dataset"
    work = ROOT / "work" / "full"
    cache = work / "cache"
    models = work / "models"
    output = work / "output"
    dev = output / "dev"
    translit = models / "translit.json"
    model = models / "matcher.pkl"


def setup(data_root=None, work_dir=None):
    if data_root:
        P.data = Path(data_root)
    if work_dir:
        P.work = Path(work_dir)
    P.cache = P.work / "cache"
    P.models = P.work / "models"
    P.output = P.work / "output"
    P.dev = P.output / "dev"
    P.translit = P.models / "translit.json"
    P.model = P.models / "matcher.pkl"
    for d in (P.cache, P.models, P.output, P.dev):
        d.mkdir(parents=True, exist_ok=True)
    log(f"data={P.data}  work={P.work}")


COUNTRIES = ["US", "India", "France"]

# --- Blocking ---------------------------------------------------------------
N_RARE_NAME = 3            # rarest name tokens per record used in keys
N_RARE_ADDR = 2            # rarest address tokens per record used in num_addr keys
N_RARE_ADDR_PAIR = 3       # rarest address tokens paired up in addr_pair keys
BUCKET_CAP = 2000          # skip key buckets with more S1 x Sx pairs than this...
KEY_CAPS = {"rare_loc": 300, "addr_pair": 300}   # ...or this, for the keys that make many pairs
KEEP_PER_S1 = 25           # after cheap scoring: keep top-k candidates per S1 entity...
KEEP_PER_SX = 5            # ...that also rank in the top-k S1 entities for the record
KEEP_ADDR = 5              # also keep the top-k by address similarity on both sides (same address,
ADDR_KEEP_MIN = 0.8        # different brand name), if the address similarity is at least this

# --- Matching model ---------------------------------------------------------
TRAIN_S1_FRACTION = 1.0    # share of train S1 entities used to fit the model (lower if RAM is short)
N_FOLDS = 4
RANDOM_STATE = 42
THRESHOLDS = [round(0.20 + 0.02 * i, 2) for i in range(40)]   # 0.20 .. 0.98
FEATURE_CHUNK = 500_000    # pairs per feature-computation chunk

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)
