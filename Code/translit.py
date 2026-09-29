"""Native-script -> English dictionaries learned from the training labels.

Noisy sources write ~30% of Indian names in Devanagari, Tamil, Gujarati,
Kannada or Bengali ("शक्ति ईस्टर्न एनर्जी प्राइवेट लिमिटेड" = "Shakti Eastern
Energy Private Limited"), and state names in native script ("ಕರ್ನಾಟಕ"). Generic
transliteration (unidecode) is lossy, so we learn a token dictionary by aligning
matched training pairs position by position and voting. Unseen tokens fall back
to unidecode. Uses only the provided training data.
"""
import json
import re

import polars as pl
from unidecode import unidecode

from config import P, log
from io_utils import load_ground_truth_pairs, read_source
from lexicon import state_map

ASCII_PUNCT = r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]"
NON_ASCII = r"[^\x00-\x7F]"
NATIVE_ONLY = r"^[^A-Za-z0-9\x80-\xFF]*[\x{0900}-\x{0DFF}][^A-Za-z0-9\x80-\xFF]*$"  # Indic, no Latin/mojibake
MOJIBAKE = r"[ÂâÏïÃ][\x80-\xBF]+"               # e.g. "Â\x80\x93" = a mangled en dash
_PUNCT_RE = re.compile(ASCII_PUNCT)
_MOJIBAKE_RE = re.compile(MOJIBAKE)


def _tokens_expr(col):
    return (pl.col(col).str.to_lowercase().str.replace_all(ASCII_PUNCT, " ")
            .str.split(" ").list.eval(pl.element().filter(pl.element() != "")))


def learn(train_dir, min_count=2, min_share=0.5):
    s1 = read_source(train_dir, 1).select("entity_id", "business_name", "business_address", "country")
    pairs = load_ground_truth_pairs(train_dir)
    sx = pl.concat([read_source(train_dir, k) for k in (2, 3)]).rename(
        {"entity_id": "m", "business_name": "name_x", "business_address": "addr_x"}).drop("country")
    j = (pairs.join(sx, on="m")
         .join(s1.rename({"entity_id": "s1"}), on="s1"))

    # Name tokens: align equal-length token lists position by position.
    nm = (j.filter(pl.col("name_x").str.contains(NON_ASCII))
          .select(t1=_tokens_expr("business_name"), t2=_tokens_expr("name_x"))
          .filter(pl.col("t1").list.len() == pl.col("t2").list.len())
          .explode(["t1", "t2"])
          .filter(pl.col("t2").str.contains(NATIVE_ONLY) & ~pl.col("t1").str.contains(NON_ASCII)))
    tok_dict = _vote(nm, "t2", "t1", min_count, min_share)

    # Address components in native script -> the S1 record's state.
    comp_dict = {}
    for country in ("US", "India"):
        smap = state_map(country)
        a = (j.filter((pl.col("country") == country) & pl.col("addr_x").str.contains(NON_ASCII))
             .select(
                 st=pl.col("business_address").str.to_lowercase().str.split(",")
                 .list.eval(pl.element().str.strip_chars().replace_strict(smap, default=None))
                 .list.drop_nulls().list.first(),
                 comp=pl.col("addr_x").str.split(",")
                 .list.eval(pl.element().str.strip_chars().filter(pl.element().str.contains(NATIVE_ONLY))))
             .drop_nulls("st").explode("comp").drop_nulls("comp"))
        # a state name is shared by many businesses; anything else is entity-specific
        comp_dict.update(_vote(a, "comp", "st", min_count=50, min_share=0.8))

    P.translit.parent.mkdir(parents=True, exist_ok=True)
    P.translit.write_text(json.dumps({"tokens": tok_dict, "components": comp_dict},
                                        ensure_ascii=False, indent=0))
    log(f"learned {len(tok_dict)} name tokens, {len(comp_dict)} address components -> {P.translit}")
    return tok_dict, comp_dict


def _vote(df, src, dst, min_count, min_share):
    c = df.group_by(src, dst).len()
    tot = c.group_by(src).agg(pl.col("len").sum().alias("tot"))
    best = (c.sort("len", descending=True).unique(src, keep="first")
            .join(tot, on=src)
            .filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("tot") >= min_share)))
    return dict(zip(best[src].to_list(), best[dst].to_list()))


def load():
    d = json.loads(P.translit.read_text())
    return d["tokens"], d["components"]


def ascii_text(text, tok_dict, comp_dict=None):
    """Convert one string to ASCII, using learned dictionaries where possible."""
    text = _MOJIBAKE_RE.sub(" ", text)
    if comp_dict is not None:
        parts = [p.strip() for p in text.split(",")]
        parts = [comp_dict.get(p, p) for p in parts]
        text = ", ".join(parts)
    out = []
    for tok in text.split():
        if tok.isascii():
            out.append(tok)
            continue
        key = _PUNCT_RE.sub("", tok).lower()
        out.append(tok_dict.get(key) or unidecode(tok))
    return " ".join(out)


def ascii_series(s, tok_dict, comp_dict=None):
    """Apply ascii_text only to the rows that need it."""
    s = s.fill_null("")
    idx = s.str.contains(NON_ASCII).arg_true()
    if len(idx) == 0:
        return s
    vals = [ascii_text(v, tok_dict, comp_dict) for v in s.gather(idx).to_list()]
    return s.clone().scatter(idx, vals)
