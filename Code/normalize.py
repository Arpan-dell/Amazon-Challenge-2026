"""Normalise names and addresses (vectorised with polars) and cache per country.

Output columns per record (data/cache/{split}_{country}/part-*.parquet):
  id, src                  entity id, source 1/2/3
  name_key                 sorted unique core name tokens ("fuel oller vanguard")
  name_seq                 core tokens in original order
  compact                  core tokens concatenated (matches domain names)
  legal                    canonical legal forms ("inc", "ltd pvt", ...)
  is_domain                name looked like "acme.com"
  name_toks                list of core tokens
  addr_key                 sorted unique address tokens (components are shuffled)
  addr_toks, nums, state   alphabetic tokens, numbers (no leading zeros), state code
"""
import re

import polars as pl

from config import COUNTRIES, P, log
from io_utils import read_source
from lexicon import (ADDR_ABBR, ADDR_COMMON, CITY_ALIASES, LEGAL, NAME_STOP,
                     state_map)
from translit import ASCII_PUNCT, MOJIBAKE, ascii_series, load

LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
                      "6": "g", "7": "t", "8": "b", "9": "g"})
_LEET_TOKEN = re.compile(r"^(?=.*[a-z].*[a-z])(?=.*\d)[a-z0-9]+$")
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
DOMAIN = r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|biz|info|us|fr)$"


def _fix_leet(text):
    out = []
    for t in text.split():
        if _LEET_TOKEN.match(t) and not _ORDINAL.match(t):
            t = t.translate(LEET)
        out.append(t)
    return " ".join(out)


def _python_subset(s, pattern, fn):
    idx = s.str.contains(pattern).arg_true()
    if len(idx) == 0:
        return s
    return s.clone().scatter(idx, [fn(v) for v in s.gather(idx).to_list()])


def _join_dotted(s):
    """"s.a.s." -> "sas", "l.l.c" -> "llc", "e.u.r.l." -> "eurl"."""
    for n in (4, 3, 2):
        pat = r"\b" + r"\.".join([r"([a-z])"] * n) + r"\.?(\s|$)"
        s = s.str.replace_all(pat, "".join(f"${{{i}}}" for i in range(1, n + 1)) + f"${{{n + 1}}}")
    return s


def _tokens(expr):
    return (expr.str.replace_all(ASCII_PUNCT, " ").str.split(" ")
            .list.eval(pl.element().filter(pl.element() != "")))


def normalise_names(names, tok_dict):
    s = ascii_series(names, tok_dict)
    s = s.str.replace_all(MOJIBAKE, " ").str.to_lowercase().str.replace_all(r"['’`]", "").str.strip_chars()
    s = s.str.replace_all(r"\(\s*id\s*:\s*[\[(]?\s*\d+\s*[\])]?\s*\)?", " ")   # "(ID: 90108)" record-id junk
    s = s.str.replace_all(r"[\[\](){}]", " ")   # "([L.L.C.])" -> " L.L.C. " so the dots can be joined
    s = _join_dotted(s)
    is_domain = s.str.contains(DOMAIN)
    s = s.str.replace(DOMAIN, "${1}")
    s = _python_subset(s, r"[a-z]\d|\d[a-z]", _fix_leet)
    df = pl.DataFrame({"s": s, "is_domain": is_domain})
    legal_keys = list(LEGAL)
    return (df.with_columns(toks=_tokens(pl.col("s")))
            .with_columns(
                legal=pl.col("toks").list.eval(
                    pl.element().filter(pl.element().is_in(legal_keys)).replace_strict(LEGAL, default=None)
                ).list.unique().list.sort().list.join(" "),
                core=pl.col("toks").list.eval(
                    pl.element().filter(~pl.element().is_in(legal_keys + list(NAME_STOP)))))
            .with_columns(core=pl.when(pl.col("core").list.len() > 0)
                          .then(pl.col("core")).otherwise(pl.col("toks")))
            .select(
                name_key=pl.col("core").list.unique().list.sort().list.join(" "),
                name_seq=pl.col("core").list.join(" "),
                compact=pl.col("core").list.join(""),
                legal="legal", is_domain="is_domain",
                name_toks=pl.col("core").list.unique()))


def normalise_addresses(addrs, country, tok_dict, comp_dict):
    smap = state_map(country)
    codes = sorted(set(smap.values()))
    abbr = {**ADDR_COMMON, **ADDR_ABBR.get(country, {})}
    cities = CITY_ALIASES.get(country, {})
    s = ascii_series(addrs, tok_dict, comp_dict)
    s = s.str.replace_all(MOJIBAKE, " ").str.to_lowercase().str.replace_all(r"['’`]", "")
    comps = s.str.split(",").list.eval(
        pl.element().str.replace_all(r"\s+", " ").str.strip_chars(" .-#()")
        .replace(smap).replace(cities))
    df = pl.DataFrame({"comps": comps})
    df = df.with_columns(
        state=pl.col("comps").list.eval(pl.element().filter(pl.element().is_in(codes))).list.first().fill_null(""),
        # state is kept as its own field, not as address tokens
        toks=_tokens(pl.col("comps").list.eval(pl.element().filter(~pl.element().is_in(codes))).list.join(" "))
        .list.eval(pl.element().replace(abbr).replace(cities))
        .list.eval(pl.element().filter(pl.element() != "").str.replace(r"^0+(\d)", "${1}")))
    return df.select(
        state="state",
        nums=pl.col("toks").list.eval(
            pl.element().str.extract_all(r"\d+").flatten().str.replace(r"^0+(\d)", "${1}")).list.drop_nulls().list.unique(),
        addr_toks=pl.col("toks").list.eval(
            pl.element().filter(~pl.element().str.contains(r"^\d+$") & (pl.element().str.len_chars() >= 2))
        ).list.unique(),
        addr_key=pl.col("toks").list.unique().list.sort().list.join(" "),
    )


def normalise_split(split, chunk=300_000):
    """Normalise one split in chunks (bounded memory) -> cache/{split}_{country}/part-*.parquet."""
    tok_dict, comp_dict = load()
    split_dir = P.data / split
    for old in P.cache.glob(f"{split}_*/part-*.parquet"):
        old.unlink()
    for k in (1, 2, 3):
        raw = read_source(split_dir, k)
        for country in COUNTRIES:
            part = raw.filter(pl.col("country") == country)
            out_dir = P.cache / f"{split}_{country}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, start in enumerate(range(0, part.height, chunk)):
                c = part.slice(start, chunk)
                out = pl.concat([
                    c.select(id="entity_id", src=pl.lit(k, pl.UInt8)),
                    normalise_names(c["business_name"], tok_dict),
                    normalise_addresses(c["business_address"], country, tok_dict, comp_dict),
                ], how="horizontal")
                out.write_parquet(out_dir / f"part-s{k}-{i:03d}.parquet", compression="zstd")
            if part.height:
                log(f"normalised {split}/{country}/source{k}: {part.height} records")
        del raw


def load_normalised(split, country, columns=None):
    """All records of one split+country, S1 first (row index = record index)."""
    d = P.cache / f"{split}_{country}"
    if not d.exists() or not any(d.glob("*.parquet")):
        return None
    return pl.read_parquet(sorted(d.glob("*.parquet")), columns=columns)
