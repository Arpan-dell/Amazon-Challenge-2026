import polars as pl
from rapidfuzz import fuzz
import os

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = "."
TRAIN_S1_PATH = f"{DATA_DIR}/train_source1.tsv"
TRAIN_S2_PATH = f"{DATA_DIR}/train_source2.tsv"
OUT_MATCHES = f"{DATA_DIR}/matching_results_v2.tsv"

def string_similarity(a: str, b: str) -> float:
    """Calculate rapidfuzz token_sort_ratio for better baseline matching."""
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(str(a).lower(), str(b).lower())

def load_source(path: str, tag: str) -> pl.LazyFrame:
    """
    Load dataset using Polars (Rule 1).
    """
    print(f"Loading data from {path} using Polars (Lazy)...")
    return pl.scan_csv(path, separator="\t", ignore_errors=True).with_columns(pl.lit(tag).alias("source_tag"))

def block_and_match(s1_lazy: pl.LazyFrame, other_lazy: pl.LazyFrame, blocking_key: str, threshold: float = 90.0) -> pl.DataFrame:
    """
    Implements a Blocking strategy to avoid O(n^2) comparisons (Rule 2).
    Uses a conservative matching threshold (Rule 4).
    """
    print(f"Applying blocking on '{blocking_key}'...")
    
    cols = ["id", "name", blocking_key, "source_tag"]
    s1_sub = s1_lazy.select([pl.col(c) for c in cols if c in s1_lazy.collect_schema().names()])
    other_sub = other_lazy.select([pl.col(c) for c in cols if c in other_lazy.collect_schema().names()])
    
    s1_sub = s1_sub.rename({"id": "id_s1", "name": "name_s1", "source_tag": "tag_s1"})
    other_sub = other_sub.rename({"id": "id_other", "name": "name_other", "source_tag": "tag_other"})
    
    # Block by joining on blocking_key
    potential_matches = s1_sub.join(other_sub, on=blocking_key, how="inner")
    
    try:
        df_matches = potential_matches.collect()
    except Exception as e:
        print(f"Error during collection: {e}")
        raise e

    print(f"Evaluating string similarities for {len(df_matches)} potential pairs...")
    
    # Using rapidfuzz for a slightly better, but still simple string-matching baseline (Rule 5)
    similarities = [
        string_similarity(row.get("name_s1", ""), row.get("name_other", "")) 
        for row in df_matches.iter_rows(named=True)
    ]
    
    df_matches = df_matches.with_columns(
        pl.Series("similarity_score", similarities)
    )
    
    # Rule 4: Conservative threshold (RapidFuzz returns 0-100)
    print(f"Filtering with strict conservative threshold >= {threshold}...")
    final_matches = df_matches.filter(pl.col("similarity_score") >= threshold)
    
    return final_matches

def main():
    # Rule 3: No external data is used.
    if not os.path.exists(TRAIN_S1_PATH):
        print("Missing dataset files.")
        return

    try:
        s1 = load_source(TRAIN_S1_PATH, "s1")
        s2 = load_source(TRAIN_S2_PATH, "s2")
        
        # Simulating a blocking key (e.g., zip_code or city in real dataset)
        # Here we block on the first 3 characters of the name for a slightly stricter block than V1
        s1 = s1.with_columns(pl.col("name").str.slice(0, 3).str.to_lowercase().alias("block_key"))
        s2 = s2.with_columns(pl.col("name").str.slice(0, 3).str.to_lowercase().alias("block_key"))
        
        # High conservative threshold (RapidFuzz uses 0-100 scale)
        strict_threshold = 90.0 
        
        results_s2 = block_and_match(s1, s2, blocking_key="block_key", threshold=strict_threshold)
        print(f"Found {len(results_s2)} highly confident matches against Source 2.")
        
        results_s2.write_csv(OUT_MATCHES, separator="\t")
        print(f"Results saved to {OUT_MATCHES}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
