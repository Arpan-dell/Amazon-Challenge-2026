import polars as pl
from difflib import SequenceMatcher
import os

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = "."
TRAIN_S1_PATH = f"{DATA_DIR}/train_source1.parquet"
TRAIN_S2_PATH = f"{DATA_DIR}/train_source2.parquet"
TRAIN_S3_PATH = f"{DATA_DIR}/train_source3.parquet"
OUT_MATCHES = f"{DATA_DIR}/matching_results.csv"

def string_similarity(a: str, b: str) -> float:
    """Calculate simple string similarity (Rule 5: Simple string-matching baseline)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def load_source(path: str, tag: str) -> pl.LazyFrame:
    """
    Load dataset using Polars to avoid OOM crashes (Rule 1).
    Returns a LazyFrame for optimized execution.
    """
    print(f"Loading data from {path} using Polars (Lazy)...")
    # Using scan_parquet for Parquet files
    df = pl.scan_parquet(path)
    
    # Prefix columns to avoid collisions, except for blocking keys
    # Assuming 'id', 'name', 'address' might be present
    return df.with_columns(pl.lit(tag).alias("source_tag"))

def block_and_match(s1_lazy: pl.LazyFrame, other_lazy: pl.LazyFrame, blocking_key: str, threshold: float = 0.9) -> pl.DataFrame:
    """
    Implements a Blocking strategy to avoid O(n^2) comparisons (Rule 2).
    Uses a conservative matching threshold (Rule 4).
    """
    print(f"Applying blocking on '{blocking_key}'...")
    
    # We select only necessary columns to save memory
    cols = ["id", "name", blocking_key, "source_tag"]
    
    s1_sub = s1_lazy.select([pl.col(c) for c in cols if c in s1_lazy.collect_schema().names()])
    other_sub = other_lazy.select([pl.col(c) for c in cols if c in other_lazy.collect_schema().names()])
    
    # Rename for join
    s1_sub = s1_sub.rename({"id": "id_s1", "name": "name_s1", "source_tag": "tag_s1"})
    other_sub = other_sub.rename({"id": "id_other", "name": "name_other", "source_tag": "tag_other"})
    
    # Block by inner join on blocking_key
    potential_matches = s1_sub.join(other_sub, on=blocking_key, how="inner")
    
    # Process the blocked pairs
    try:
        df_matches = potential_matches.collect()
    except Exception as e:
        print(f"Error during collection (blocking key might be too broad): {e}")
        raise e

    print(f"Evaluating string similarities for {len(df_matches)} potential pairs...")
    
    # Rule 5: Simple string similarity baseline
    similarities = [
        string_similarity(row.get("name_s1", ""), row.get("name_other", "")) 
        for row in df_matches.iter_rows(named=True)
    ]
    
    df_matches = df_matches.with_columns(
        pl.Series("similarity_score", similarities)
    )
    
    # Rule 4: Conservative threshold
    print(f"Filtering with strict conservative threshold >= {threshold}...")
    final_matches = df_matches.filter(pl.col("similarity_score") >= threshold)
    
    return final_matches

def main():
    # Rule 3: No external data or APIs are used.
    
    # Using a naive blocking key for the baseline.
    # Depending on the dataset schema, use 'country', 'state', or first 3 letters of name
    # We will simulate a blocking key 'first_letter' of the name for this generic script
    
    # Create mock files for demonstration if they don't exist
    if not os.path.exists(TRAIN_S1_PATH):
        print(f"Missing data files. Place them in {DATA_DIR} to run the pipeline.")
        return

    try:
        s1 = load_source(TRAIN_S1_PATH, "s1")
        s2 = load_source(TRAIN_S2_PATH, "s2")
        
        # Add a simple blocking key (first letter of name) if an explicit one isn't available
        # In a real scenario, use 'country' or 'zip_code'
        s1 = s1.with_columns(pl.col("name").str.slice(0, 1).alias("block_key"))
        s2 = s2.with_columns(pl.col("name").str.slice(0, 1).alias("block_key"))
        
        strict_threshold = 0.90 
        
        results_s2 = block_and_match(s1, s2, blocking_key="block_key", threshold=strict_threshold)
        
        print(f"Found {len(results_s2)} highly confident matches against Source 2.")
        
        # Save results
        results_s2.write_csv(OUT_MATCHES, separator="\t")
        print(f"Results saved to {OUT_MATCHES}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
