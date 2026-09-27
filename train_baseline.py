import polars as pl
from difflib import SequenceMatcher
import os

def string_similarity(a: str, b: str) -> float:
    """Calculate simple string similarity (Rule 5: Simple string-matching baseline)."""
    if not a or not b:
        return 0.0
    # Convert to strings in case of nulls or other types
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def load_data(filepath: str) -> pl.LazyFrame:
    """
    Load dataset using Polars to avoid OOM crashes (Rule 1).
    Returns a LazyFrame for optimized execution.
    """
    print(f"Loading data from {filepath} using Polars (Lazy)...")
    return pl.scan_parquet(filepath)

def block_and_match(df_lazy: pl.LazyFrame, blocking_key: str, threshold: float = 0.9) -> pl.DataFrame:
    """
    Implements a Blocking strategy to avoid O(n^2) comparisons (Rule 2).
    Uses a conservative matching threshold (Rule 4).
    """
    print(f"Applying blocking on '{blocking_key}'...")
    
    # We select only the columns we need to save memory
    # Assuming 'id', 'name', and blocking_key are available in the dataset
    df_subset = df_lazy.select([
        pl.col("id"),
        pl.col("name"),
        pl.col(blocking_key)
    ])
    
    # Create left and right frames for the self-join
    left = df_subset.rename({"id": "id_left", "name": "name_left"})
    right = df_subset.rename({"id": "id_right", "name": "name_right"})
    
    # Block by joining on the blocking key
    # This ensures we only compare entities within the same block (e.g., country, region)
    print("Performing blocking self-join...")
    potential_matches = left.join(right, on=blocking_key, how="inner")
    
    # Filter out self-matches (A-A) and duplicate pairs (B-A if A-B exists)
    potential_matches = potential_matches.filter(pl.col("id_left") < pl.col("id_right"))
    
    print("Collecting potential matches into memory...")
    # Collect the lazy frame to compute string similarity using Python UDFs
    try:
        df_matches = potential_matches.collect()
    except Exception as e:
        print(f"Error during collection: {e}")
        print("Dataset might be too large for in-memory collection after blocking.")
        print("Consider a more granular blocking key (e.g., zip code instead of country).")
        raise e

    print(f"Evaluating string similarities for {len(df_matches)} potential pairs...")
    
    # Calculate similarity score
    # Note: map_elements is used here for simplicity as per baseline rules.
    # For a highly optimized approach later, one could use polars plugins or rust extensions.
    similarities = [
        string_similarity(row["name_left"], row["name_right"]) 
        for row in df_matches.iter_rows(named=True)
    ]
    
    df_matches = df_matches.with_columns(
        pl.Series("similarity_score", similarities)
    )
    
    # Apply conservative threshold (Rule 4)
    print(f"Filtering with strict conservative threshold >= {threshold}...")
    final_matches = df_matches.filter(pl.col("similarity_score") >= threshold)
    
    return final_matches

def main():
    # Rule 3: No external data is used anywhere in this script.
    
    # Update this path if the dataset schema differs
    data_path = "data/train_source1.parquet"
    
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Ensure the dataset is downloaded locally.")
        print("This script is ready to run once the data is present.")
        # Proceeding with mock schema just for demonstration if needed, but we'll stop here.
        # return
        
    try:
        df_lazy = load_data(data_path)
    except Exception as e:
        print(f"Could not initialize LazyFrame: {e}")
        return
        
    # Example blocking key. Change this depending on the exact dataset columns available.
    # E.g., 'country', 'city', 'region', or a custom phonetic code blocking column.
    blocking_key = "country" 
    
    # High threshold (0.9) to avoid false positives (Rule 4)
    strict_threshold = 0.90 
    
    try:
        results = block_and_match(df_lazy, blocking_key, threshold=strict_threshold)
        print(f"Found {len(results)} highly confident matches.")
        
        # Save results
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "baseline_predictions.csv")
        results.write_csv(output_path)
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
