import polars as pl
import json
import os

# Path to the main metadata file
METADATA_PATH = 'spotify_411k.parquet'
OUTPUT_PATH = 'assets/metadata_lite.json'

def generate_lite_index():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: {METADATA_PATH} not found.")
        return

    print(f"Loading metadata from {METADATA_PATH}...")
    # Load only necessary columns to keep the index small
    df = pl.read_parquet(METADATA_PATH)
    
    # Sort by popularity and take top 5000 tracks for the search index
    lite_df = df.sort('popularity', descending=True).head(5000)
    
    # Select columns for the search index
    lite_df = lite_df.select(['track_id', 'name', 'artist_name', 'genre', 'popularity'])
    
    # Convert to list of dicts
    records = lite_df.to_dicts()
    
    # Save as JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(records, f)
    
    print(f"Successfully generated lite index with {len(records)} tracks at {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_lite_index()
