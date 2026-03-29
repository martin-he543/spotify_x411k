import polars as pl
import json
import os

# Path to the main metadata file
METADATA_PATH = 'spotify_411k.parquet'
OUTPUT_PATH = 'assets/metadata_full.json'

def generate_full_index():
    if not os.path.exists(METADATA_PATH):
        print(f"Error: {METADATA_PATH} not found.")
        return

    print(f"Loading metadata from {METADATA_PATH}...")
    # Load all 411k tracks
    df = pl.read_parquet(METADATA_PATH)
    
    # Select only the columns needed for search/display to save space
    # we use 'track_id', 'name', 'artist_name', 'genre'
    df = df.select(['track_id', 'name', 'artist_name', 'genre'])
    
    # Convert to columnar format (dictionary of lists)
    # This is much smaller than a list of dictionaries as keys are not repeated
    columnar_data = {
        "ids": df['track_id'].to_list(),
        "names": df['name'].to_list(),
        "artists": df['artist_name'].to_list(),
        "genres": df['genre'].to_list()
    }
    
    # Save as JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"Saving full columnar index to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(columnar_data, f)
    
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"Successfully generated full index with {len(df)} tracks.")
    print(f"Output size: {size_mb:.2f} MB")

if __name__ == "__main__":
    generate_full_index()
