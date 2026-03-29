import os
import shutil
import polars as pl
import numpy as np

OUTPUT_DIR = "export"
LITE_SIZE = 10000

PATHS = {
    'metadata'   : 'spotify_411k.parquet',
    'musicnn'    : 'musicnn_final.parquet',
    'nlp'        : 'nlp_features_v2.parquet',
    'sentiment'  : 'album_jonah/sentiment_matrix.npy',
    'ct'         : 'checkpoints_album/colour_texture.parquet',
    'zs'         : 'checkpoints_album/zero_shot_labels.parquet',
    'spectrogram': 'spectrogram_features.parquet',
    'resnet'     : 'album_jonah/resnet_embeddings.npy',
    'clip'       : 'checkpoints_album/clip_embeddings.npy',
}

def build_stlite():
    print(f"Building stlite deployment payload into '{OUTPUT_DIR}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    metadata = pl.read_parquet(PATHS['metadata'])
    
    # Save FULL metadata JSON/parquet
    metadata.write_parquet(os.path.join(OUTPUT_DIR, f"lite_{PATHS['metadata'].split('/')[-1]}"))
    
    lite_tids = set(metadata['track_id'].to_list())
    original_tids = metadata['track_id'].to_list()
    lite_meta = metadata
    
    # Map track_id to original index (for NPY slicing, even if no-op)
    orig_idx_map = {t: i for i, t in enumerate(original_tids)}
    lite_indices = list(range(len(original_tids)))
    
    # Process features
    print("Slicing feature matrices to match lite track list...")
    def slice_parquet(path, prefix='lite_'):
        if not os.path.exists(path): 
            print(f" -> WARNING: Skipping {path} because it does not exist.")
            return
        df = pl.read_parquet(path)
        lite_df = df.filter(pl.col('track_id').is_in(lite_tids))
        out_path = os.path.join(OUTPUT_DIR, prefix + path.split('/')[-1])
        lite_df.write_parquet(out_path)
        print(f" -> Exported {out_path} ({len(lite_df)} tracks)")

    slice_parquet(PATHS['musicnn'])
    slice_parquet(PATHS['nlp'])
    slice_parquet(PATHS['spectrogram'])
    slice_parquet(PATHS['ct'])
    slice_parquet(PATHS['zs'])
    
    # Slicing NPY is harder if we don't have track_ids in the NPY. 
    # Usually NPY matches the full spotify_411k.parquet index rows.
    # Therefore, we need the row indices of lite_tids in the *original* metadata.
    original_tids = metadata['track_id'].to_list()
    # Map track_id to original index
    orig_idx_map = {t: i for i, t in enumerate(original_tids)}
    
    # The indices we want to save
    lite_indices = [orig_idx_map[t] for t in lite_meta['track_id'].to_list()]
    
    def slice_npy(path, prefix='lite_'):
        if not os.path.exists(path): return
        arr = np.load(path)
        if len(arr) != len(original_tids):
            print(f" -> Skipping {path} because size mismatch ({len(arr)} != {len(original_tids)})")
            return
        lite_arr = arr[lite_indices]
        out_path = os.path.join(OUTPUT_DIR, prefix + path.split('/')[-1])
        np.save(out_path, lite_arr)
        print(f" -> Exported {out_path} ({len(lite_arr)} rows)")

    slice_npy(PATHS['sentiment'])
    slice_npy(PATHS['resnet'])
    slice_npy(PATHS['clip'])
    
    print("Copying App.py...")
    shutil.copy("app.py", os.path.join(OUTPUT_DIR, "app.py"))
    
    print("Generating stlite HTML wrapper...")
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <title>Multimodal Discovery (Lite)</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.58.0/build/stlite.css" />
  </head>
  <body>
    <div id="root"></div>
    <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.58.0/build/stlite.js"></script>
    <script>
        stlite.mount(
            {
                requirements: ["streamlit", "polars", "pandas", "numpy", "scipy", "scikit-learn", "plotly"],
                entrypoint: "app.py",
                files: {
                    "app.py": { url: "./app.py" },
                    "spotify_411k.parquet": { url: "./lite_spotify_411k.parquet" },
                    "musicnn_final.parquet": { url: "./lite_musicnn_final.parquet" },
                    "nlp_features_v2.parquet": { url: "./lite_nlp_features_v2.parquet" },
                    "spectrogram_features.parquet": { url: "./lite_spectrogram_features.parquet" },
                    "album_jonah/sentiment_matrix.npy": { url: "./lite_sentiment_matrix.npy" },
                    "checkpoints_album/colour_texture.parquet": { url: "./lite_colour_texture.parquet" },
                    "checkpoints_album/zero_shot_labels.parquet": { url: "./lite_zero_shot_labels.parquet" },
                    "album_jonah/resnet_embeddings.npy": { url: "./lite_resnet_embeddings.npy" },
                    "checkpoints_album/clip_embeddings.npy": { url: "./lite_clip_embeddings.npy" }
                }
            },
            document.getElementById("root")
        );
    </script>
  </body>
</html>
        """)

    print("Success! Static site bundle ready in /export/")

if __name__ == "__main__":
    build_stlite()
