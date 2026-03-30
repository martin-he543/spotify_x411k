# spotify_x411k
### It's like spotify_x5000, but with 411,000 songs instead of 5,000.
Created by Martin He, Jonah Kastner, Ryan Chawla, Lisa Vilceanu.

`multimodal_v9h.ipynb` is the most up-to-date model.

## File Structure
```text
spotify_x411k/
├── 📂 album_jonah/                   # Feature extraction for Jonah's album subset
│   ├── clip_embeddings.npy           # CLIP embeddings for album art
│   ├── clip_track_ids.csv            # Mapping for CLIP features
│   ├── colour_features.parquet       # Dominant colour/aesthetic features
│   ├── resnet_embeddings.npy         # ResNet visual embeddings
│   ├── resnet_track_ids.csv          # Mapping for ResNet features
│   └── sentiment_matrix.npy          # Textual sentiment features
│
├── 📂 checkpoints_album/             # Model checkpoints and visual features
│   ├── clip_embeddings.npy           # CLIP visual embeddings
│   ├── clip_track_ids.json           # Track ID mapping for visual features
│   ├── colour_texture.parquet        # Extracted texture/colour descriptors
│   └── zero_shot_labels.parquet      # Zero-shot classification labels
│
├── 📂 notebooks/                     # Development and extraction notebooks
│   ├── album_art_recommender.ipynb   # Main recommender logic (visual)
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   ├── language_tagging.ipynb        # NLP language processing
│   ├── musicnn.ipynb                 # Music audio feature extraction
│   ├── parquet_batch_inspector.ipynb # Utility for dataset inspection
│   ├── spectrogram.ipynb             # Audio spectrogram processing
│   ├── spotify_data.parquet          # Local sample dataset
│   ├── spotify_extractor_covers.ipynb # Album art scraping script
│   ├── spotify_extractor_sc_v2.ipynb  # Stream count scraping script
│   └── spotify_extractor_v9.ipynb     # Main data extraction pipeline
│
├── 📂 Root Files
│   ├── README.md                     # Project documentation
│   ├── eda_diagnostics.png           # Visualization of data distributions
│   ├── training_diagnostics.png      # Visualization of model training metrics
│   ├── multimodal_v9g.ipynb          # Principal Multimodal Model development
│   ├── multimodal_v9g.pth            # Trained PyTorch model weights
│   ├── update_notebook.py            # Script for keeping notebooks in sync
│   │
│   ├── 📊 Datasets (Parquet)
│   │   ├── languages.parquet         # Language metadata
│   │   ├── merged_sc.parquet         # Merged stream count data
│   │   ├── musicnn_final.parquet     # Final audio feature set
│   │   ├── nlp_features_v2.parquet   # Processed NLP/Text features
│   │   ├── spectrogram_features.parquet # Core audio spectrogram features
│   │   ├── spotify_411k.parquet      # Primary 411k track dataset
│   │   └── spotify_stream_counts.parquet # Raw stream count tracking
│   │
│   └── ⚙️ Configuration
│       └── .gitattributes            # LFS and attribute configuration
```

# Notebook Summary

Below is a detailed overview of the Jupyter notebooks used in this project, documenting their specific roles in the recommendation pipeline.

#### 1. [multimodal_v9g.ipynb](./multimodal_v9g.ipynb)
*   **Core Recommendation Engine**: Implements the final v9 multimodal fusion model, combining audio, lyrics, and visual signals.
*   **Asymmetric Architecture**: Uses a deep branch for audio (MusicNN + Spectrogram), a medium branch for lyrics (NLP), and a shallow branch for album art (Colour Palette).
*   **Dynamic Weighting**: Features a gated attention mechanism that learns to dynamically weigh different modalities based on input signals.
*   **Interactive Interface**: Provides a rich UI with hyperparameter sliders, pagination, and transparency into similarity breakdowns for different components.
*   **Weighted Metadata Fusion**: Integrates genre similarity, popularity signals, language matching, and audio feature proximity into the final ranking.

#### 2. [musicnn.ipynb](./notebooks/musicnn.ipynb)
*   **Audio Feature Extraction**: Uses the MusicNN deep learning library to extract 200-dimensional semantic embeddings from audio previews.
*   **Large-Scale Processing**: Implements a multi-GPU data-parallel sharding strategy to efficiently process 411,000 tracks.
*   **Process Isolation**: Employs a specific architecture (one process per GPU) to circumvent TensorFlow session lock contention.
*   **Batch Checkpointing**: Saves intermediate results as `.parquet` files to ensure progress is maintained across long-running extraction jobs.
*   **Semantic Understanding**: Captures high-level musical concepts and tags not present in raw waveform or low-level spectral data.

#### 3. [language_tagging.ipynb](./notebooks/language_tagging.ipynb)
*   **Metadata Enrichment**: Enriches the dataset by detecting the primary language of each track using titles and lyrics.
*   **Multi-Engine Fallback**: Uses `pycld2` for robust multi-language detection and falls back to `langdetect` for edge cases.
*   **Large-Scale NLP**: Efficiently processes hundreds of thousands of entries using vectorized operations.
*   **Feature Integration**: Creates the `languages.parquet` file used by the recommendation engine for language-based filtering.
*   **Data Validation**: Includes sanity check cells to visualise the distribution of languages across different genres.

#### 4. [album_art_recommender.ipynb](./notebooks/album_art_recommender.ipynb)
*   **Visual Similarity Pipeline**: Implements a two-stage recommendation process starting with visual signals from album covers.
*   **Multimodal Refinement**: Uses Stage 2 ranking to incorporate audio features and lyrics to refine the initial visual matches.
*   **Extensible Signal Architecture**: Contains implementations and stubs for ResNet, CLIP, and Sentiment analysis.
*   **Colour Mood Mapping**: Analyses dominant colour palettes and maps them to emotional/tonal signals for recommendation.
*   **Performance Optimisation**: Uses pre-calculated embeddings and efficient nearest-neighbour search for large image datasets.

#### 5. [eda.ipynb](./notebooks/eda.ipynb)
*   **Exploratory Data Analysis**: Performs a deep dive into the 411,000 track dataset to understand distributions and correlations.
*   **Statistical Visualisation**: Generates charts for track popularity, release years, and various audio features like valence and tempo.
*   **Outlier Detection**: Identifies gaps or anomalies in the source data that might affect recommendation quality.
*   **Genre Analysis**: Visualises the density and overlap of different musical genres within the Spotify dataset.
*   **Data Cleaning**: Documents the initial steps taken to normalise and prepare the raw data for feature extraction.

#### 6. [spectrogram.ipynb](./notebooks/spectrogram.ipynb)
*   **Spectral Feature Extraction**: Uses `scipy.signal.spectrogram` to extract frequency-domain features from audio files.
*   **Statistical Fingerprinting**: Computes mean and standard deviation of energy across frequency bins to create a feature vector.
*   **Large-Scale Batching**: Implements a robust checkpointing system to process 400,000+ tracks over multiple sessions.
*   **Cosine Similarity Search**: Builds an initial recommendation system based purely on spectral similarity vectors.
*   **Data Alignment**: Handles the merging of spectral features with the main track metadata via `track_id`.

#### 7. [parquet_batch_inspector.ipynb](./notebooks/parquet_batch_inspector.ipynb)
*   **Live Extraction Monitoring**: Allows for real-time inspection of `.parquet` and `.pkl` checkpoints generated during extraction.
*   **Progress Tracking**: Visualises batch indices, file sizes, and modification times to monitor long-running background processes.
*   **Data Validation**: Performs quick sanity checks on the most recently saved rows to ensure data integrity during harvesting.
*   **Status Aggregation**: Summarises extraction success rates (e.g., lyrics "ok" vs "not found") across all processed batches.
*   **Directory Monitoring**: Implements a loop-based watcher to refresh statistics as new checkpoint files appear.

#### 8. [spotify_extractor_covers.ipynb](./notebooks/spotify_extractor_covers.ipynb)
*   **Image Acquisition**: Automates the downloading of album cover art from Spotify's CDN in various resolutions.
*   **Restart Functionality**: Scans existing directories to skip already downloaded images, making the process resumable.
*   **Resolution Selection**: Allows toggling between Large (640px), Medium (300px), and Small (64px) image sizes.
*   **Error Handling**: Implements retry logic and checks for missing URLs or invalid image formats during mass download.
*   **Dataset Alignment**: Uses `track_id` as the filename to maintain a 1:1 mapping between metadata and visual assets.

#### 9. [spotify_extractor_sc_v2.ipynb](./notebooks/spotify_extractor_sc_v2.ipynb)
*   **Stream Count Harvesting**: Extracts numerical play counts for tracks by scraping artist data from aggregate sources like `kworb.net`.
*   **Fuzzy Name Matching**: Implements matching logic to align scraped stream counts with tracks in the local dataset.
*   **High-Volume Scraping**: Optimised for processing the top tracks and beyond, with batching and progress saving.
*   **Popularity Alignment**: Correlates Spotify's internal popularity score with raw global stream counts for better metric normalisation.
*   **Resumable Workflow**: Checks for existing results in `batch_results_streams/` before starting new requests to avoid redundant traffic.

#### 10. [spotify_extractor_v9.ipynb](./notebooks/spotify_extractor_v9.ipynb)
*   **Lyrics Extraction Pipeline**: Implements a sophisticated fallback chain to harvest song lyrics from multiple APIs (LRCLib, LibreLyrics, Genius).
*   **Multi-Provider Fallback**: Prioritises LRCLib (auth-free), then attempts Spotify-internal lyrics via `sp_dc` cookies, and finally Genius.
*   **Polite Scraping**: Features configurable sleep intervals, jitter, and retry strategies to avoid rate limits and IP blocks.
*   **Threaded Performance**: Uses `ThreadPoolExecutor` to parallelise API requests while maintaining thread-safe caches.
*   **Status Fingerprinting**: Tracks the exact source and error type for every track to measure data quality and source coverage.


# Parquet/Data Files

This section documents the various data assets and feature matrices used by the multimodal recommendation engine.

### 📊 Core Datasets (Root)

#### [spotify_411k.parquet](./spotify_411k.parquet)
*   **Master Metadata**: Contains the primary records for all 411,000 tracks, including IDs, names, and artists.
*   **Base Features**: Includes Spotify's native audio features like danceability, energy, and tempo for the full dataset.

#### [musicnn_final.parquet](./musicnn_final.parquet)
*   **Deep Audio Embeddings**: Stores high-dimensional semantic vectors extracted using the MusicNN architecture.
*   **Primary Signal**: Serves as the principal audio similarity feature within the v9 fusion model.

#### [nlp_features_v2.parquet](./nlp_features_v2.parquet)
*   **Textual Representations**: Contains dense vector embeddings derived from track lyrics and genre descriptions.
*   **Secondary Signal**: Provides the linguistic context used to refine recommendations based on lyrical themes.

#### [spectrogram_features.parquet](./spectrogram_features.parquet)
*   **Spectral Energy**: Encodes the frequency distribution and energy patterns across multiple spectral bands.
*   **Acoustic Fingerprint**: Used to match songs based on raw sonic characteristics and frequency response.

#### [languages.parquet](./languages.parquet)
*   **Language ISO Tags**: Maps each track ID to its detected primary language (e.g., 'en', 'es', 'ko').
*   **Filtering Signal**: Enables the recommendation engine to respect language preferences and regional constraints.

#### [spotify_stream_counts.parquet](./spotify_stream_counts.parquet)
*   **Popularity Metrics**: Tracks raw global play counts and historical streaming performance for the dataset.
*   **Ranking Signal**: Used to calculate a popularity-biased recommendation score for "trending" tracks.

#### [merged_sc.parquet](./merged_sc.parquet)
*   **Aligned Metrics**: A cleaned and deduplicated version of the stream count data mapped to the master track index.
*   **Data Integrity**: Ensures consistent numerical popularity values are available across different model versions.

#### [multimodal_v9g.pth](./multimodal_v9g.pth)
*   **Model Weights**: Stores the trained parameters for the v9 multimodal gated attention network.
*   **Inference State**: Required to run the recommendation engine without re-training the fusion layers.

### 🖼️ Visual & Auxiliary Features

#### [album_jonah/clip_embeddings.npy](./album_jonah/clip_embeddings.npy)
*   **Visual-Semantic Vectors**: Stores OpenAI CLIP embeddings representing the semantic content of album artwork.
*   **Cross-Modal Matching**: Allows for finding tracks with visual covers that match specific textual or conceptual themes.

#### [album_jonah/resnet_embeddings.npy](./album_jonah/resnet_embeddings.npy)
*   **Visual Feature Maps**: Contains deep convolutional features extracted using a ResNet architecture.
*   **Texture & Pattern Detection**: Captured low-level visual similarities like artistic style, line work, and complexity.

#### [album_jonah/colour_features.parquet](./album_jonah/colour_features.parquet)
*   **Aesthetic Descriptors**: Encodes dominant colour palettes, saturation levels, and brightness for each cover.
*   **Mood Mapping**: Used to correlate visual aesthetic signals with musical mood components (e.g., "dark" vs "vibrant").

#### [album_jonah/sentiment_matrix.npy](./album_jonah/sentiment_matrix.npy)
*   **Emotional Scoring**: Maps visual and textual metadata to a multi-dimensional sentiment and valence space.
*   **Affective Alignment**: Ensures recommended tracks share similar emotional "weight" as the seed track.

#### [checkpoints_album/zero_shot_labels.parquet](./checkpoints_album/zero_shot_labels.parquet)
*   **Categorical Tags**: Contains inferred genre and mood labels generated via zero-shot classification on images.
*   **Auxiliary Taxonomy**: Provides labels for tracks missing traditional genre metadata via visual inference.


# Hyperparameter Tuning

This section provides an in-depth breakdown of the tunable parameters and configurations used across the various notebooks to optimise performance and data quality.

### 🧠 Multimodal Recommendation (`multimodal_v9g.ipynb`)

The v9 model uses a sophisticated fusion strategy with three distinct levels of tuning:

#### Modality Fusion Weights
*   **Audio Primary (`musicnn`: 3.0, `spectrogram`: 2.0)**: Strategically set as the strongest signals. Audio represents the core artistic essence of the track.
*   **Lyrics Secondary (`nlp`: 1.5, `sentiment`: 0.8)**: Mid-level weights that provide textual context and emotional grounding.
*   **Visual Auxiliary (`colour`: 0.3, `clip`: 0.2, `resnet`: 0.2)**: Downweighted to act as subtle aesthetic bias rather than a primary driver.

#### Metadata Fusion Weights
*   **`genre_sim` (0.50)**: High weight to ensure results stay within a similar musical "neighbourhood."
*   **`language_sim` (0.30)**: Medium weight to respect linguistic preferences while allowing for some cross-cultural discovery.
*   **Audio Metadata (0.10 - 0.20)**: Lower weights for danceability, energy, and valence to allow the deep embeddings to handle the bulk of sonic matching.

#### Neural Network & Training
*   **Architectural Asymmetry**: Audio branch (512-dim hidden) is wider than NLP (256-dim) or Image (128-dim) to match signal complexity.
*   **Learning Rate (`3e-4`)**: Optimised for the AdamW optimiser with a cosine scheduler to ensure stable convergence.
*   **Label Smoothing (`0.05`)**: Prevents the model from becoming overconfident during genre-supervised pre-training.

---

### 🔊 Audio Feature Extraction (`musicnn.ipynb`)

The feature extraction pipeline is tuned for high-throughput and semantic depth:

*   **`PROCS_PER_GPU` (8)**: Tuned to maximise VRAM utilisation while avoiding TensorFlow session contention.
*   **`INPUT_LENGTH` (3s)**: Audio patches are processed in 3-second segments to capture rhythmic and melodic motifs efficiently.
*   **`LAYER` ('penultimate')**: Specifically targets the 200-dimensional semantic layer to capture high-level tags rather than low-level acoustic features.
*   **`L2_NORMALISE` (True)**: Ensures all embeddings live on a unit hypersphere, making cosine similarity comparisons mathematically sound.

---

### 📊 Spectral Analysis (`spectrogram.ipynb`)

Tuned for raw acoustic fingerprinting consistency:

*   **`sr` (22,050 Hz)**: Standardised sample rate to ensure frequency bins are consistent across the entire 411k track dataset.
*   **Feature Mapping**: Uses 129 frequency bins (FFT-derived) for both Mean and Std, resulting in a compressed 258-dimensional "acoustic fingerprint."
*   **`CHECKPOINT_N` (1000)**: Optimised flush frequency to balance disk I/O with progress safety for long-running jobs.

---

### 🕸️ Data Harvesting (`spotify_extractor_v9.ipynb`)

Tuned for reliability and politeness during large-scale API interaction:

*   **Concurrency (`MAX_WORKERS`: 16)**: Tuned to provide high throughput without triggering systemic IP blocks from primary providers.
*   **Rate Limiting**: Specifically tuned sleep intervals for different providers (LRCLib: 0.05s, Genius: 0.15s) to stay within "fair use" thresholds.
*   **Retry Logic (`MAX_RETRIES`: 2)**: Combined with an exponential backoff factor of 1.0 to handle transient network issues or API timeouts effectively.
*   **`BATCH_SIZE` (1000)**: Segmented processing ensures that partial successes are saved even if a network partition occurs.


# Neural Network Topology

The recommendation engine is powered by **MultimodalNet**, an asymmetric multimodal architecture designed to fuse heterogeneous signals (Audio, NLP, and Visual) into a unified semantic space for track ranking and classification.

### 🏛️ Model Architecture Overview

The network follows a modular, multi-branch design where each modality is projected through an independent branch before being fused via a gated attention mechanism.

#### 🎼 Audio Branch (Primary Highway)
*   **Depth**: 3-layer MLP with residual skips.
*   **Width**: 512-unit hidden dimension, projecting to a 256-unit output embedding.
*   **Regularisation**: Dropout (0.25) and Batch Normalisation.
*   **Rationale**: As the widest and deepest branch, it is designed to handle the high entropy of the 458-dimensional combined audio signal (MusicNN + Spectrogram).

#### 📜 NLP Branch (Secondary Path)
*   **Depth**: 2-layer MLP.
*   **Width**: 256-unit hidden dimension, projecting to a 128-unit output embedding.
*   **Regularisation**: Dropout (0.2).
*   **Rationale**: Captures linguistic and conceptual relationships from the 768-dimensional textual embeddings.

#### 👁️ Image Branch (Auxiliary Path)
*   **Depth**: 2-layer MLP.
*   **Width**: 128-unit hidden dimension, projecting to a 64-unit output embedding.
*   **Regularisation**: Dropout (0.3).
*   **Rationale**: Provides a lightweight representation of visual aesthetics from the 48-dimensional colour/texture features.

### ⚡ Gated Fusion Mechanism

Unlike simple concatenation, MultimodalNet employs a **Dynamic Gated Attention** layer to combine the branch outputs:

1.  **Concatenation**: Branch outputs are concatenated into a 448-dimensional intermediate vector.
2.  **Attention Gating**: A softmax-activated gated sub-network learns to assign three dynamic weights (summing to 1) representing the "trustworthiness" of each modality for a given track.
3.  **Weighted Fusion**: The gated weights are element-wise multiplied by their respective branch embeddings.
4.  **Classification Head**: The weighted vector passes through a final 256-unit hidden layer to predict the top-15 target genres.

### 🔬 Technical Summary
*   **Activation Function**: `GELU` (Gaussian Error Linear Unit) used across all layers for smoother gradient flow.
*   **Optimisation**: `AdamW` with weight decay (1e-5) to prevent catastrophic forgetting of pre-trained feature relationships.
*   **Learning Progression**: Uses a `CosineAnnealingLR` scheduler to fine-tune weights during the final stages of convergence.
