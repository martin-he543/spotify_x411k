import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import io

# --- Constants & Paths ---
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
    'popularity' : 'spotify_stream_counts.parquet',
}

# Modality and Metadata Weights Defaults
MODALITY_WEIGHTS = {'musicnn':3.0, 'spectrogram':2.0, 'nlp':1.5, 'sentiment':0.8, 'colour':0.3, 'clip':0.2, 'resnet':0.2}
METADATA_WEIGHTS = {'genre_sim':0.5, 'popularity':0.15, 'language_sim':0.3, 'danceability':0.2, 'energy':0.2, 'valence':0.2, 'tempo':0.1, 'acousticness':0.1}

PRESETS = {
    'Default': {
        'modality': {'musicnn':3.0,'spectrogram':2.0,'nlp':1.5,'sentiment':0.8,'colour':0.3,'clip':0.2,'resnet':0.2},
        'metadata': {'genre_sim':0.5,'popularity':0.15,'language_sim':0.3,'danceability':0.2,'energy':0.2,'valence':0.2,'tempo':0.1,'acousticness':0.1},
        'serendipity': 0.0,
    },
    '🧭 Discovery': {
        'modality': {'musicnn':3.0,'spectrogram':2.0,'nlp':1.5,'sentiment':0.8,'colour':0.3,'clip':0.2,'resnet':0.2},
        'metadata': {'genre_sim':0.3,'popularity':-1.5,'language_sim':0.3,'danceability':0.2,'energy':0.2,'valence':0.2,'tempo':0.1,'acousticness':0.1},
        'serendipity': 0.0,
    },
    '🎩 Music Snob': {
        'modality': {'musicnn':5.0,'spectrogram':3.5,'nlp':2.0,'sentiment':1.5,'colour':0.0,'clip':0.0,'resnet':0.0},
        'metadata': {'genre_sim':0.8,'popularity':0.0,'language_sim':0.0,'danceability':0.0,'energy':0.0,'valence':0.0,'tempo':0.0,'acousticness':0.0},
        'serendipity': 0.0,
    },
    '🎨 Aesthete': {
        'modality': {'musicnn':1.0,'spectrogram':0.5,'nlp':0.5,'sentiment':0.3,'colour':2.0,'clip':3.0,'resnet':2.5},
        'metadata': {'genre_sim':0.1,'popularity':0.1,'language_sim':0.0,'danceability':0.0,'energy':0.0,'valence':0.0,'tempo':0.0,'acousticness':0.0},
        'serendipity': 0.0,
    },
    '🐦‍⬛ Maverick': {
        'modality': {'musicnn':2.0,'spectrogram':2.0,'nlp':0.5,'sentiment':0.3,'colour':0.1,'clip':0.1,'resnet':0.1},
        'metadata': {'genre_sim':0.0,'popularity':-2.0,'language_sim':0.0,'danceability':0.0,'energy':0.0,'valence':0.0,'tempo':0.0,'acousticness':0.0},
        'serendipity': 0.0,
    },
    '❤️‍🔥 Devotee': {
        'modality': {'musicnn':3.0,'spectrogram':2.5,'nlp':3.5,'sentiment':2.0,'colour':0.5,'clip':0.3,'resnet':0.3},
        'metadata': {'genre_sim':1.0,'popularity':0.5,'language_sim':0.8,'danceability':0.3,'energy':0.3,'valence':0.3,'tempo':0.2,'acousticness':0.1},
        'serendipity': 0.0,
    },
    '📜 Lyricist': {
        'modality': {'musicnn':0.5,'spectrogram':0.3,'nlp':5.0,'sentiment':3.5,'colour':0.0,'clip':0.0,'resnet':0.0},
        'metadata': {'genre_sim':0.4,'popularity':0.0,'language_sim':1.5,'danceability':0.0,'energy':0.0,'valence':0.4,'tempo':0.0,'acousticness':0.0},
        'serendipity': 0.0,
    },
    '🏃‍♂️ Kinetic': {
        'modality': {'musicnn':5.0,'spectrogram':4.0,'nlp':0.0,'sentiment':0.0,'colour':0.0,'clip':0.0,'resnet':0.0},
        'metadata': {'genre_sim':0.0,'popularity':0.0,'language_sim':0.0,'danceability':2.0,'energy':2.5,'valence':0.0,'tempo':2.0,'acousticness':-1.0},
        'serendipity': 0.0,
    },
}

PRESET_DESCRIPTIONS = {
    'Default': 'Balanced default — Audio > NLP > Image.',
    '🧭 Discovery': 'Surface hidden gems: penalises popular tracks, keeps default audio/NLP balance.',
    '🎩 Music Snob': 'Audio purist: maximises MusicNN + Spectrogram, solid NLP, zero image.',
    '🎨 Aesthete': 'Visual vibes: CLIP, colour, ResNet dominate. Lower audio, minimal lyrics.',
    '🐦‍⬛ Maverick': 'Structural outliers via deep audio latent space, strongly penalises popularity.',
    '❤️‍🔥 Devotee': 'Same-artist affinity: high audio + NLP + genre + language matching.',
    '📜 Lyricist': 'Semantic depth & storytelling: prioritises NLP lyric/thematic embeddings, language match and emotional valence. Ignores album art entirely.',
    '🏃‍♂️ Kinetic': 'Energy, tempo & movement: max BPM, energy and danceability. Ignores lyrics and popularity. Built for high-intensity environments and the dance floor.',
}

_COLOUR_MAP = {
    'musicnn':'#1DB954','spectrogram':'#57CC99','nlp':'#FFB347',
    'sentiment':'#FDCB6E','colour':'#FF6B6B','clip':'#FD79A8',
    'resnet':'#A29BFE','metadata':'#74B9FF','popularity':'#74B9FF',
}

_PLOTLY_FONT_FAMILY = "'CMU Sans Serif', 'Latin Modern Sans', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans', sans-serif"


# --- Data Loading ---
@st.cache_resource
def load_data():
    # If lite metadata exists, we are in stlite mode
    meta_path = PATHS['metadata']
    if os.path.exists('assets/metadata_lite.json') and not os.path.exists(meta_path):
        import json
        with open('assets/metadata_lite.json', 'r') as f:
            metadata_df = pl.DataFrame(json.load(f))
        prefix = 'lite_'
    else:
        metadata_df = pl.read_parquet(meta_path)
        prefix = ''

    all_track_ids = metadata_df['track_id'].to_list()
    id_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}
    N = len(all_track_ids)
    
    feature_store = {}
    
    def load_align_pq(key, path, col_prefix=None):
        path = prefix + path if prefix and os.path.exists(prefix + path) else path
        if not os.path.exists(path): return None
        if path.endswith('.npy'):
            # Use mmap_mode='r' to handle massive 411k files without loading everything into RAM at once
            arr = np.load(path, mmap_mode='r')
            if len(arr) != N: return None
            # We return as is to preserve mmap, but keep in mind some operations may copy it into RAM
            return arr
            
        df = pl.read_parquet(path).filter(pl.col('track_id').is_in(all_track_ids))
        if col_prefix:
            f_cols = [c for c in df.columns if c.startswith(col_prefix)]
        else:
            f_cols = [c for c, dt in df.schema.items() if dt in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] and c != 'track_id']
        arr = np.zeros((N, len(f_cols)), dtype=np.float32)
        tids = df['track_id'].to_list()
        mat = df.select(f_cols).to_numpy()
        for i, tid in enumerate(tids): arr[id_to_idx[tid]] = mat[i]
        return arr

    feature_store['musicnn'] = load_align_pq('musicnn', PATHS['musicnn'])
    feature_store['spectrogram'] = load_align_pq('spectrogram', PATHS['spectrogram'])
    feature_store['nlp'] = load_align_pq('nlp', PATHS['nlp'], col_prefix='emb_')
    feature_store['sentiment'] = load_align_pq('sentiment', PATHS['sentiment'])
    feature_store['resnet'] = load_align_pq('resnet', PATHS['resnet'])
    feature_store['clip'] = load_align_pq('clip', PATHS['clip'])
    
    # Load and join image features
    ct_path = prefix + PATHS['ct'] if prefix and os.path.exists(prefix + PATHS['ct']) else PATHS['ct']
    zs_path = prefix + PATHS['zs'] if prefix and os.path.exists(prefix + PATHS['zs']) else PATHS['zs']
    if os.path.exists(ct_path) and os.path.exists(zs_path):
        ct_df = pl.read_parquet(ct_path)
        zs_df = pl.read_parquet(zs_path)
        it_df = ct_df.join(zs_df, on='track_id', how='inner').filter(pl.col('track_id').is_in(all_track_ids))
        f_cols = [c for c, dt in it_df.schema.items() if dt in [pl.Float32, pl.Float64, pl.Int32, pl.Int64] and c != 'track_id']
        ifeats = np.zeros((N, len(f_cols)), dtype=np.float32)
        for r in it_df.select(['track_id']+f_cols).iter_rows(named=True):
            if r['track_id'] in id_to_idx:
                ifeats[id_to_idx[r['track_id']]] = [r[c] for c in f_cols]
        feature_store['colour'] = ifeats
    
    # Simple normalisation
    for k, v in feature_store.items():
        if v is not None:
            norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            feature_store[k] = (v / norms).astype(np.float32)
            
    return metadata_df, feature_store, id_to_idx, N

metadata_df, feature_store, id_to_idx, N = load_data()


# --- Recommendation Logic ---
def _cosine_sim_batch(query_vec, matrix):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    return (matrix @ q.T).ravel()

def _dot_sim_batch(query_vec, matrix):
    return (matrix @ query_vec.T).ravel()

def _euclidean_sim_batch(query_vec, matrix):
    dists = np.linalg.norm(matrix - query_vec, axis=1)
    return 1.0 / (1.0 + dists)

def _manhattan_sim_batch(query_vec, matrix):
    dists = np.sum(np.abs(matrix - query_vec), axis=1)
    return 1.0 / (1.0 + dists)

def _mmr(query_sims, sim_matrix, alpha, k):
    selected, remaining = [], list(range(len(query_sims)))
    for _ in range(k):
        if not remaining: break
        if not selected:
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            best = max(remaining, key=lambda i: (1 - alpha) * query_sims[i] - alpha * max(sim_matrix[i, s] for s in selected))
        selected.append(best)
        remaining.remove(best)
    return selected

def _metadata_sim(picked_idx, meta_weights):
    meta_sim = np.zeros(N, dtype=np.float64)
    total_w  = 0.0
    row_seed = metadata_df.row(int(picked_idx), named=True)

    w_genre = meta_weights.get('genre_sim', 0.0)
    if w_genre != 0 and 'genre' in metadata_df.columns:
        seed_genre = row_seed.get('genre', '')
        genres = metadata_df['genre'].to_list()
        genre_match = np.array([1.0 if g == seed_genre else 0.0 for g in genres], dtype=np.float64)
        meta_sim += genre_match * w_genre
        total_w += abs(w_genre)

    scalar_feats = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
    for feat in scalar_feats:
        w_f = meta_weights.get(feat, 0.0)
        if w_f == 0 or feat not in metadata_df.columns: continue
        seed_val = row_seed.get(feat)
        if seed_val is None: continue
        vals = metadata_df[feat].fill_null(seed_val).to_numpy().astype(np.float64)
        max_range = vals.max() - vals.min() + 1e-9
        meta_sim += (1.0 - np.abs(vals - seed_val) / max_range) * w_f
        total_w += abs(w_f)

    w_pop = meta_weights.get('popularity', 0.0)
    if w_pop != 0 and 'popularity' in metadata_df.columns:
        pop_vals = metadata_df['popularity'].fill_null(0).to_numpy().astype(np.float64)
        meta_sim += (pop_vals / (pop_vals.max() + 1e-9)) * w_pop
        total_w += abs(w_pop)

    if total_w > 0:
        meta_sim /= total_w
    return meta_sim

def get_recommendations(seed_idx, top_n, weights, meta_weights, metric='cosine', temp=1.0, alpha=0.0, min_score=0.0, seren=0.0):
    combined_sim = np.zeros(N, dtype=np.float64)
    total_weight = 0.0
    per_mod_sims = {}

    sim_fns = {'cosine': _cosine_sim_batch, 'dot': _dot_sim_batch, 'euclidean': _euclidean_sim_batch, 'manhattan': _manhattan_sim_batch}
    sim_fn = sim_fns.get(metric, _cosine_sim_batch)

    for k, v in feature_store.items():
        w = weights.get(k, 0.0)
        if w == 0 or v is None: continue
        q = v[seed_idx : seed_idx + 1]
        if np.linalg.norm(q) < 1e-9: continue
        sims = np.nan_to_num(sim_fn(q, v).astype(np.float64), nan=0.0)
        per_mod_sims[k] = sims
        combined_sim += sims * w
        total_weight += abs(w)

    meta_total_w = sum(abs(v) for v in meta_weights.values())
    if meta_total_w > 0:
        meta_sim = _metadata_sim(seed_idx, meta_weights)
        per_mod_sims['metadata'] = meta_sim
        combined_sim += meta_sim * meta_total_w
        total_weight += meta_total_w

    if total_weight > 0: combined_sim /= total_weight
    if temp != 1.0 and temp > 0: combined_sim = combined_sim ** (1.0 / temp)

    candidate_mask = (combined_sim >= min_score)
    candidate_mask[seed_idx] = False
    cand_indices = np.where(candidate_mask)[0]

    if seren > 0 and len(cand_indices) > 0:
        n_seren = max(1, int(top_n * seren))
        n_normal = top_n - n_seren
        cand_sims = combined_sim[cand_indices]
        order_desc = np.argsort(cand_sims)[::-1]
        normal_pool = cand_indices[order_desc[:max(n_normal, 1)]]
        
        order_asc = np.argsort(cand_sims)
        seren_pool_size = min(len(cand_indices) // 2, max(n_seren * 5, 50))
        seren_pool_idx = cand_indices[order_asc[:seren_pool_size]]
        seren_chosen = np.random.choice(seren_pool_idx, size=n_seren, replace=False) if len(seren_pool_idx) > n_seren else seren_pool_idx
        
        top_indices = np.unique(np.concatenate([normal_pool[:n_normal], seren_chosen]))[:top_n]
        scores = combined_sim[top_indices]
        sort_order = np.argsort(scores)[::-1]
        return top_indices[sort_order], scores[sort_order], per_mod_sims

    elif alpha > 0 and len(cand_indices) > 0:
        cand_sims = combined_sim[cand_indices]
        audio_key = 'musicnn' if 'musicnn' in feature_store and feature_store['musicnn'] is not None else list(feature_store.keys())[0]
        a_norm = feature_store[audio_key][cand_indices]
        sim_matrix = a_norm @ a_norm.T
        selected_local = _mmr(cand_sims, sim_matrix, alpha, top_n)
        top_indices = cand_indices[selected_local]
        return top_indices, combined_sim[top_indices], per_mod_sims

    order = np.argsort(combined_sim[cand_indices])[::-1][:top_n]
    top_indices = cand_indices[order]
    return top_indices, combined_sim[top_indices], per_mod_sims


# --- App State Management ---
if 'seed_idx' not in st.session_state: st.session_state.seed_idx = None
if 'page' not in st.session_state: st.session_state.page = 0
if 'search_page' not in st.session_state: st.session_state.search_page = 0
if 'last_search' not in st.session_state: st.session_state.last_search = ""
# Background colors for glassmorphism (darker, lighter)
if 'bg_colors' not in st.session_state: st.session_state.bg_colors = ("#f5f7fa", "#c3cfe2")

# --- Shared Utilities ---
def _get_album_cover_url(row, target_size=300):
    if not isinstance(row, dict):
        try: row = dict(row)
        except: return ""
    album = row.get('album')
    if not album or 'images' not in album or not album['images']: return ""
    # album['images'] might be a string if not properly deserialized, let's play it safe
    if isinstance(album['images'], str): return "" 
    valid_imgs = [img for img in album['images'] if isinstance(img, dict) and img.get('url')]
    if not valid_imgs: return ""
    best = min(valid_imgs, key=lambda img: abs((img.get('height') or 0) - target_size))
    return best.get('url', "")

# --- Streamlit UI Configurations ---
st.set_page_config(page_title="Multimodal Discovery", layout="wide", page_icon="🎵", initial_sidebar_state="expanded")

# Inject Font and Dynamic Background CSS
st.markdown(
    f"""
    <style>
    /* Apply Inter surgically only to recognized text containers */
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stMetric, [data-testid="stText"], 
    .stSelectbox label, .stSlider label, .stButton p, .stDownloadButton p,
    .stCaption, div[data-testid="stExpander"] p {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }}

    /* Explicitly prevent font inheritance for Streamlit's icon system */
    [data-testid="stIcon"], .stIcon, svg, i, 
    [font-family*="Material"], [class*="Icon"], [class*="Symbol"] {{
        font-family: inherit !important;
    }}
    
    html, body {{
        background: linear-gradient(135deg, {st.session_state.bg_colors[0]} 0%, {st.session_state.bg_colors[1]} 100%);
        background-attachment: fixed;
    }}

    /* Glassmorphism Containers - Compact for Search */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        padding: 0.8rem !important;
    }}

    /* Translucent Plotly Charts */
    .js-plotly-plot .plotly .main-svg {{
        background: transparent !important;
    }}
    .js-plotly-plot .plotly .bg {{
        fill: transparent !important;
    }}

    /* Translucent Dataframes */
    [data-testid="stDataFrame"] {{
        background: transparent !important;
    }}
    [data-testid="stDataFrame"] div {{
        background-color: transparent !important;
    }}

    /* Prettier Buttons (Apple Style) */
    .stButton > button {{
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(4px);
        color: #333 !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .stButton > button:hover {{
        border-color: #1DB954 !important;
        background: rgba(29, 185, 84, 0.1) !important;
        color: #1DB954 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    div.stMarkdown div[style*="text-align: center"] {{
        font-weight: 600;
        color: #555;
        letter-spacing: 0.5px;
    }}

    .genre-tag {{ 
        background: rgba(29, 185, 84, 0.15); 
        color: #1DB954; 
        font-size: 10px; 
        padding: 3px 10px; 
        border-radius: 20px; 
        font-weight: 600; 
        border: 1px solid rgba(29, 185, 84, 0.3);
    }}
    
    .score-badge {{ color: #1DB954; font-size: 14px; font-weight: 700; }}
    
    .progress-bg {{ background-color: rgba(0,0,0,0.05); border-radius: 10px; width: 100%; height: 8px; margin-top: 4px; }}
    .progress-fill {{ border-radius: 10px; height: 100%; transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1); }}
    
    [data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.3) !important;
        backdrop-filter: blur(10px);
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# --- Plotting Functions ---
def plot_pca(seed_idx, cand_indices, cand_scores, per_mod_sims):
    vecs = feature_store.get('musicnn', feature_store.get(list(feature_store.keys())[0]))
    if vecs is None: return None
    seed_idx = int(seed_idx)
    cand_indices = [int(i) for i in cand_indices]
    
    all_idx = list(cand_indices) + [seed_idx]
    if len(all_idx) < 4: return None
    
    pca = PCA(n_components=3)
    proj = pca.fit_transform(vecs[all_idx])
    rec_proj = proj[:-1]
    seed_proj = proj[-1]
    
    hover_texts = [f"{metadata_df['name'][i]} - {metadata_df['artist_name'][i]}" for i in cand_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=rec_proj[:, 0], y=rec_proj[:, 1], z=rec_proj[:, 2],
        mode='markers+text',
        marker=dict(size=7, color=cand_scores, colorscale='YlGn', cmin=0, cmax=1, opacity=0.9, line=dict(width=0.5, color='white')),
        text=[f'{j+1}' for j in range(len(cand_indices))],
        textposition='top center',
        textfont=dict(size=9, color='#ddd'),
        name='Recommendations',
        customdata=hover_texts,
        hovertemplate='<b>%{customdata}</b><br>Score: %{marker.color:.4f}<extra></extra>'
    ))
    
    seed_name = metadata_df['name'][seed_idx]
    fig.add_trace(go.Scatter3d(
        x=[seed_proj[0]], y=[seed_proj[1]], z=[seed_proj[2]],
        mode='markers+text',
        marker=dict(size=12, color='#FF4444', symbol='diamond', opacity=1.0, line=dict(width=1, color='white')),
        text=[f'⭑ Seed'],
        textposition='top center',
        textfont=dict(size=10, color='#FF4444'),
        name='Seed',
        hovertemplate=f'<b>{seed_name}</b><extra></extra>',
    ))
    
    fig.update_layout(
        title=dict(text=f"3D PCA Audio Latent Space<br><sup>{pca.explained_variance_ratio_.sum()*100:.1f}% variance</sup>", font=dict(family="Inter, sans-serif", color='black')),
        scene=dict(xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#DDD'), yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#DDD'), zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='#DDD')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter, sans-serif", color='black'),
        margin=dict(l=0, r=0, b=0, t=50), height=500
    )
    return fig

def plot_modality_breakdown(seed_idx, indices, scores, per_mod_sims, weights, total_weight):
    seed_idx = int(seed_idx)
    indices = [int(i) for i in indices]
    mods = [k for k in per_mod_sims.keys() if k != 'metadata']
    short_names = [metadata_df['name'][i][:16] for i in indices]
    
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], specs=[[{'type': 'bar'}, {'type': 'polar'}]], subplot_titles=('Modality Breakdown', 'Radar: Seed vs Top Match'))
    
    for mod in mods:
        w = weights.get(mod, 0)
        contrib = [float(per_mod_sims[mod][i]) * w / (total_weight + 1e-9) for i in indices]
        fig.add_trace(go.Bar(name=mod, x=short_names, y=contrib, marker_color=_COLOUR_MAP.get(mod, '#aaa')), row=1, col=1)
        
    seed_vals = [float(per_mod_sims[m][seed_idx]) for m in mods]
    top1_vals = [float(per_mod_sims[m][indices[0]]) for m in mods]
    
    fig.add_trace(go.Scatterpolar(r=seed_vals + [seed_vals[0]], theta=mods + [mods[0]], fill='toself', name='Seed', line_color='#E53935'), row=1, col=2)
    fig.add_trace(go.Scatterpolar(r=top1_vals + [top1_vals[0]], theta=mods + [mods[0]], fill='toself', name='Top Match', line_color='#1DB954'), row=1, col=2)
    
    fig.update_layout(barmode='stack', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter, sans-serif", color='black'), polar=dict(bgcolor='rgba(255,255,255,0.2)', radialaxis=dict(gridcolor='#DDD')), margin=dict(t=50, b=50))
    fig.update_xaxes(tickangle=-35)
    return fig


# --- App State & Sidebar (Pre-load) ---
# None of these need to be inside with st.sidebar
if 'seed_idx' not in st.session_state: st.session_state.seed_idx = None
if 'page' not in st.session_state: st.session_state.page = 0
if 'search_page' not in st.session_state: st.session_state.search_page = 0
if 'last_search' not in st.session_state: st.session_state.last_search = ""

with st.sidebar:
    st.header("⚙️ Settings & Tuning")
    preset_name = st.selectbox("Presets", list(PRESETS.keys()))
    p = PRESETS[preset_name]
    st.caption(PRESET_DESCRIPTIONS.get(preset_name, ""))
    
    with st.expander("Modality Weights", expanded=True):
        w_audio = st.slider("Audio (MusicNN)", -5.0, 5.0, p['modality'].get('musicnn', 3.0), help="MusicNN: Deep audio embedding (200-d) trained on music tags. Captures timbral texture, genre, and high-level sonic character. The primary signal — keep this high.")
        w_spec  = st.slider("Audio (Spectrogram)", -5.0, 5.0, p['modality'].get('spectrogram', 2.0), help="Spectrogram: Raw mel-spectrogram features capturing frequency content over time. Complements MusicNN with low-level audio structure like rhythm and harmonics.")
        w_nlp   = st.slider("Lyrics (NLP)", -5.0, 5.0, p['modality'].get('nlp', 1.5), help="NLP / Lyrics: Multilingual sentence embeddings of lyrics + genre text. Captures thematic similarity, poetic style, and semantic meaning across languages.")
        w_sent  = st.slider("Lyrics (Sentiment)", -5.0, 5.0, p['modality'].get('sentiment', 0.8), help="Sentiment: Emotional tone matrix derived from lyric analysis. Captures mood dimensions like positivity, aggression, melancholy. Complements NLP.")
        w_col   = st.slider("Image (Colour)", -5.0, 5.0, p['modality'].get('colour', 0.3), help="Colour Palette: K-means colour features from album art. Low-level visual proxy for aesthetic/era. Keep weight low — art rarely encodes musical content.")
        w_clip  = st.slider("Image (CLIP)", -5.0, 5.0, p['modality'].get('clip', 0.2), help="CLIP Visual: OpenAI CLIP embeddings of album art. Richer visual semantics than raw colour. Useful for The Aesthete preset — art-driven recommendations.")
        w_res   = st.slider("Image (ResNet)", -5.0, 5.0, p['modality'].get('resnet', 0.2), help="ResNet Image: Deep ResNet-50 image embeddings of album covers. Captures compositional and stylistic visual features. Auxiliary signal only.")

    with st.expander("Metadata Weights"):
        w_genre = st.slider("Genre Match", -5.0, 5.0, p['metadata'].get('genre_sim', 0.5), help="Genre Similarity: Exact-match bonus for same genre. High values keep recs genre-locked; negative values push cross-genre discovery.")
        w_pop   = st.slider("Popularity Boost", -5.0, 5.0, p['metadata'].get('popularity', 0.15), help="Popularity: Boosts (positive) or penalises (negative) mainstream tracks. Set negative to surface obscure gems; set positive for chart-friendly picks.")
        w_lang  = st.slider("Language Match", -5.0, 5.0, p['metadata'].get('language_sim', 0.3), help="Language Similarity: Rewards tracks in the same language as the seed. Set negative (Globetrotter) to surface international interpretations.")
        w_dance = st.slider("Danceability", -5.0, 5.0, p['metadata'].get('danceability', 0.2), help="Danceability: Spotify feature [0–1] measuring rhythmic suitability for dancing. High = prioritise similar groove/rhythm.")
        w_energy= st.slider("Energy", -5.0, 5.0, p['metadata'].get('energy', 0.2), help="Energy: Spotify perceptual intensity [0–1] — loudness, tempo, activity. High = match energy level; useful for workout/party playlists.")

    with st.expander("Search & Pipeline Configuration"):
        sim_metric = st.selectbox("Metric", ['cosine', 'dot', 'euclidean', 'manhattan'], index=0)
        max_recs   = st.number_input("Max Recommendations Pool", min_value=10, max_value=300, value=100)
        alpha      = st.slider("MMR Diversity (Alpha)", 0.0, 1.0, 0.0, help="0 = pure similarity, higher = more diverse.")
        seren      = st.slider("Serendipity", 0.0, 1.0, p['serendipity'], help="Injects distant neighbours for discovery.")
        temp       = st.slider("Temperature", 0.1, 3.0, 1.0)
        show_charts = st.checkbox("Show Breakdown Charts", value=True)

    weights = {'musicnn': w_audio, 'spectrogram': w_spec, 'nlp': w_nlp, 'sentiment': w_sent, 'colour': w_col, 'clip': w_clip, 'resnet': w_res}
    meta_weights = {'genre_sim': w_genre, 'popularity': w_pop, 'language_sim': w_lang, 'danceability': w_dance, 'energy': w_energy}

st.title("🎵 Multimodal Discovery Engine")
col1, col2 = st.columns([3, 1])

with col1:
    search_q = st.text_input("Search for a song or artist...", placeholder="e.g. Radiohead 411k")

# Reset search page if query changes
if search_q != st.session_state.last_search:
    st.session_state.last_search = search_q
    st.session_state.search_page = 0
    st.session_state.bg_colors = ("#f5f7fa", "#c3cfe2") # Revert to neutral on search

if search_q:
    results = metadata_df.filter((pl.col('name').str.to_lowercase().str.contains(search_q.lower())) | (pl.col('artist_name').str.to_lowercase().str.contains(search_q.lower()))).to_dicts()
    
    if len(results) > 0:
        s_page = st.session_state.search_page
        s_per_page = 10
        total_s_pages = max(1, (len(results) + s_per_page - 1) // s_per_page)
        
        st.write(f"Found {len(results)} matches (Page {s_page+1}/{total_s_pages})")
        for res in results[s_page*s_per_page : (s_page+1)*s_per_page]:
            idx = id_to_idx.get(res['track_id'])
            if idx is not None:
                row = metadata_df.row(idx, named=True)
                img_url = _get_album_cover_url(row, 300)
                
                with st.container(border=True):
                    c_img, c_data, c_btn = st.columns([1.5, 3.5, 4])
                    with c_img:
                        if img_url: st.image(img_url, use_container_width=True)
                    with c_data:
                        st.markdown(f"<b>{row['name']}</b><br><span style='color:#555;'>{row['artist_name']}</span> <span class='genre-tag' style='margin-left:5px;'>{row.get('genre', '')}</span>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size:12px; color:#555; margin-top:2px;'>Popularity: <b>{(row.get('popularity') or 0)}</b> | Streams: <b>{(row.get('stream_count') or 0):,}</b></div>", unsafe_allow_html=True)
                        if row.get('preview_url'): st.audio(row['preview_url'])
                    with c_btn:
                        with st.expander("Metrics", expanded=False):
                            m_cols = st.columns(2)
                            m_cols[0].metric("Dance", f"{row.get('danceability', 0):.2f}")
                            m_cols[1].metric("Energy", f"{row.get('energy', 0):.2f}")
                            m_cols2 = st.columns(2)
                            m_cols2[0].metric("Valence", f"{row.get('valence', 0):.2f}")
                            m_cols2[1].metric("Tempo", f"{int(row.get('tempo', 0))}")

                        st.write("")
                        if st.button("✨ Select", key=f"s_{res['track_id']}", use_container_width=True):
                            st.session_state.seed_idx = idx
                            st.session_state.page = 0
                            # Extract background colors
                            if 'colour' in feature_store and feature_store['colour'] is not None:
                                c_vec = feature_store['colour'][idx]
                                r, g, b = int(c_vec[0] * 255), int(c_vec[5] * 255), int(c_vec[10] * 255)
                                # Create a lighter/softer version for background
                                st.session_state.bg_colors = (f"rgb({int(r*0.2 + 200)}, {int(g*0.2 + 200)}, {int(b*0.2 + 200)})", 
                                                              f"rgb({int(r*0.5 + 120)}, {int(g*0.5 + 120)}, {int(b*0.5 + 120)})")
                            st.rerun()
                    
        c1, c2, c3 = st.columns([1,2,1])
        if c1.button("⬅️ Previous", key="sprev", disabled=s_page==0):
            st.session_state.search_page -= 1; st.rerun()
        if c3.button("Next ➡️", key="snext", disabled=s_page>=total_s_pages-1):
            st.session_state.search_page += 1; st.rerun()

st.divider()

# --- Recommendations Render ---
if st.session_state.seed_idx is not None:
    seed_idx = st.session_state.seed_idx
    seed_row = metadata_df.row(seed_idx, named=True)
    st.markdown(f"### Recommendations for: **{seed_row['name']}** by **{seed_row['artist_name']}**")
    
    with st.spinner("Calculating 411k similarities..."):
        indices, scores, per_mod = get_recommendations(
            seed_idx, max_recs, weights, meta_weights, metric=sim_metric, temp=temp, alpha=alpha, seren=seren
        )
    
    r_page = st.session_state.page
    r_per_page = 10
    total_r_pages = max(1, (len(indices) + r_per_page - 1) // r_per_page)
    chunk_idx = indices[r_page*r_per_page : (r_page+1)*r_per_page]
    chunk_scores = scores[r_page*r_per_page : (r_page+1)*r_per_page]

    total_w = sum(abs(v) for v in weights.values()) + sum(abs(v) for v in meta_weights.values())

    if show_charts and len(chunk_idx) > 0:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_pca(seed_idx, chunk_idx, chunk_scores, per_mod), use_container_width=True)
        with c2: st.plotly_chart(plot_modality_breakdown(seed_idx, chunk_idx, chunk_scores, per_mod, weights, total_w), use_container_width=True)

    c1, c2, c3 = st.columns([4.4, 1.3, 1.3])
    with c2:
        if st.download_button("📥 Export CSV", pd.DataFrame({'track_id': [metadata_df['track_id'][int(i)] for i in indices], 'name': [metadata_df['name'][int(i)] for i in indices], 'score': scores}).to_csv(index=False), file_name="recommendations.csv", use_container_width=True):
            st.toast("Exported to CSV!")
    with c3:
        with st.popover("📋 Copy Spotify URIs", use_container_width=True):
            st.markdown("Click upper-right corner of code block to copy all:")
            all_uris = [f"spotify:track:{metadata_df['track_id'][int(i)]}" for i in indices]
            st.code("\n".join(all_uris), language='text')

    # Pagination controls
    pc1, pc2, pc3 = st.columns([1,3,1])
    with pc1:
        if st.button("⬅️ Previous Recs", disabled=r_page==0, use_container_width=True): 
            st.session_state.page -= 1; st.rerun()
    pc2.markdown(f"<div style='text-align: center; line-height:38px;'>Page {r_page+1} / {total_r_pages}</div>", unsafe_allow_html=True)
    with pc3:
        # Pushing this button to the right of its column is natural with use_container_width
        if st.button("Next Recs ➡️", disabled=r_page>=total_r_pages-1, use_container_width=True): 
            st.session_state.page += 1; st.rerun()

    spotify_uris = [f"spotify:track:{metadata_df['track_id'][int(i)]}" for i in chunk_idx]
    
    for rank, (idx, score) in enumerate(zip(chunk_idx, chunk_scores)):
        idx = int(idx)
        row = metadata_df.row(idx, named=True)
        img_url = _get_album_cover_url(row, 300)
        
        # Calculate sub-scores mapping
        subscores = {"Audio": 0.0, "Word": 0.0, "Image": 0.0, "Meta": 0.0}
        df_breakdown = []
        for mod in per_mod.keys():
            raw = float(per_mod[mod][idx])
            mw = sum(abs(v) for v in meta_weights.values()) if mod == 'metadata' else weights.get(mod, 0)
            contrib = raw * mw / (total_w + 1e-9)
            pct = max(0, min(100, abs(contrib) / (abs(score) + 1e-9) * 100))
            if mw > 0 or mod == 'metadata':
                df_breakdown.append({"Modality": mod, "Raw": raw, "W": mw, "Contrib": contrib, "Impact": pct})
                if mod in ['musicnn', 'spectrogram']: subscores['Audio'] += contrib
                elif mod in ['nlp', 'sentiment']: subscores['Word'] += contrib
                elif mod in ['colour', 'clip', 'resnet']: subscores['Image'] += contrib
                else: subscores['Meta'] += contrib
        
        with st.container(border=True):
            col_img, col_data, col_breakdown = st.columns([1.5, 3.5, 4])
            
            with col_img:
                if img_url: 
                    st.image(img_url, use_container_width=True)
            
            with col_data:
                st.markdown(f"""
                <div style="font-size: 18px; line-height: 1.2;">
                    <b>{r_page*r_per_page + rank + 1}. {row['name']}</b>
                </div>
                <div style="font-size: 15px; color:#555; margin-bottom: 5px;">
                    {row['artist_name']}
                </div>
                <span class="genre-tag" style="margin-left: 0;">{row.get('genre', '')}</span>
                <br><br>
                <div class="score-badge">Combined Score: {score:.4f}</div>
                <div style="font-size: 13px; margin-top: 3px; color: #444;">
                    <b>Audio:</b> {subscores['Audio']:.3f} &nbsp;|&nbsp; 
                    <b>Word:</b> {subscores['Word']:.3f} &nbsp;|&nbsp; 
                    <b>Image:</b> {subscores['Image']:.3f} &nbsp;|&nbsp; 
                    <b>Meta:</b> {subscores['Meta']:.3f}
                </div>
                <div style="font-size: 13px; color:#aaa; margin-top: 5px;">
                    <b>URI</b>: <code>spotify:track:{row['track_id']}</code><br>
                    <b>Popularity</b>: {row.get('popularity', 'N/A')} | <b>Lang</b>: {row.get('language', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
                
                if row.get('preview_url'):
                    st.audio(row['preview_url'])
                    
                if row.get('lyrics') and len(row['lyrics']) > 10:
                    with st.expander("Show Lyrics"):
                        st.markdown(row['lyrics'].replace('\n', '<br>'), unsafe_allow_html=True)
            
            with col_breakdown:
                st.markdown("**Modality Breakdown**")
                
                b_df = pd.DataFrame(df_breakdown)
                if not b_df.empty:
                    # Sort by impact
                    b_df = b_df.sort_values('Contrib', ascending=False)
                    st.dataframe(
                        b_df,
                        column_config={
                            "Modality": st.column_config.TextColumn("Modality"),
                            "Raw": st.column_config.NumberColumn("Raw", format="%.3f"),
                            "W": st.column_config.NumberColumn("W", format="%.1f"),
                            "Contrib": st.column_config.NumberColumn("Contrib", format="%.3f"),
                            "Impact": st.column_config.ProgressColumn("Impact", format="%.1f%%", min_value=0, max_value=100)
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=200
                    )

