import nbformat as nbf
import os

path = 'multimodal_v9g.ipynb'
if not os.path.exists(path):
    print(f"Error: {path} not found.")
    exit(1)

with open(path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

print("Updating notebook structure...")

# 1. Update PATHS
for cell in nb.cells:
    if 'PATHS = {' in cell.source:
        if "'languages'  :" not in cell.source:
            cell.source = cell.source.replace(
                "'popularity' : 'spotify_stream_counts.parquet',",
                "'popularity' : 'spotify_stream_counts.parquet',\n    'languages'  : 'languages.parquet',"
            )
            print(" - Updated PATHS")
        break

# 2. Update Data Loading (join languages)
for cell in nb.cells:
    if 'metadata_df = pl.read_parquet(PATHS[\'metadata\'])' in cell.source:
        if "PATHS['languages']" not in cell.source:
            insertion_point = "metadata_df = metadata_df.select(available_cols)"
            new_code = (
                "metadata_df = metadata_df.select(available_cols)\n"
                "print(\"Loading language metadata...\")\n"
                "if os.path.exists(PATHS.get('languages', '')):\n"
                "    lang_df = pl.read_parquet(PATHS['languages'])\n"
                "    metadata_df = metadata_df.join(lang_df, on='track_id', how='left').with_columns(pl.col('language').fill_null('unknown'))\n"
                "    print(f\"  Languages loaded: {metadata_df['language'].n_unique()} unique values\")\n"
                "else:\n"
                "    print(\"  Warning: languages.parquet not found.\")"
            )
            cell.source = cell.source.replace(insertion_point, new_code)
            print(" - Updated Data Loading")
        break

# 3. Update create_metadata_accordion
for cell in nb.cells:
    if 'def create_metadata_accordion' in cell.source:
        if 'Lang:' not in cell.source:
            cell.source = cell.source.replace(
                "f\"<td><b>Genre:</b> {row_meta.get('genre','N/A')}</td></tr>\"",
                "f\"<td><b>Genre:</b> {row_meta.get('genre','N/A')}</td>\"\n"
                "        f\"<td><b>Lang:</b> {row_meta.get('language','unknown')}</td></tr>\""
            )
            print(" - Updated Metadata Accordion")
        break

# 4. Update Search UI and Logic
for cell in nb.cells:
    if 'search_input   = widgets.Text' in cell.source:
        if 'lang_filter' not in cell.source:
            # Add filter widget
            cell.source = cell.source.replace(
                "search_input   = widgets.Text(",
                "lang_filter    = widgets.Dropdown(options=['all'], value='all', description='Lang:', layout=widgets.Layout(width='120px'))\n"
                "search_input   = widgets.Text("
            )
            # Update search_button layout
            cell.source = cell.source.replace(
                "layout=widgets.Layout(width='90px')",
                "layout=widgets.Layout(width='80px')"
            )
            # Update perform_search logic
            cell.source = cell.source.replace(
                "hits = metadata_df.filter(\n        (pl.col('name').str.to_lowercase().str.contains(q)) |\n        (pl.col('artist_name').str.to_lowercase().str.contains(q))\n    )",
                "hits = metadata_df.filter(\n        ((pl.col('name').str.to_lowercase().str.contains(q)) |\n         (pl.col('artist_name').str.to_lowercase().str.contains(q))) &\n        ((pl.col('language') == lang_filter.value) if lang_filter.value != 'all' else True)\n    )"
            )
            # Update final layout
            cell.source = cell.source.replace(
                "search_row        = widgets.HBox([search_input, search_button])",
                "lang_filter.options = ['all'] + sorted(metadata_df['language'].unique().to_list())\n"
                "search_row        = widgets.HBox([lang_filter, search_input, search_button])"
            )
            print(" - Updated Search UI & Logic")
        break

# 5. Update _HELP_TEXTS
for cell in nb.cells:
    if '_HELP_TEXTS = {' in cell.source:
        if 'Now active' not in cell.source:
            cell.source = cell.source.replace(
                "'language_sim': 'Language Similarity: Rewards tracks in the same language as the seed. Set negative (Globetrotter) to surface international interpretations.',",
                "'language_sim': 'Language Similarity: Rewards tracks in the same language as the seed. Set negative (Globetrotter) to surface international interpretations. Now active using language.parquet data.',"
            )
            print(" - Updated Help Texts")
        break

with open(path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("\n✅ Notebook updated successfully.")
