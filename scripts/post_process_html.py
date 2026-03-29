import os
import sys

def post_process_html(html_path):
    if not os.path.exists(html_path):
        print(f"Error: {html_path} not found.")
        return

    print(f"Post-processing {html_path}...")
    with open(html_path, 'r') as f:
        html = f.read()

    # 1. Inject Styles
    if os.path.exists('assets/style.css'):
        with open('assets/style.css', 'r') as f:
            css = f.read()
        style_tag = f"<style>\n{css}\n</style>"
        html = html.replace('</head>', f'{style_tag}\n</head>')
    
    # 2. Inject Navbar and Search Results container
    # Since we want it at the top of the body
    navbar_html = """
<header class="nav-bar">
    <div class="nav-logo">🎵 Multimodal Discovery</div>
    <div class="nav-search-container">
        <svg height="16" width="16" viewBox="0 0 16 16" fill="white"><path d="M7 1.75a5.25 5.25 0 102.828 9.688l4.192 4.192a.75.75 0 101.06-1.06l-4.19-4.19A5.25 5.25 0 007 1.75zM3.25 7a3.75 3.75 0 117.5 0 3.75 3.75 0 01-7.5 0z"></path></svg>
        <input type="text" placeholder="Search 411,000 tracks...">
        <div id="search-results"></div>
    </div>
    <div style="display: flex; align-items: center; gap: 20px;">
        <div id="db-status-pill" class="searching-pulse" style="font-size: 10px; color: var(--text-muted); background: var(--glass-border); padding: 4px 12px; border-radius: 50px;">
            Initializing index...
        </div>
        <a href="https://github.com/martin-he543/spotify_x411k" style="color: white; text-decoration: none; font-size: 14px;">GitHub</a>
        <button onclick="window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})" style="padding: 8px 16px !important; font-size: 12px !important; border-radius: 50px; background: #1DB954; cursor: pointer;">Launch</button>
    </div>
</header>
"""
    # 3. Inject Search Logic
    search_script = """
<script>
document.addEventListener('DOMContentLoaded', () => {
    let db = null;
    const searchInput = document.querySelector('.nav-search-container input');
    const resultsDiv = document.getElementById('search-results');
    const statusPill = document.getElementById('db-status-pill');

    console.log("Fetching track database...");
    fetch('assets/metadata_full.json')
        .then(response => response.json())
        .then(data => {
            db = data;
            statusPill.textContent = "411k Tracks Live";
            statusPill.classList.remove('searching-pulse');
            console.log("Database loaded: 411,000 tracks.");
        })
        .catch(err => {
            console.error("Failed to load database", err);
            statusPill.textContent = "Offline Mode";
        });

    let debounceTimer;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        const query = e.target.value.toLowerCase().trim();
        
        if (query.length < 2) {
            resultsDiv.style.display = 'none';
            return;
        }

        debounceTimer = setTimeout(() => {
            if (!db) return;
            const matches = [];
            const maxResults = 50;

            for (let i = 0; i < db.names.length; i++) {
                if (db.names[i].toLowerCase().includes(query) || 
                    db.artists[i].toLowerCase().includes(query)) {
                    matches.push({
                        name: db.names[i],
                        artist: db.artists[i],
                        genre: db.genres[i],
                        id: db.ids[i]
                    });
                    if (matches.length >= maxResults) break;
                }
            }
            renderResults(matches);
        }, 150);
    });

    function renderResults(matches) {
        if (matches.length === 0) {
            resultsDiv.innerHTML = '<div class="search-item"><span class="track-name">No matches found</span></div>';
        } else {
            resultsDiv.innerHTML = matches.map(m => `
                <div class="search-item" onclick="window.open('https://open.spotify.com/track/${m.id}', '_blank')">
                    <span class="track-name">${m.name} <span class="genre-tag">${m.genre}</span></span>
                    <span class="artist-name">${m.artist}</span>
                </div>
            `).join('');
        }
        resultsDiv.style.display = 'block';
    }

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.nav-search-container')) {
            resultsDiv.style.display = 'none';
        }
    });
});
</script>
"""
    # Find the body start and inject
    body_start = html.find('<body')
    if body_start != -1:
        # Move past the <body ...> tag
        body_tag_end = html.find('>', body_start) + 1
        html = html[:body_tag_end] + navbar_html + html[body_tag_end:]
    
    # Inject search script at the end of body
    html = html.replace('</body>', f'{search_script}\n</body>')

    # Save processed HTML
    with open(html_path, 'w') as f:
        f.write(html)
    
    print(f"Successfully processed {html_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        post_process_html(sys.argv[1])
    else:
        post_process_html('index.html')
