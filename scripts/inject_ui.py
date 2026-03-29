import nbformat as nbf
import os

# Path to the notebook
NOTEBOOK_PATH = 'multimodal_v9g.ipynb'

def inject_ui():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    print(f"Reading {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'r') as f:
        ntbk = nbf.read(f, as_version=4)

    # Premium UI Code for Voila
    ui_code = r'''from IPython.display import HTML, display

def load_premium_ui():
    """
    Injects a premium Spotify-inspired UI into the notebook.
    This UI includes the 411k track search engine and custom styling.
    """
    display(HTML("""
        <style>
            /* Reset some Jupyter/Voila defaults */
            body.jp-Notebook { background-color: #121212 !important; color: white !important; }
            .jp-Cell { border: none !important; background: transparent !important; }
            .jp-InputArea-prompt { display: none !important; }
            
            /* Add our custom CSS (linked from assets/style.css) */
            @import url('assets/style.css');

            /* Navbar Layout */
            .nav-bar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 40px;
                background: #000000;
                height: 64px;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 10000;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }

            /* Push content down to compensate for fixed navbar */
            #main-content { margin-top: 80px; }
            
            #live-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 11px;
                color: #B3B3B3;
                background: rgba(255,255,255,0.05);
                padding: 6px 14px;
                border-radius: 50px;
                border: 1px solid rgba(255,255,255,0.1);
            }

            .dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #1DB954;
                box-shadow: 0 0 8px #1DB954;
            }

            .pulse { animation: pulse-kf 1.5s infinite; }
            @keyframes pulse-kf { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
        </style>

        <header class="nav-bar">
            <div class="nav-logo" style="font-weight: 700; color: #1DB954; font-size: 20px;">🎵 Multimodal discovery engine</div>
            <div class="nav-search-container">
                <svg height="16" width="16" viewBox="0 0 16 16" fill="white"><path d="M7 1.75a5.25 5.25 0 102.828 9.688l4.192 4.192a.75.75 0 101.06-1.06l-4.19-4.19A5.25 5.25 0 007 1.75zM3.25 7a3.75 3.75 0 117.5 0 3.75 3.75 0 01-7.5 0z"></path></svg>
                <input type="text" placeholder="Search 411,000 tracks locally...">
                <div id="search-results"></div>
            </div>
            <div style="display: flex; align-items: center; gap: 24px;">
                <div id="live-indicator">
                    <div class="dot pulse" id="status-dot"></div>
                    <span id="status-text">KERNEL: INITIALIZING...</span>
                </div>
                <button onclick="window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})" style="font-size: 12px !important; padding: 10px 20px !important;">Jump to bottom</button>
            </div>
        </header>

        <script>
            // High-Performance Search Logic inside Voila
            (function() {
                const searchInput = document.querySelector('.nav-search-container input');
                const resultsDiv = document.getElementById('search-results');
                const statusText = document.getElementById('status-text');
                const statusDot = document.getElementById('status-dot');
                let db = null;

                console.log("App: Initializing local search...");
                
                fetch('assets/metadata_full.json')
                    .then(r => r.json())
                    .then(data => {
                        db = data;
                        statusText.textContent = "KERNEL: LIVE (411k DB LOADED)";
                        statusDot.classList.remove('pulse');
                        console.log("App: Search database loaded.");
                    })
                    .catch(e => {
                        statusText.textContent = "KERNEL: OFFLINE MODE";
                        statusDot.style.background = "#e74c3c";
                    });

                let timer;
                searchInput.addEventListener('input', (e) => {
                    clearTimeout(timer);
                    const q = e.target.value.toLowerCase().trim();
                    if (q.length < 2) { resultsDiv.style.display = 'none'; return; }
                    
                    timer = setTimeout(() => {
                        if (!db) return;
                        const matches = [];
                        for (let i = 0; i < db.names.length; i++) {
                            if (db.names[i].toLowerCase().includes(q) || db.artists[i].toLowerCase().includes(q)) {
                                matches.push({ n: db.names[i], a: db.artists[i], g: db.genres[i], id: db.ids[i] });
                                if (matches.length >= 40) break;
                            }
                        }
                        resultsDiv.innerHTML = matches.map(m => `
                            <div class="search-item" onclick="window.open('https://open.spotify.com/track/'+m.id, '_blank')">
                                <span class="track-name">${m.n} <span class="genre-tag">${m.g}</span></span>
                                <span class="artist-name">${m.a}</span>
                            </div>
                        `).join('');
                        resultsDiv.style.display = 'block';
                    }, 150);
                });

                document.addEventListener('click', (e) => {
                    if (!e.target.closest('.nav-search-container')) resultsDiv.style.display = 'none';
                });
            })();
        </script>
    """))

# Execute the UI loader
load_premium_ui()
'''
    # Check if UI cell already exists
    cell_exists = False
    for i, cell in enumerate(ntbk.cells):
        if 'load_premium_ui()' in cell.source:
            print(f"Updating existing UI cell at index {i}...")
            cell.source = ui_code
            cell_exists = True
            break
            
    if not cell_exists:
        print("Inserting new UI cell at the top...")
        new_cell = nbf.v4.new_code_cell(ui_code)
        ntbk.cells.insert(0, new_cell)
        
    print(f"Saving modified notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, 'w') as f:
        nbf.write(ntbk, f)
    print("UI injection complete.")

if __name__ == "__main__":
    inject_ui()
