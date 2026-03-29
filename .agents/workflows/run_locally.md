---
description: How to run the song discovery engine on localhost
---

To run the site locally and experience the full 411k song search as an interactive application, follow these steps in your terminal:

### 1. Install Dependencies
Ensure you have Streamlit installed.
// turbo
```bash
pip install streamlit
```

### 2. Run the Streamlit Application
This will start the discovery engine with full 411k track search capability.

// turbo
```bash
streamlit run app.py
```

### 3. Open in Browser
Visit the following URL to see your app live on localhost:
[http://localhost:8501](http://localhost:8501)

> [!NOTE]
> The app will load the 30MB local database into memory upon the first run. Subsequent interactions and searches across the 411,000 tracks are extremely fast.
