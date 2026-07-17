# 🔬 Worm Counter

Automated detection and manual correction of *C. elegans* worms in microscopy images.
Built for Sun Lab, UAB — Fatty Acids / Lifespan experiments.

---

## Three ways to use this tool

| Method | Best for | Setup |
|--------|----------|-------|
| **Streamlit Cloud** (web) | Sharing with collaborators, remote access | Open a URL — no install |
| **Local Flask** (double-click) | Working with large image folders on your PC | Needs Python once |
| **Standalone .exe / .app** (PyInstaller) | Non-technical users, no Python needed | One-time build |

---

## Option 1 — Streamlit Cloud (free public URL)

### Deploy in 3 steps

1. **Push this repo to GitHub** (or fork it).

2. Go to **[share.streamlit.io](https://share.streamlit.io)** → "New app" → connect your GitHub repo.
   - Main file: `streamlit_app.py`
   - Branch: `main`

3. Click **Deploy** — Streamlit installs `requirements.txt` automatically.

You get a permanent URL like `https://your-lab-worm-counter.streamlit.app` that anyone can open in any browser — no Python, no install.

### Streamlit features
- Upload TIF / PNG / JPG images directly in the browser
- Auto-detection with adjustable parameters
- Click image to add worm markers; click again to remove
- Download annotated images and CSV summary
- Tune parameters automatically from corrected counts

---

## Option 2 — Local Flask app (double-click launcher)

### One-time setup

**macOS:**
```bash
pip3 install opencv-python-headless flask flask-cors numpy Pillow
```

**Windows:**
```cmd
pip install opencv-python-headless flask flask-cors numpy Pillow
```

Or install everything at once:
```bash
pip install -r requirements.txt
```

### Run
- **macOS:** Double-click `Start Worm Counter.command`
- **Windows:** Double-click `Start Worm Counter.bat`

Opens at `http://localhost:8080`. Share on your local network with the IP address shown in the terminal.

### Flask-only features (beyond Streamlit)
- Browse local image folders without uploading
- Interactive canvas: drag to select/detect/remove areas
- ROI circle drag-adjustment on the image
- Grid overlay for systematic manual counting
- YOLO annotation export for deep-learning training
- Full keyboard shortcuts (press `?` in the app)

---

## Option 3 — Standalone executable (PyInstaller)

No Python required on the target machine — useful for distributing to colleagues.

### Build
```bash
pip install pyinstaller
pip install -r requirements.txt
python build_exe.py
```

Creates `dist/WormCounter/`. Zip that folder and send it. Users double-click `WormCounter` (macOS) or `WormCounter.exe` (Windows) — the browser opens automatically.

---

## Keyboard shortcuts (Flask/local version)

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `A` | Add Worm mode | `D` | Run Auto Detect |
| `R` | Remove mode | `S` | Save Result |
| `E` | Remove Area (drag) | `H` | Toggle markers |
| `W` | Detect Area (drag) | `G` | Toggle grid |
| `V` / `Esc` | View mode | `I` | Save image with marks |
| `O` | Adjust ROI | `← ↑` | Previous image |
| `C` | Mark Grid Cells | `→ ↓` | Next image |
| `?` | Show help | | |

---

## Detection algorithm

1. **Plate detection** — Hough circle transform locates the circular plate boundary
2. **Background model** — large Gaussian blur estimates the background
3. **Difference image** — `background − image` highlights darker worm bodies
4. **Threshold + morphology** — binary threshold + morphological closing merges nearby pixels
5. **Contour filtering** — contours outside the `min_area`–`max_area` range are discarded

All five parameters are adjustable in real time. Use **Tune Parameters** to auto-optimize threshold and area bounds from your corrected counts (grid search over all saved results).

---

## File structure

```
Worm Counter App_V1/
├── streamlit_app.py           ← Streamlit web version (Streamlit Cloud entry point)
├── requirements.txt            ← pip dependencies
├── .streamlit/
│   └── config.toml            ← Nature journal color theme
├── build_exe.py                ← PyInstaller standalone build script
├── worm_app/
│   ├── app.py                 ← Flask desktop backend
│   └── templates/
│       └── index.html         ← Full-featured interactive web UI
├── Start Worm Counter.command  ← macOS double-click launcher
└── Start Worm Counter.bat      ← Windows double-click launcher
```

---

## Acknowledgement

If you use this tool in a publication, please acknowledge:

> Worm Counter v1 — automated *C. elegans* detection tool, Sun Lab, University of Alabama at Birmingham.
