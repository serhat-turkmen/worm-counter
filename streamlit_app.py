"""
Worm Counter — Streamlit web app for C. elegans counting in microscopy images.
Designed to run on Streamlit Community Cloud (free, public URL).

Developed by Serhat Turkmen, PhD
Sun Lab · PI: Dr. HaoSheng Sun
University of Alabama at Birmingham (UAB)

Nature color palette (CMYK → RGB):
  True Blue  #0092EB   Cyan       #29A3CC
  Dark Blue  #0E5881   Red        #D90200
  Green      #3EAA1E   Purple     #6649D0
"""

import io
import csv
import math

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Worm Counter",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Nature-inspired CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif;
}

/* Header */
.nat-header {
    background: linear-gradient(135deg, #0E5881 0%, #0092EB 100%);
    color: white; padding: 14px 20px; border-radius: 8px;
    margin-bottom: 14px;
}
.nat-header h1 { font-size: 1.3rem; font-weight: 700; margin: 0; letter-spacing: 0.4px; }
.nat-header p  { font-size: 0.78rem; opacity: 0.85; margin: 3px 0 0; }

/* Count badge */
.count-badge {
    background: #0E5881; color: white;
    padding: 7px 20px; border-radius: 20px;
    font-size: 1.05rem; font-weight: 700;
    display: inline-block; text-align: center; line-height: 1.35;
}
.count-sub { font-size: 0.62rem; font-weight: 400; opacity: 0.82; display: block; }

/* Mode status pill */
.mode-pill {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
}
.mode-view   { background: #E5E7EB; color: #4B5563; }
.mode-add    { background: #D1FAE5; color: #065F46; }
.mode-remove { background: #FEE2E2; color: #7F1D1D; }

/* Section label */
.sec-label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: #6B7280; margin: 14px 0 6px;
}

/* Result table */
.res-table { width: 100%; border-collapse: collapse; font-size: 0.76rem; }
.res-table th { background: #F3F4F6; padding: 4px 8px; text-align: left; color: #6B7280; font-size: 0.68rem; }
.res-table td { padding: 3px 8px; border-bottom: 1px solid #E5E7EB; }

/* File item row */
.file-row {
    padding: 4px 8px; border-left: 3px solid transparent;
    border-radius: 3px; font-size: 0.78rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.file-row.active { border-left-color: #0092EB; background: #EFF6FF; }
.file-row.saved  { color: #065F46; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── Session-state initialisation ────────────────────────────────────────────
_DEFAULTS = {
    "file_data":    {},      # name → bytes
    "file_names":   [],      # ordered list
    "current_name": None,
    "worms":        [],      # [{x_orig, y_orig, area, manual}]
    "auto_count":   0,
    "saved_results":{},      # name → result dict
    "plate": {"cx": 0, "cy": 0, "cr": 0},
    "orig_w": 0, "orig_h": 0,
    "disp_scale": 1.0,
    "mode": "view",
    "notes": "",
    "last_click": None,
    "show_markers": True,
    "params": {
        "blur_kernel": 51,
        "threshold":   15,
        "min_area":    30.0,
        "max_area":    4000.0,
        "roi_scale":   0.90,
    },
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

ss = st.session_state  # shorthand

# ─── Core image processing ────────────────────────────────────────────────────
def bytes_to_gray(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(io.BytesIO(file_bytes)).convert("L"))
    return img


def bytes_to_pil(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def auto_detect_plate(gray: np.ndarray):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=500,
        param1=50, param2=30,
        minRadius=int(min(h, w) * 0.3),
        maxRadius=int(min(h, w) * 0.6),
    )
    if circles is not None:
        cx, cy, cr = np.round(circles[0, 0]).astype(int)
        return int(cx), int(cy), int(cr)
    return w // 2, h // 2, int(min(h, w) * 0.48)


def detect_worms(gray: np.ndarray, cx: int, cy: int, cr: int, p: dict) -> list:
    roi_scale    = p["roi_scale"]
    blur_kernel  = p["blur_kernel"]
    threshold    = p["threshold"]
    min_area     = p["min_area"]
    max_area     = p["max_area"]

    h, w = gray.shape
    mask_r = int(cr * roi_scale)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), mask_r, 255, -1)

    k = int(blur_kernel)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    blur_img = cv2.GaussianBlur(gray, (k, k), 0)
    diff = cv2.subtract(blur_img, gray)
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    _, thresh_img = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    worms = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            M = cv2.moments(c)
            if M["m00"] > 0:
                px = int(M["m10"] / M["m00"])
                py = int(M["m01"] / M["m00"])
                if 0 <= py < h and 0 <= px < w and mask[py, px] > 0:
                    worms.append({"x_orig": px, "y_orig": py,
                                  "area": float(area), "manual": False})
    return worms


def count_fast(blur_img, gray, cx, cy, cr, p, thresh, min_a, max_a):
    h, w = gray.shape
    mask_r = int(cr * p["roi_scale"])
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), mask_r, 255, -1)
    diff = cv2.subtract(blur_img, gray)
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    _, t = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if min_a <= cv2.contourArea(c) <= max_a)

# ─── Rendering ────────────────────────────────────────────────────────────────
DISPLAY_MAX = 950  # max px for display


def _dashed_circle(draw: ImageDraw.ImageDraw, cx, cy, r, color, width=2, dash=14, gap=7):
    """Draw a dashed circle using short arc segments."""
    if r <= 0:
        return
    total = 2 * math.pi
    a_dash = dash / r
    a_gap  = gap / r
    a = -math.pi / 2
    while a < -math.pi / 2 + total:
        end = min(a + a_dash, -math.pi / 2 + total)
        draw.arc(
            [cx - r, cy - r, cx + r, cy + r],
            start=math.degrees(a),
            end=math.degrees(end),
            fill=color,
            width=width,
        )
        a = end + a_gap


def render_image(file_bytes: bytes, worms: list, plate: dict, params: dict,
                 show_markers: bool = True) -> tuple:
    """Return (PIL Image at display size, scale factor)."""
    pil = bytes_to_pil(file_bytes)
    orig_w, orig_h = pil.size

    scale = min(DISPLAY_MAX / orig_w, DISPLAY_MAX / orig_h, 1.0)
    disp_w = max(1, int(orig_w * scale))
    disp_h = max(1, int(orig_h * scale))
    pil = pil.resize((disp_w, disp_h), Image.LANCZOS)

    draw = ImageDraw.Draw(pil)
    cx  = int(plate["cx"] * scale)
    cy  = int(plate["cy"] * scale)
    cr  = int(plate["cr"] * scale)
    roi_r = max(1, int(cr * params["roi_scale"]))

    # Plate circle (dim blue)
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                 outline=(100, 150, 220), width=1)
    # ROI circle (dashed green)
    _dashed_circle(draw, cx, cy, roi_r, color=(0, 180, 80), width=2)

    if show_markers:
        arm = max(5, int(9 * scale))
        for w in worms:
            wx = int(w["x_orig"] * scale)
            wy = int(w["y_orig"] * scale)
            col = (241, 180, 20) if w.get("manual") else (220, 50, 50)
            draw.line([(wx - arm, wy), (wx + arm, wy)], fill=col, width=2)
            draw.line([(wx, wy - arm), (wx, wy + arm)], fill=col, width=2)
            draw.ellipse([wx - 2, wy - 2, wx + 2, wy + 2], fill=col)

    return pil, scale

# ─── State helpers ────────────────────────────────────────────────────────────
def hit_test(x_orig: int, y_orig: int, radius: int = 18) -> int:
    best_i, best_d = -1, float("inf")
    for i, w in enumerate(ss.worms):
        d = math.hypot(w["x_orig"] - x_orig, w["y_orig"] - y_orig)
        if d < radius and d < best_d:
            best_d, best_i = d, i
    return best_i


def load_image(name: str):
    """Load image into session state, run detection if no saved result."""
    fb = ss.file_data[name]
    gray = bytes_to_gray(fb)
    ss.orig_h, ss.orig_w = gray.shape
    ss.plate = dict(zip(("cx", "cy", "cr"), auto_detect_plate(gray)))
    ss.current_name = name
    ss.notes = ""
    ss.last_click = None
    ss.mode = "view"

    if name in ss.saved_results:
        r = ss.saved_results[name]
        ss.worms = [dict(w) for w in r["worms"]]
        ss.auto_count = r["auto_count"]
        if "params" in r:
            ss.params = dict(r["params"])
    else:
        ss.worms = detect_worms(
            gray, ss.plate["cx"], ss.plate["cy"], ss.plate["cr"], ss.params
        )
        ss.auto_count = len(ss.worms)


def save_result():
    if not ss.current_name:
        return
    ss.saved_results[ss.current_name] = {
        "filename":        ss.current_name,
        "auto_count":      ss.auto_count,
        "corrected_count": len(ss.worms),
        "notes":           ss.notes,
        "worms":           [dict(w) for w in ss.worms],
        "params":          dict(ss.params),
    }


def csv_bytes() -> bytes:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Filename", "Auto Count", "Corrected Count", "Notes"])
    for r in ss.saved_results.values():
        w.writerow([r["filename"], r["auto_count"], r["corrected_count"], r.get("notes", "")])
    return out.getvalue().encode("utf-8")


def tune_params() -> dict | None:
    """Grid-search best threshold/min_area/max_area from saved results."""
    items = [r for r in ss.saved_results.values() if r.get("worms") is not None]
    if not items:
        return None

    p = ss.params
    k = int(p["blur_kernel"]) | 1  # ensure odd
    k = max(3, k)

    img_data = []
    for r in items:
        fb = ss.file_data.get(r["filename"])
        if fb is None:
            continue
        gray = bytes_to_gray(fb)
        cx, cy, cr = auto_detect_plate(gray)
        blur_img = cv2.GaussianBlur(gray, (k, k), 0)
        img_data.append({
            "blur": blur_img, "gray": gray,
            "cx": cx, "cy": cy, "cr": cr,
            "corrected": int(r["corrected_count"]),
        })

    if not img_data:
        return None

    best_params, best_mae = None, float("inf")
    for thresh in [5, 8, 10, 12, 15, 18, 22, 28]:
        for min_a in [10, 20, 35, 55, 80]:
            for max_a in [1500, 2500, 4000, 6000, 10000]:
                if min_a >= max_a:
                    continue
                total_err = sum(
                    abs(count_fast(d["blur"], d["gray"], d["cx"], d["cy"], d["cr"],
                                   p, thresh, min_a, max_a) - d["corrected"])
                    for d in img_data
                )
                mae = total_err / len(img_data)
                if mae < best_mae:
                    best_mae = mae
                    best_params = {**p, "threshold": thresh,
                                   "min_area": min_a, "max_area": max_a,
                                   "_mae": round(mae, 2)}
    return best_params

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="nat-header">
      <h1>🔬 Worm Counter</h1>
      <p><em>C. elegans</em> automated detection</p>
      <p style="margin-top:6px;font-size:0.72rem;opacity:0.8">
        Developed by <b>Serhat Turkmen, PhD</b><br>
        Sun Lab · PI: Dr. HaoSheng Sun<br>
        University of Alabama at Birmingham · HHMI
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── File upload ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Images</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload images",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="TIF, PNG, JPG — select multiple with Ctrl/Cmd+click",
    )

    if uploaded:
        new_names = []
        for f in uploaded:
            if f.name not in ss.file_data:
                ss.file_data[f.name] = f.read()
            new_names.append(f.name)
        if new_names != ss.file_names:
            ss.file_names = new_names
            if ss.current_name not in new_names:
                ss.current_name = None

    if ss.file_names:
        st.caption(f"{len(ss.file_names)} image{'s' if len(ss.file_names) != 1 else ''} loaded")
        for name in ss.file_names:
            active = name == ss.current_name
            saved  = name in ss.saved_results
            prefix = "✓ " if saved else ""
            label  = f"{prefix}{name}"
            if st.button(label, key=f"f_{name}", use_container_width=True,
                         type="primary" if active else "secondary"):
                if name != ss.current_name:
                    load_image(name)
                    st.rerun()
    else:
        st.info("Upload microscopy images above to begin.")

    st.divider()

    # ── Detection settings ───────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Detection Settings</div>', unsafe_allow_html=True)

    ss.params["blur_kernel"] = st.slider(
        "BG Blur Kernel", 11, 151, int(ss.params["blur_kernel"]), step=2,
        help="Larger = smoother background model"
    )
    ss.params["threshold"] = st.slider(
        "Detection Threshold", 3, 60, int(ss.params["threshold"]),
        help="Higher = only detect darker worms"
    )
    ss.params["min_area"] = float(st.slider(
        "Min Worm Area (px²)", 5, 500, int(ss.params["min_area"]), step=5
    ))
    ss.params["max_area"] = float(st.slider(
        "Max Worm Area (px²)", 200, 20000, int(ss.params["max_area"]), step=100
    ))
    ss.params["roi_scale"] = st.slider(
        "ROI Margin (%)", 60, 100, int(ss.params["roi_scale"] * 100)
    ) / 100

    if st.button("⚡ Auto Detect", use_container_width=True, type="primary"):
        if ss.current_name:
            with st.spinner("Detecting…"):
                gray = bytes_to_gray(ss.file_data[ss.current_name])
                ss.worms = detect_worms(
                    gray, ss.plate["cx"], ss.plate["cy"], ss.plate["cr"], ss.params
                )
                ss.auto_count = len(ss.worms)
            st.rerun()

    st.divider()

    # ── Plate ROI ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Plate ROI (original px)</div>', unsafe_allow_html=True)
    _cx_key = f"roi_cx_{ss.current_name or 'none'}"
    _cy_key = f"roi_cy_{ss.current_name or 'none'}"
    _cr_key = f"roi_cr_{ss.current_name or 'none'}"
    c1, c2, c3 = st.columns(3)
    with c1:
        new_cx = st.number_input("CX", value=int(ss.plate["cx"]), step=10, key=_cx_key,
                                 label_visibility="visible")
    with c2:
        new_cy = st.number_input("CY", value=int(ss.plate["cy"]), step=10, key=_cy_key,
                                 label_visibility="visible")
    with c3:
        new_cr = st.number_input("R",  value=int(ss.plate["cr"]), step=10, key=_cr_key,
                                 label_visibility="visible")
    ss.plate["cx"] = new_cx
    ss.plate["cy"] = new_cy
    ss.plate["cr"] = new_cr

    st.divider()

    # ── Notes + Save ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Notes</div>', unsafe_allow_html=True)
    ss.notes = st.text_area(
        "notes", value=ss.notes, height=56,
        label_visibility="collapsed", placeholder="Optional notes…"
    )
    save_col, toggle_col = st.columns(2)
    with save_col:
        if st.button("💾 Save Result", use_container_width=True, type="primary"):
            save_result()
            st.success("Saved!", icon="✅")
            st.rerun()
    with toggle_col:
        ss.show_markers = st.toggle("Show marks", value=ss.show_markers)

    st.divider()

    # ── Improve / Tune ───────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Improve Detection</div>', unsafe_allow_html=True)
    if st.button("🔧 Tune Parameters", use_container_width=True,
                 help="Grid-search best params from your corrected counts"):
        if not ss.saved_results:
            st.warning("Save at least one corrected result first.")
        else:
            with st.spinner("Searching parameters…"):
                best = tune_params()
            if best:
                mae = best.pop("_mae")
                ss.params = best
                st.success(
                    f"Done! Avg error: **{mae:.1f} worms/image** "
                    f"— Threshold {best['threshold']}, "
                    f"Area {best['min_area']}–{best['max_area']} px²",
                    icon="✅",
                )
                st.rerun()
            else:
                st.error("Could not load images for tuning.")

    st.divider()

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Export</div>', unsafe_allow_html=True)
    if ss.saved_results:
        st.download_button(
            "⬇ Download CSV",
            data=csv_bytes(),
            file_name="worm_counts.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Save at least one result to export.")

    st.divider()

    # ── Saved results table ──────────────────────────────────────────────────
    if ss.saved_results:
        n = len(ss.saved_results)
        st.markdown(
            f'<div class="sec-label">Saved Results ({n})</div>',
            unsafe_allow_html=True,
        )
        rows = []
        for r in ss.saved_results.values():
            short = ("…" + r["filename"][-18:]) if len(r["filename"]) > 20 else r["filename"]
            rows.append({
                "File":  short,
                "Auto":  r["auto_count"],
                "Final": r["corrected_count"],
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

# ─── MAIN AREA ────────────────────────────────────────────────────────────────

# Auto-load first image
if not ss.current_name and ss.file_names:
    load_image(ss.file_names[0])
    st.rerun()

if not ss.current_name:
    st.markdown("""
    <div style="text-align:center;padding:90px 40px;color:#6B7280">
      <div style="font-size:4rem;margin-bottom:16px">🔬</div>
      <h2 style="color:#0E5881;font-weight:700;margin-bottom:8px">Worm Counter</h2>
      <p style="max-width:420px;margin:0 auto;line-height:1.7">
        Upload TIF or PNG microscopy images in the sidebar.<br>
        Detection runs automatically on each new image.<br><br>
        <b>Add mode:</b> click to place a worm marker<br>
        <b>Remove mode:</b> click near a marker to delete it
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Count badge + mode selector ──────────────────────────────────────────────
n_total  = len(ss.worms)
n_manual = sum(1 for w in ss.worms if w.get("manual"))
n_auto   = n_total - n_manual

badge_sub = ""
if n_manual > 0 and n_auto > 0:
    badge_sub = f'<span class="count-sub">{n_auto} auto · {n_manual} manual</span>'
elif n_manual > 0:
    badge_sub = f'<span class="count-sub">{n_manual} manual</span>'

col_badge, col_modes = st.columns([1.4, 5])
with col_badge:
    st.markdown(
        f'<div class="count-badge">'
        f'{n_total} worm{"s" if n_total != 1 else ""}'
        f'{badge_sub}</div>',
        unsafe_allow_html=True,
    )

with col_modes:
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    with m1:
        if st.button("👁 View", use_container_width=True,
                     type="primary" if ss.mode == "view" else "secondary"):
            ss.mode = "view"; st.rerun()
    with m2:
        if st.button("＋ Add", use_container_width=True,
                     type="primary" if ss.mode == "add" else "secondary"):
            ss.mode = "add"; st.rerun()
    with m3:
        if st.button("✕ Remove", use_container_width=True,
                     type="primary" if ss.mode == "remove" else "secondary"):
            ss.mode = "remove"; st.rerun()
    with m4:
        if st.button("✕ All", use_container_width=True,
                     help="Remove all worm markers"):
            ss.worms = []; st.rerun()
    with m5:
        # Prev / Next navigation
        idx = (ss.file_names.index(ss.current_name)
               if ss.current_name in ss.file_names else 0)
        if st.button("◀ Prev", use_container_width=True, disabled=idx == 0):
            load_image(ss.file_names[idx - 1]); st.rerun()
    with m6:
        if st.button("Next ▶", use_container_width=True,
                     disabled=idx >= len(ss.file_names) - 1):
            load_image(ss.file_names[idx + 1]); st.rerun()
    with m7:
        st.caption(f"{idx + 1} / {len(ss.file_names)}")

# Mode hint
_hints = {
    "view":   ("👁 View mode — click to add worms",          "mode-view"),
    "add":    ("＋ Add mode — click on image to place a worm", "mode-add"),
    "remove": ("✕ Remove mode — click near a marker to delete","mode-remove"),
}
_hint_text, _hint_cls = _hints.get(ss.mode, ("", "mode-view"))
st.markdown(
    f'<span class="mode-pill {_hint_cls}">{_hint_text}</span>',
    unsafe_allow_html=True,
)

# ── Render image with overlays ───────────────────────────────────────────────
rendered_pil, disp_scale = render_image(
    ss.file_data[ss.current_name],
    ss.worms, ss.plate, ss.params,
    show_markers=ss.show_markers,
)
ss.disp_scale = disp_scale

# ── Interactive image click ───────────────────────────────────────────────────
try:
    from streamlit_image_coordinates import streamlit_image_coordinates  # type: ignore

    click = streamlit_image_coordinates(
        rendered_pil,
        key=f"img_{ss.current_name}",
    )

    if click is not None and click != ss.last_click:
        ss.last_click = click
        x_orig = round(click["x"] / disp_scale)
        y_orig = round(click["y"] / disp_scale)

        if ss.mode in ("view", "add"):
            ss.worms.append({"x_orig": x_orig, "y_orig": y_orig,
                             "area": 0, "manual": True})
        elif ss.mode == "remove":
            hit = hit_test(x_orig, y_orig, radius=round(20 / disp_scale))
            if hit >= 0:
                ss.worms.pop(hit)
        st.rerun()

except ImportError:
    st.image(rendered_pil, use_container_width=False)
    st.warning(
        "Interactive clicking requires `streamlit-image-coordinates`. "
        "Add it to `requirements.txt` and redeploy.",
        icon="⚠️",
    )

# ── Footer / metadata ─────────────────────────────────────────────────────────
is_saved = ss.current_name in ss.saved_results
save_status = "✓ Saved" if is_saved else "⚠ Unsaved changes"
st.caption(
    f"📄 **{ss.current_name}** · {ss.orig_w}×{ss.orig_h} px · {save_status}"
)

# ── Download annotated image ──────────────────────────────────────────────────
img_buf = io.BytesIO()
rendered_pil.save(img_buf, format="PNG")
st.download_button(
    "🖼 Download image with markers",
    data=img_buf.getvalue(),
    file_name=ss.current_name.rsplit(".", 1)[0] + "_counted.png",
    mime="image/png",
)

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#9CA3AF;font-size:0.75rem;padding:4px 0 12px">
      Developed by <b>Serhat Turkmen, PhD</b> ·
      <b>Sun Lab</b> (PI: Dr. HaoSheng Sun) ·
      University of Alabama at Birmingham · HHMI
    </div>
    """,
    unsafe_allow_html=True,
)
