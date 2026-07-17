import os
import sys
import io
import base64
import csv
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS


def _resource_path(relative: str) -> str:
    """Resolve path to a bundled resource (works for dev AND PyInstaller builds)."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative)


app = Flask(__name__, template_folder=_resource_path("templates"))
CORS(app)

# Main images folder — default is parent of worm_app, overridable via CLI arg or UI
_DEFAULT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DIR

# Scale factor for display images sent to browser
DISPLAY_MAX_SIZE = 1400


def list_images():
    """List image files in IMAGE_DIR only (no subfolders)."""
    files = []
    try:
        for f in os.listdir(IMAGE_DIR):
            if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                full = os.path.join(IMAGE_DIR, f)
                if os.path.isfile(full):
                    files.append(f)
    except OSError:
        pass
    files.sort()
    return files


def load_image_gray(filename):
    path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {filename}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def load_image_color(filename):
    path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {filename}")
    return img


def auto_detect_plate(gray):
    """Auto-detect the circular plate using Hough circles."""
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1, minDist=500,
        param1=50, param2=30,
        minRadius=int(min(h, w) * 0.3),
        maxRadius=int(min(h, w) * 0.6)
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')
        cx, cy, cr = circles[0]
        return int(cx), int(cy), int(cr)
    # Fallback: assume centered circle
    cx, cy = w // 2, h // 2
    cr = int(min(h, w) * 0.48)
    return cx, cy, cr


def detect_worms(gray, cx, cy, cr, roi_scale, blur_kernel, threshold, min_area, max_area):
    """
    Detect worms in the image using background subtraction.

    Returns list of dicts: {x, y, area}
    """
    h, w = gray.shape
    mask_r = int(cr * roi_scale)

    # Circular mask
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), mask_r, 255, -1)

    # Background model via Gaussian blur
    k = int(blur_kernel)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    # Difference: areas darker than background
    diff = cv2.subtract(blur, gray)
    diff = cv2.bitwise_and(diff, diff, mask=mask)

    # Threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological closing to merge nearby segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    worms = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            M = cv2.moments(c)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                # Make sure center is inside mask
                if 0 <= py < h and 0 <= px < w and mask[py, px] > 0:
                    # Bounding box
                    x, y, bw, bh = cv2.boundingRect(c)
                    worms.append({
                        'x': px,
                        'y': py,
                        'area': float(area),
                        'bbox': [x, y, bw, bh]
                    })

    return worms


def image_to_jpeg_b64(img_bgr, max_size=DISPLAY_MAX_SIZE):
    """Convert BGR image to JPEG base64 string, scaled for display."""
    h, w = img_bgr.shape[:2]
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode('utf-8')
    return b64, img_bgr.shape[1], img_bgr.shape[0]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/folder', methods=['GET'])
def api_get_folder():
    return jsonify({'folder': IMAGE_DIR, 'image_count': len(list_images())})


@app.route('/api/folder/browse', methods=['POST'])
def api_browse_folder():
    """Open a native macOS/OS folder picker and return chosen path."""
    global IMAGE_DIR
    chosen = None
    error = None
    done = threading.Event()

    def _pick():
        nonlocal chosen, error
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.call('wm', 'attributes', '.', '-topmost', True)
            path = filedialog.askdirectory(title='Select image folder', initialdir=IMAGE_DIR)
            root.destroy()
            if path:
                chosen = path
        except Exception as e:
            error = str(e)
        finally:
            done.set()

    t = threading.Thread(target=_pick)
    t.start()
    done.wait(timeout=60)

    if error:
        return jsonify({'error': error}), 500
    if chosen:
        IMAGE_DIR = chosen
        return jsonify({'folder': IMAGE_DIR, 'image_count': len(list_images())})
    return jsonify({'folder': IMAGE_DIR, 'image_count': len(list_images()), 'cancelled': True})


@app.route('/api/folder/set', methods=['POST'])
def api_set_folder():
    """Set folder by path typed directly in the UI."""
    global IMAGE_DIR
    data = request.json
    path = data.get('path', '').strip()
    path = os.path.expanduser(path)  # expand ~ if present
    if not os.path.isdir(path):
        return jsonify({'error': f'Directory not found: {path}'}), 400
    IMAGE_DIR = path
    return jsonify({'folder': IMAGE_DIR, 'image_count': len(list_images())})


@app.route('/api/images')
def api_images():
    return jsonify(list_images())


@app.route('/api/image/<path:filename>')
def api_image(filename):
    """Return display-sized image as JPEG base64 + plate detection."""
    try:
        gray = load_image_gray(filename)
        color = load_image_color(filename)
        h_orig, w_orig = gray.shape

        # Auto-detect plate
        cx, cy, cr = auto_detect_plate(gray)

        # Scale for display
        scale = min(DISPLAY_MAX_SIZE / w_orig, DISPLAY_MAX_SIZE / h_orig, 1.0)
        b64, disp_w, disp_h = image_to_jpeg_b64(color)

        return jsonify({
            'image_b64': b64,
            'orig_w': w_orig,
            'orig_h': h_orig,
            'disp_w': disp_w,
            'disp_h': disp_h,
            'scale': scale,
            'plate': {
                'cx': cx,
                'cy': cy,
                'cr': cr,
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Run worm detection with given parameters.
    Optional 'bbox': [x, y, w, h] in original coords restricts detection to that rectangle.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'Invalid JSON body'}), 400

    for field in ('filename', 'cx', 'cy', 'cr'):
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    filename = data['filename']
    cx = int(data['cx'])
    cy = int(data['cy'])
    cr = int(data['cr'])
    roi_scale = float(data.get('roi_scale', 0.90))
    blur_kernel = int(data.get('blur_kernel', 51))
    threshold = int(data.get('threshold', 15))
    min_area = float(data.get('min_area', 30))
    max_area = float(data.get('max_area', 4000))
    bbox = data.get('bbox', None)  # [x, y, w, h] in original coords

    try:
        gray = load_image_gray(filename)
        h_orig, w_orig = gray.shape

        # If a bounding box is given, crop the image for detection
        if bbox:
            bx, by, bw, bh = [int(v) for v in bbox]
            bx = max(0, bx); by = max(0, by)
            bw = min(bw, w_orig - bx); bh = min(bh, h_orig - by)
            gray_crop = gray[by:by+bh, bx:bx+bw]
            # Shift plate center into crop coordinates
            cx_crop = cx - bx
            cy_crop = cy - by
            worms_crop = detect_worms(gray_crop, cx_crop, cy_crop, cr,
                                       roi_scale, blur_kernel, threshold, min_area, max_area)
            # Shift worm coords back to full image
            worms = []
            for w in worms_crop:
                w['x'] += bx; w['y'] += by
                w['bbox'][0] += bx; w['bbox'][1] += by
                worms.append(w)
        else:
            worms = detect_worms(gray, cx, cy, cr, roi_scale, blur_kernel, threshold, min_area, max_area)

        # Scale worm coords to display size
        scale = min(DISPLAY_MAX_SIZE / w_orig, DISPLAY_MAX_SIZE / h_orig, 1.0)

        worms_disp = []
        for w in worms:
            worms_disp.append({
                'x': w['x'] * scale,
                'y': w['y'] * scale,
                'x_orig': w['x'],
                'y_orig': w['y'],
                'area': w['area'],
                'bbox': [int(v * scale) for v in w['bbox']]
            })

        return jsonify({
            'count': len(worms_disp),
            'worms': worms_disp,
            'scale': scale
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def api_export():
    """Export results as CSV."""
    data = request.json or {}
    if 'results' not in data:
        return jsonify({'error': 'Missing results field'}), 400
    results = data['results']  # list of {filename, count, corrected_count, worms}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Filename', 'Auto Count', 'Corrected Count', 'Notes'])
    for r in results:
        writer.writerow([
            r.get('filename', ''),
            r.get('auto_count', ''),
            r.get('corrected_count', ''),
            r.get('notes', '')
        ])

    output.seek(0)
    buf = io.BytesIO(output.getvalue().encode('utf-8'))
    return send_file(buf, mimetype='text/csv',
                     as_attachment=True, download_name='worm_counts.csv')


def _count_fast(blur, gray, cx, cy, cr, roi_scale, threshold, min_area, max_area):
    """Detection count using a pre-blurred image (avoids re-blurring in grid search)."""
    h, w = gray.shape
    mask_r = int(cr * roi_scale)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), mask_r, 255, -1)
    diff = cv2.subtract(blur, gray)
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    _, thresh_img = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if min_area <= cv2.contourArea(c) <= max_area)


@app.route('/api/tune', methods=['POST'])
def api_tune():
    """Grid-search detection parameters using saved corrected counts."""
    data = request.json
    results = data.get('results', [])
    roi_scale   = float(data.get('roi_scale', 0.90))
    blur_kernel = int(data.get('blur_kernel', 51))

    if not results:
        return jsonify({'error': 'No saved results to tune with'}), 400

    # Load images, auto-detect plates, pre-blur once per image
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    k = max(3, k)
    images_data = []
    for r in results:
        try:
            gray = load_image_gray(r['filename'])
            cx, cy, cr = auto_detect_plate(gray)
            blur = cv2.GaussianBlur(gray, (k, k), 0)
            images_data.append({
                'blur': blur, 'gray': gray,
                'cx': cx, 'cy': cy, 'cr': cr,
                'corrected': int(r['corrected_count'])
            })
        except Exception:
            pass

    if not images_data:
        return jsonify({'error': 'Could not load any images'}), 500

    # Grid search — blur pre-computed, search threshold / min_area / max_area
    thresholds = [5, 8, 10, 12, 15, 18, 22, 28]
    min_areas  = [10, 20, 35, 55, 80]
    max_areas  = [1500, 2500, 4000, 6000, 10000]

    best_params, best_mae = None, float('inf')
    for thresh in thresholds:
        for min_a in min_areas:
            for max_a in max_areas:
                if min_a >= max_a:
                    continue
                total_err = sum(
                    abs(_count_fast(d['blur'], d['gray'], d['cx'], d['cy'], d['cr'],
                                    roi_scale, thresh, min_a, max_a) - d['corrected'])
                    for d in images_data
                )
                mae = total_err / len(images_data)
                if mae < best_mae:
                    best_mae = mae
                    best_params = {
                        'blur_kernel': blur_kernel,
                        'threshold': thresh,
                        'min_area': min_a,
                        'max_area': max_a,
                        'roi_scale': roi_scale,
                    }

    return jsonify({'params': best_params, 'mae': round(best_mae, 2), 'n_images': len(images_data)})


@app.route('/api/export/yolo', methods=['POST'])
def api_export_yolo():
    """Export YOLO-format annotation labels for all saved corrected results."""
    import zipfile
    data = request.json
    results = data.get('results', [])

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:

        zf.writestr('worm_dataset/data.yaml',
                    "path: ./worm_dataset\ntrain: images\nval: images\n\nnc: 1\nnames: ['worm']\n")

        readme = (
            "Worm Counter — YOLO Annotation Export\n"
            "======================================\n\n"
            "Contents:\n"
            "  worm_dataset/data.yaml      — dataset config\n"
            "  worm_dataset/labels/*.txt   — one label file per corrected image\n"
            "  worm_dataset/images/        — copy your original TIF/PNG images here\n\n"
            "Label format (YOLO): class cx cy width height  (all values 0–1)\n\n"
            "To train YOLOv8-nano (recommended first step):\n"
            "  pip install ultralytics\n"
            "  yolo detect train data=worm_dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=1280\n\n"
            "Tip: collect 50+ corrected images for reliable training.\n"
        )
        zf.writestr('worm_dataset/README.txt', readme)

        n_exported = 0
        for r in results:
            filename = r.get('filename', '')
            worms = r.get('worms', [])
            if not worms:
                continue
            try:
                gray = load_image_gray(filename)
                h_img, w_img = gray.shape
            except Exception:
                continue

            lines = []
            for w in worms:
                cx_n = w['x'] / w_img
                cy_n = w['y'] / h_img
                area = w.get('area', 0)
                side = (area ** 0.5) if area > 0 else 50
                bw_n = min(side * 1.5 / w_img, 0.08)
                bh_n = min(side * 1.5 / h_img, 0.08)
                lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

            label_name = filename.rsplit('.', 1)[0] + '.txt'
            zf.writestr(f'worm_dataset/labels/{label_name}', '\n'.join(lines))
            n_exported += 1

    if n_exported == 0:
        return jsonify({'error': 'No annotated worm positions to export. Save results with worm markers first.'}), 400

    buf.seek(0)
    return send_file(buf, mimetype='application/zip',
                     as_attachment=True, download_name='worm_annotations_yolo.zip')


if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = '127.0.0.1'
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Found {len(list_images())} images")
    print(f"\n  Local:   http://127.0.0.1:8080")
    print(f"  Network: http://{local_ip}:8080  (share this with lab colleagues)\n")
    app.run(host='0.0.0.0', debug=False, use_reloader=False, port=8080)
