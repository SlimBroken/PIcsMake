"""
PicsMake - Scanned Photo Splitter Web App
Upload a scanned image containing multiple photos, and get each one
individually cropped at the highest quality.
"""

import os
import uuid
import zipfile
import io
import shutil
from flask import Flask, request, jsonify, send_file, render_template_string

from photo_splitter import detect_and_extract_photos, save_photo_to_bytes

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max upload

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── HTML Template ────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PicsMake - Scanned Photo Splitter</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242734;
    --border: #2e3144;
    --text: #e4e6f0;
    --text-dim: #8b8fa3;
    --accent: #6c63ff;
    --accent-hover: #5a52e0;
    --success: #4ade80;
    --error: #f87171;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  header {
    width: 100%;
    padding: 24px 32px;
    text-align: center;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  header h1 {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  header h1 span { color: var(--accent); }
  header p {
    color: var(--text-dim);
    margin-top: 6px;
    font-size: 15px;
  }
  .container {
    max-width: 900px;
    width: 100%;
    padding: 32px 24px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  /* Drop zone */
  .drop-zone {
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 60px 32px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    background: var(--surface);
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--accent);
    background: var(--surface2);
  }
  .drop-zone .icon { font-size: 48px; margin-bottom: 12px; display: block; }
  .drop-zone .main-text { font-size: 18px; font-weight: 600; }
  .drop-zone .sub-text { color: var(--text-dim); font-size: 14px; margin-top: 8px; }
  .drop-zone input[type="file"] { display: none; }

  /* Preview of uploaded image */
  .preview-area {
    display: none;
    background: var(--surface);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid var(--border);
  }
  .preview-area img {
    max-width: 100%;
    max-height: 360px;
    border-radius: 8px;
    display: block;
    margin: 0 auto;
  }
  .preview-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
  }
  .preview-header h3 { font-size: 16px; }
  .btn-remove {
    background: none;
    border: 1px solid var(--border);
    color: var(--text-dim);
    padding: 6px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
  }
  .btn-remove:hover { border-color: var(--error); color: var(--error); }

  /* Options */
  .options-bar {
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
  }
  .options-bar label {
    font-size: 14px;
    color: var(--text-dim);
  }
  .options-bar select {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 14px;
  }

  /* Process button */
  .btn-process {
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 14px 32px;
    font-size: 16px;
    font-weight: 600;
    border-radius: 12px;
    cursor: pointer;
    transition: background 0.2s;
    width: 100%;
  }
  .btn-process:hover { background: var(--accent-hover); }
  .btn-process:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Progress */
  .progress-bar {
    display: none;
    background: var(--surface);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--border);
    text-align: center;
  }
  .progress-bar .spinner {
    width: 40px; height: 40px;
    border: 4px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .progress-bar p { color: var(--text-dim); font-size: 15px; }

  /* Results */
  .results {
    display: none;
    background: var(--surface);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid var(--border);
  }
  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }
  .results-header h3 { font-size: 18px; }
  .btn-download-all {
    background: var(--success);
    color: #000;
    border: none;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: 600;
    border-radius: 10px;
    cursor: pointer;
    transition: opacity 0.2s;
  }
  .btn-download-all:hover { opacity: 0.85; }
  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
  }
  .result-card {
    background: var(--surface2);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
    transition: border-color 0.2s;
  }
  .result-card:hover { border-color: var(--accent); }
  .result-card img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
  }
  .result-card .card-footer {
    padding: 10px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .result-card .card-footer span {
    font-size: 13px;
    color: var(--text-dim);
  }
  .btn-dl {
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    text-decoration: none;
  }

  /* Error */
  .error-msg {
    display: none;
    background: #2d1b1b;
    border: 1px solid var(--error);
    border-radius: 12px;
    padding: 16px 20px;
    color: var(--error);
    font-size: 14px;
  }

  footer {
    margin-top: auto;
    padding: 20px;
    text-align: center;
    color: var(--text-dim);
    font-size: 13px;
    border-top: 1px solid var(--border);
    width: 100%;
  }
</style>
</head>
<body>

<header>
  <h1>Pics<span>Make</span></h1>
  <p>Upload a scanned page of photos &mdash; get each one perfectly cropped</p>
</header>

<div class="container">
  <!-- Drop zone -->
  <div class="drop-zone" id="dropZone">
    <span class="icon">&#128247;</span>
    <div class="main-text">Drop your scanned image here</div>
    <div class="sub-text">or click to browse &middot; PNG, JPG, TIFF, BMP, WebP &middot; up to 100 MB</div>
    <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp">
  </div>

  <!-- Preview -->
  <div class="preview-area" id="previewArea">
    <div class="preview-header">
      <h3 id="fileName">scan.jpg</h3>
      <button class="btn-remove" id="btnRemove">Remove</button>
    </div>
    <img id="previewImg" src="" alt="Preview">
  </div>

  <!-- Options -->
  <div class="options-bar" id="optionsBar" style="display:none;">
    <label>Output format:</label>
    <select id="outputFormat">
      <option value="png" selected>PNG (lossless)</option>
      <option value="jpeg">JPEG (max quality)</option>
    </select>
  </div>

  <!-- Process button -->
  <button class="btn-process" id="btnProcess" style="display:none;">Split Photos</button>

  <!-- Progress -->
  <div class="progress-bar" id="progressBar">
    <div class="spinner"></div>
    <p>Detecting and extracting photos&hellip;</p>
  </div>

  <!-- Error -->
  <div class="error-msg" id="errorMsg"></div>

  <!-- Results -->
  <div class="results" id="results">
    <div class="results-header">
      <h3 id="resultsTitle">Extracted Photos</h3>
      <button class="btn-download-all" id="btnDownloadAll">Download All (.zip)</button>
    </div>
    <div class="results-grid" id="resultsGrid"></div>
  </div>
</div>

<footer>PicsMake &mdash; restore your memories, one scan at a time</footer>

<script>
const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const previewArea  = document.getElementById('previewArea');
const previewImg   = document.getElementById('previewImg');
const fileName     = document.getElementById('fileName');
const btnRemove    = document.getElementById('btnRemove');
const optionsBar   = document.getElementById('optionsBar');
const btnProcess   = document.getElementById('btnProcess');
const progressBar  = document.getElementById('progressBar');
const errorMsg     = document.getElementById('errorMsg');
const results      = document.getElementById('results');
const resultsGrid  = document.getElementById('resultsGrid');
const resultsTitle = document.getElementById('resultsTitle');
const btnDownloadAll = document.getElementById('btnDownloadAll');
const outputFormat = document.getElementById('outputFormat');

let currentFile = null;
let currentJobId = null;

// ── Drag & drop ──
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(file) {
  currentFile = file;
  fileName.textContent = file.name;
  previewImg.src = URL.createObjectURL(file);
  previewArea.style.display = 'block';
  dropZone.style.display = 'none';
  optionsBar.style.display = 'flex';
  btnProcess.style.display = 'block';
  results.style.display = 'none';
  errorMsg.style.display = 'none';
}

btnRemove.addEventListener('click', () => {
  currentFile = null;
  fileInput.value = '';
  previewArea.style.display = 'none';
  dropZone.style.display = 'block';
  optionsBar.style.display = 'none';
  btnProcess.style.display = 'none';
  results.style.display = 'none';
  errorMsg.style.display = 'none';
});

// ── Process ──
btnProcess.addEventListener('click', async () => {
  if (!currentFile) return;

  btnProcess.disabled = true;
  progressBar.style.display = 'block';
  results.style.display = 'none';
  errorMsg.style.display = 'none';

  const formData = new FormData();
  formData.append('scan', currentFile);
  formData.append('format', outputFormat.value);

  try {
    const resp = await fetch('/api/split', { method: 'POST', body: formData });
    let data;
    try {
      data = await resp.json();
    } catch (_) {
      throw new Error(`Server error (HTTP ${resp.status}). The image may be too large for the free server — try a smaller file.`);
    }

    if (!resp.ok) throw new Error(data.error || 'Processing failed');

    currentJobId = data.job_id;
    resultsTitle.textContent = `Extracted ${data.count} Photo${data.count !== 1 ? 's' : ''}`;
    resultsGrid.innerHTML = '';

    data.photos.forEach((photo, i) => {
      const card = document.createElement('div');
      card.className = 'result-card';
      card.innerHTML = `
        <img src="/api/photo/${data.job_id}/${photo.filename}" alt="Photo ${i+1}">
        <div class="card-footer">
          <span>${photo.width}&times;${photo.height}</span>
          <a class="btn-dl" href="/api/photo/${data.job_id}/${photo.filename}" download="${photo.filename}">Save</a>
        </div>`;
      resultsGrid.appendChild(card);
    });

    results.style.display = 'block';
  } catch (err) {
    errorMsg.textContent = err.message;
    errorMsg.style.display = 'block';
  } finally {
    btnProcess.disabled = false;
    progressBar.style.display = 'none';
  }
});

// ── Download all ──
btnDownloadAll.addEventListener('click', () => {
  if (currentJobId) window.location.href = `/api/download-all/${currentJobId}`;
});
</script>
</body>
</html>
"""


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/split", methods=["POST"])
def split_photos():
    if "scan" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["scan"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, TIFF, BMP, or WebP."}), 400

    output_format = request.form.get("format", "png").lower()
    if output_format not in ("png", "jpeg"):
        output_format = "png"

    ext = "png" if output_format == "png" else "jpg"
    image_bytes = file.read()

    try:
        photos = detect_and_extract_photos(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        del image_bytes  # release upload bytes as soon as processing is done

    if len(photos) == 0:
        return jsonify({"error": "No individual photos detected. Try a scan with more contrast between photos and the background."}), 422

    # Save results — save and release each photo immediately to keep memory low
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    photo_info = []
    for i, img in enumerate(photos):
        fname = f"photo_{i+1}.{ext}"
        fpath = os.path.join(job_dir, fname)
        w, h = img.width, img.height
        if output_format == "jpeg":
            img.save(fpath, "JPEG", quality=100, subsampling=0)
        else:
            img.save(fpath, "PNG", optimize=False)
        img.close()
        photos[i] = None  # allow GC to reclaim immediately
        photo_info.append({
            "filename": fname,
            "width": w,
            "height": h,
        })

    return jsonify({
        "job_id": job_id,
        "count": len(photos),
        "photos": photo_info,
    })


@app.route("/api/photo/<job_id>/<filename>")
def get_photo(job_id, filename):
    # Sanitize to prevent directory traversal
    safe_job = os.path.basename(job_id)
    safe_name = os.path.basename(filename)
    fpath = os.path.join(OUTPUT_DIR, safe_job, safe_name)

    if not os.path.isfile(fpath):
        return jsonify({"error": "Photo not found"}), 404

    return send_file(fpath)


@app.route("/api/download-all/<job_id>")
def download_all(job_id):
    safe_job = os.path.basename(job_id)
    job_dir = os.path.join(OUTPUT_DIR, safe_job)

    if not os.path.isdir(job_dir):
        return jsonify({"error": "Job not found"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(job_dir)):
            fpath = os.path.join(job_dir, fname)
            zf.write(fpath, fname)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"picsmake_{safe_job}.zip",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
