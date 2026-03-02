#!/usr/bin/env python3
"""
Groq JD Parser — API Server
Flask API for parsing job descriptions using Llama 3.1 via Groq.
Supports single file, bulk upload, and text input.
"""

import os
import time
import tempfile
import concurrent.futures
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from groq_jd_parser import parse_jd, extract_text_from_file, is_groq_configured, GROQ_MODEL

app = Flask(__name__, static_folder=".")
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt", "html", "htm", "jpg", "jpeg", "png", "tiff", "bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024
BULK_MAX_SIZE = 50 * 1024 * 1024
BULK_MAX_FILES = 50

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = BULK_MAX_SIZE


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_single_file(filepath, filename):
    """Parse a single JD file and return result dict."""
    start = time.time()
    try:
        jd_text = extract_text_from_file(filepath)
        if not jd_text or len(jd_text.strip()) < 30:
            return {"filename": filename, "error": "Could not extract text from file"}

        result = parse_jd(jd_text, filename=filename)
        elapsed = int((time.time() - start) * 1000)

        if "error" in result:
            return {"filename": filename, "error": result["error"], "processing_time_ms": elapsed}

        return {"filename": filename, "processing_time_ms": elapsed, "result": result}
    except Exception as e:
        return {"filename": filename, "error": str(e)}
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


# --- Routes ---

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "parser": "groq_jd_parser",
        "groq_configured": is_groq_configured(),
        "model": GROQ_MODEL,
        "supported_formats": sorted(ALLOWED_EXTENSIONS),
        "max_bulk_files": BULK_MAX_FILES,
        "timestamp": time.time(),
    })


@app.route("/parse", methods=["POST"])
def parse():
    """Parse a single uploaded JD file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send a file with key \"file\"."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result = parse_single_file(filepath, filename)

    if "error" in result and "result" not in result:
        return jsonify(result), 502

    return jsonify(result)


@app.route("/parse/text", methods=["POST"])
def parse_text():
    """Parse raw JD text (no file upload)."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Send JSON with \"text\" field containing JD text."}), 400

    jd_text = data["text"]
    filename = data.get("filename", "text_input")

    if len(jd_text.strip()) < 30:
        return jsonify({"error": "JD text is too short."}), 400

    result = parse_jd(jd_text, filename=filename)

    if "error" in result:
        return jsonify(result), 502

    return jsonify({"result": result})


@app.route("/parse/bulk", methods=["POST"])
def parse_bulk():
    """Parse multiple JD files (up to 50)."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided. Send files with key \"files\"."}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    if len(files) > BULK_MAX_FILES:
        return jsonify({"error": f"Too many files. Maximum {BULK_MAX_FILES} per request."}), 400

    start = time.time()

    tasks = []
    for file in files:
        if file.filename == "" or not allowed_file(file.filename):
            continue
        filename = secure_filename(file.filename)
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"], f"bulk_{int(time.time()*1000)}_{filename}"
        )
        file.save(filepath)
        tasks.append((filepath, filename))

    if not tasks:
        return jsonify({"error": "No valid files found in upload."}), 400

    # Parse sequentially for accuracy (Groq rate limits)
    # Use max 3 workers to avoid rate limiting
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(parse_single_file, fp, fn): fn for fp, fn in tasks}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    elapsed = int((time.time() - start) * 1000)
    successful = sum(1 for r in results if "result" in r)

    return jsonify({
        "total_files": len(tasks),
        "successful": successful,
        "failed": len(tasks) - successful,
        "total_processing_time_ms": elapsed,
        "results": results,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print(f"Starting Groq JD Parser on port {port}")
    print(f"Model: {GROQ_MODEL}")
    print(f"Groq API: {'configured' if is_groq_configured() else 'NOT SET'}")
    print(f"Single-pass parsing: enabled")
    print(f"Formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    app.run(host="0.0.0.0", port=port, debug=True)
