#!/usr/bin/env python3
"""Local web page for clipper.py: paste a YouTube link, get vertical captioned clips.

Run:  .venv/bin/python app.py   then open http://localhost:8765
"""
import dataclasses
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, request, send_from_directory

import clipper

ROOT = Path(__file__).parent
CLIPS = ROOT / "clips"
app = Flask(__name__, static_folder=None)
jobs = {}
lock = threading.Lock()

STAGE_LABELS = {"download": "Downloading video", "model": "Downloading speech model (one-time)",
                "transcribe": "Transcribing speech", "pick": "Picking the best moments",
                "render": "Rendering clips"}


def on_progress(job, ev):
    now = time.time()
    if ev["stage"] != job["stage_key"]:
        job.update(stage_key=ev["stage"], stage=STAGE_LABELS.get(ev["stage"], ev["stage"]), stage_started=now)
    job.update(frac=ev["frac"], est=ev["est"], rest=ev["rest"])


def estimate(job):
    """Overall percent + seconds left. The current stage's remaining time comes from its measured
    rate (blended with the upfront estimate while progress is still small); later stages use the
    estimates clipper.py sent."""
    if job["status"] != "running":
        return 100.0, 0.0
    if job["stage_key"] is None:
        return 0.0, None
    now = time.time()
    f, est, elapsed = job["frac"], job["est"], now - job["stage_started"]
    guess = max(est - elapsed, est * (1 - f) * 0.5, 1.0)
    if f > 0.01 and elapsed > 2:
        w = min(1.0, f * 5)
        remaining = w * elapsed * (1 - f) / f + (1 - w) * guess
    else:
        remaining = guess
    deadline = now + remaining + job["rest"]
    if job["deadline"]:  # smooth so the ETA doesn't jump around between polls
        deadline = 0.7 * job["deadline"] + 0.3 * deadline
    job["deadline"] = deadline
    eta = max(deadline - now, 0.0)
    total = now - job["started"]
    pct = max(job["pct"], min(99.0, 100 * total / (total + eta)))
    job["pct"] = pct
    return pct, eta


def parse_style(data):
    """Build a validated CaptionStyle from request JSON/args, falling back to defaults."""
    d = clipper.CaptionStyle()
    data = data or {}

    def num(key, default, lo, hi):
        try:
            return max(lo, min(hi, int(float(data.get(key, default)))))
        except (TypeError, ValueError):
            return default

    def colour(key, default):
        v = str(data.get(key, default))
        v = v if v.startswith("#") else "#" + v
        return v.upper() if re.fullmatch(r"#[0-9a-fA-F]{6}", v) else default

    def flag(key, default):
        v = data.get(key, default)
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")

    font = data.get("font", d.font)
    return clipper.CaptionStyle(
        font=font if font in clipper.available_fonts() else d.font,
        size=num("size", d.size, 30, 200), text_color=colour("text_color", d.text_color),
        highlight=colour("highlight", d.highlight), outline_color=colour("outline_color", d.outline_color),
        outline=num("outline", d.outline, 0, 30), position=num("position", d.position, 5, 95),
        words=num("words", d.words, 1, 6), uppercase=flag("uppercase", d.uppercase), pop=flag("pop", d.pop))


def style_args(st):
    args = ["--font", st.font, "--font-size", str(st.size), "--text-color", st.text_color,
            "--highlight", st.highlight, "--outline-color", st.outline_color, "--outline", str(st.outline),
            "--caption-pos", str(st.position), "--words", str(st.words)]
    return args + ([] if st.uppercase else ["--no-uppercase"]) + ([] if st.pop else ["--no-pop"])


_preview_bg = {}


def preview_background():
    """A 9:16 still from the most recently downloaded video, so the preview looks like a real clip."""
    videos = sorted((ROOT / "work").glob("*/source.mp4"), key=lambda p: p.stat().st_mtime)
    if not videos:
        return None
    src = videos[-1]
    if src not in _preview_bg:
        dur = clipper.probe_duration(src)
        w, h = clipper.probe_size(src)
        frame = next(clipper.read_frames(src, dur * 0.35, 1, w, h, 1), None)
        if frame is None:
            return None
        cw = min(w, int(h * 9 / 16))
        x0 = (w - cw) // 2
        _preview_bg.clear()
        _preview_bg[src] = cv2.resize(frame[:, x0:x0 + cw], (clipper.OUT_W, clipper.OUT_H))
    return _preview_bg[src]


def run_job(job):
    cmd = [sys.executable, "-u", str(ROOT / "clipper.py"), job["url"], "--num", str(job["num"]),
           "--min", str(job["min"]), "--max", str(job["max"]),
           "--whisper", "small" if job["fast"] else "turbo", *style_args(job["style"])]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_dir = None
    for line in proc.stdout:
        line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        if line.startswith("@@ "):
            on_progress(job, json.loads(line[3:]))
            continue
        if not line.startswith("==>"):
            job["tail"] = line[-200:]
            continue
        msg = line[3:].strip()
        job["log"].append(msg)
        m = re.search(r"clips in (.+)$", msg)
        if m:
            out_dir = Path(m.group(1))
    proc.wait()
    job["ended"] = time.time()
    if proc.returncode == 0 and out_dir and (out_dir / "clips.json").exists():
        clips = json.loads((out_dir / "clips.json").read_text())
        for c in clips:
            c["url"] = "/clips/" + Path(c["file"]).relative_to(CLIPS).as_posix()
            c.pop("file", None)
        job.update(status="done", stage="Done", clips=clips)
    else:
        job.update(status="error", error=job.get("tail") or "clipper.py failed")


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "index.html")


@app.post("/api/jobs")
def create_job():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not re.match(r"^https?://", url):
        return jsonify(error="Paste a full video link starting with https://"), 400
    with lock:
        if any(j["status"] == "running" for j in jobs.values()):
            return jsonify(error="A video is already being processed - wait for it to finish."), 409
        job = {"id": uuid.uuid4().hex[:8], "url": url, "status": "running", "stage": "Fetching video info",
               "stage_key": None, "stage_started": time.time(), "started": time.time(), "frac": 0.0,
               "est": 0.0, "rest": 0.0, "deadline": None, "pct": 0.0,
               "log": [], "tail": "", "fast": bool(data.get("fast")), "clips": [],
               "style": parse_style(data.get("style")),
               "num": max(1, min(10, int(data.get("num", 5)))),
               "min": max(10, min(120, float(data.get("min", 25)))),
               "max": max(15, min(180, float(data.get("max", 60))))}
        if job["max"] <= job["min"]:
            job["max"] = job["min"] + 15
        jobs[job["id"]] = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return jsonify(id=job["id"])


@app.get("/api/jobs/<job_id>")
def get_job(job_id):
    job = jobs.get(job_id) or abort(404)
    pct, eta = estimate(job)
    return jsonify({k: job[k] for k in ("id", "status", "stage", "stage_key", "log", "tail", "clips", "num")}
                   | {"error": job.get("error"), "pct": round(pct, 1), "eta": None if eta is None else round(eta),
                      "stage_pct": round(job["frac"] * 100), "elapsed": round((job.get("ended") or time.time()) - job["started"])})


@app.get("/api/fonts")
def fonts():
    return jsonify(fonts=list(clipper.available_fonts()), default=dataclasses.asdict(clipper.CaptionStyle()))


@app.get("/api/preview")
def preview():
    frame = clipper.caption_preview(parse_style(request.args), preview_background())
    small = cv2.resize(frame, (360, 640), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/clips/<path:name>")
def clip_file(name):
    return send_from_directory(CLIPS, name, as_attachment=request.args.get("download") == "1")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, threaded=True)
