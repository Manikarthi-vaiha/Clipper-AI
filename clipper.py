#!/usr/bin/env python3
"""Mini OpusClip: YouTube link -> vertical short clips with animated word-by-word captions.

Pipeline: download -> transcribe (word timestamps) -> pick best moments -> reframe 16:9 to 9:16
around the speaker's face -> burn in animated captions -> encode.

Usage:
    python clipper.py "https://www.youtube.com/watch?v=..." --num 5
    python clipper.py my_video.mp4 --min 20 --max 60
"""
import argparse
import glob
import json
import shutil
import os
import re
import subprocess
import sys
import time
import types
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
OUT_W, OUT_H = 1080, 1920
FPS = 30
WHISPER_MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "small": "mlx-community/whisper-small-mlx",
}


def log(msg):
    print(f"\033[1;36m==>\033[0m {msg}", flush=True)


# Rough speeds measured on an M1 MacBook, used for the ETA before a stage has reported any progress.
EST_DOWNLOAD_BPS = 8e6            # network bytes/sec
EST_TRANSCRIBE_X = {"turbo": 4, "small": 12}   # x faster than real time
EST_RENDER_PER_SEC = 0.45         # seconds of work per second of clip


class Progress:
    """Emits machine-readable '@@ {json}' lines that app.py turns into a percentage and ETA.

    plan maps each pipeline stage (in order) to its estimated duration in seconds.
    """

    def __init__(self):
        self.plan, self.last = {}, 0.0

    def __call__(self, stage, frac, force=False):
        now = time.time()
        if not force and now - self.last < 0.5:
            return
        self.last = now
        keys = list(self.plan)
        rest = sum(self.plan[k] for k in keys[keys.index(stage) + 1:]) if stage in self.plan else 0
        print("@@ " + json.dumps({"stage": stage, "frac": round(min(max(frac, 0.0), 1.0), 4),
                                  "est": round(self.plan.get(stage, 0), 1), "rest": round(rest, 1)}),
              flush=True)


PROGRESS = Progress()


def dir_size(folder):
    return sum(f.stat().st_size for f in Path(folder).rglob("*") if f.is_file() and not f.is_symlink())


# ---------------------------------------------------------------- 1. download

def js_runtimes():
    """YouTube requires solving a JS challenge; without a runtime downloads fail with HTTP 403."""
    for name in ("deno", "node", "bun"):
        path = shutil.which(name)
        if path:
            return {name: {"path": path}}
    nvm = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/*/bin/node")))  # not on PATH for GUI apps
    if nvm:
        return {"node": {"path": nvm[-1]}}
    log("Warning: no JavaScript runtime (node/deno) found - YouTube downloads may fail with 403")
    return {}


def probe_duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(video)], capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def fetch_source(src, work_root, on_info):
    """Download (or locate) the video. on_info(duration, work_dir, download_bytes) is called before
    any heavy work so the caller can build the progress plan."""
    if os.path.exists(src):
        path = Path(src).resolve()
        work = work_root / re.sub(r"\W+", "_", path.stem)
        work.mkdir(parents=True, exist_ok=True)
        on_info(probe_duration(path), work, 0)
        return path, work, path.stem

    import yt_dlp

    opts = {
        "quiet": True, "no_warnings": True, "noprogress": True, "js_runtimes": js_runtimes(),
        "retries": 5, "fragment_retries": 5,
        "format": "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(src, download=False)
    work = work_root / info["id"]
    work.mkdir(parents=True, exist_ok=True)
    out = work / "source.mp4"

    # size of each stream yt-dlp will fetch (video + audio), used to weight the progress
    fmts = info.get("requested_formats") or [info]
    sizes = [f.get("filesize") or f.get("filesize_approx") or 1 for f in fmts]
    on_info(float(info.get("duration") or 0), work, 0 if out.exists() else sum(sizes))

    if not out.exists():
        log(f"Downloading: {info.get('title')}")
        seen = []

        def hook(d):
            if d["status"] not in ("downloading", "finished"):
                return
            name = d.get("filename")
            if name not in seen:
                seen.append(name)
            idx = min(seen.index(name), len(sizes) - 1)
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or sizes[idx]
            got = total if d["status"] == "finished" else d.get("downloaded_bytes", 0)
            PROGRESS("download", (sum(sizes[:idx]) + min(got / total, 1) * sizes[idx]) / sum(sizes))

        with yt_dlp.YoutubeDL({**opts, "outtmpl": str(out), "progress_hooks": [hook]}) as ydl:
            ydl.download([src])
        PROGRESS("download", 1, force=True)
    return out, work, info.get("title", info["id"])


# ---------------------------------------------------------------- 2. transcribe

def extract_audio(video, work):
    wav = work / "audio.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", "16000", str(wav)], check=True)
    return wav


def model_cache_dir(repo):
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def model_missing_bytes(repo):
    """0 if the Whisper model is already downloaded, else (bytes still to fetch, total bytes)."""
    from huggingface_hub import HfApi, snapshot_download

    try:
        snapshot_download(repo, local_files_only=True)
        return 0, 0
    except Exception:
        pass
    total = sum(s.size or 0 for s in HfApi().model_info(repo, files_metadata=True).siblings)
    folder = model_cache_dir(repo)
    have = dir_size(folder) if folder.exists() else 0
    return max(total - have, 0), total


def download_model(repo, missing, total):
    import tqdm as _tqdm
    from huggingface_hub import snapshot_download

    have = total - missing

    class Bar(_tqdm.tqdm):  # hub reports received bytes on its "Downloading bytes" bar
        def __init__(self, *a, **kw):
            super().__init__(*a, **{**kw, "file": open(os.devnull, "w")})

        def update(self, n=1):
            r = super().update(n)
            if str(self.desc).startswith("Downloading bytes") and total:
                PROGRESS("model", (have + self.n) / total)
            return r

    log("Downloading speech model (one-time)")
    PROGRESS("model", have / total if total else 0, force=True)
    snapshot_download(repo, tqdm_class=Bar)
    PROGRESS("model", 1, force=True)


def transcribe(wav, work, model):
    cache = work / f"transcript_{model}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import tqdm as _tqdm

    import mlx_whisper

    class Bar(_tqdm.tqdm):  # mirror Whisper's internal progress bar into PROGRESS
        def __init__(self, *a, **kw):
            super().__init__(*a, **{**kw, "file": open(os.devnull, "w")})

        def update(self, n=1):
            r = super().update(n)
            if self.total:
                PROGRESS("transcribe", self.n / self.total)
            return r

    sys.modules["mlx_whisper.transcribe"].tqdm = types.SimpleNamespace(tqdm=Bar)
    log(f"Transcribing with Whisper ({model})")
    PROGRESS("transcribe", 0, force=True)
    result = mlx_whisper.transcribe(str(wav), path_or_hf_repo=WHISPER_MODELS[model],
                                    word_timestamps=True, verbose=False)
    PROGRESS("transcribe", 1, force=True)
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            text = w["word"].strip()
            if text:
                words.append({"text": text, "start": float(w["start"]), "end": float(w["end"])})
    cache.write_text(json.dumps(words))
    return words


def split_sentences(words):
    sentences, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        gap = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 99
        if re.search(r"[.?!]$", w["text"]) or gap > 1.2:
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    return [{"first": s[0], "last": s[-1], "start": words[s[0]]["start"], "end": words[s[-1]]["end"],
             "text": " ".join(words[i]["text"] for i in s)} for s in sentences]


# ---------------------------------------------------------------- 3. pick moments

HOOK_WORDS = {"secret", "mistake", "never", "always", "best", "worst", "why", "how", "truth", "actually",
              "crazy", "insane", "biggest", "nobody", "everyone", "stop", "money", "million", "wrong",
              "problem", "important", "hack", "imagine", "shocking", "real", "exactly", "must"}
WEAK_STARTS = {"and", "but", "so", "because", "or", "which", "that", "then", "also"}


def loudness_curve(wav):
    with wave.open(str(wav)) as f:
        data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
        rate = f.getframerate()
    n = len(data) // rate
    rms = np.sqrt((data[: n * rate].reshape(n, rate) ** 2).mean(axis=1) + 1e-9)
    return rms / (np.median(rms) + 1e-9)  # per-second loudness relative to the median


def score_window(sents, loud):
    text = " ".join(s["text"] for s in sents).lower()
    tokens = re.findall(r"[a-z']+", text)
    first = sents[0]["text"].lower()
    dur = sents[-1]["end"] - sents[0]["start"]
    score = 50.0
    score += 10 if "?" in first else 0                                   # opens with a question
    score += 6 if re.search(r"\d", first) else 0                         # opens with a number
    score += 4 * sum(t in HOOK_WORDS for t in re.findall(r"[a-z']+", first))
    score += 60 * sum(t in HOOK_WORDS for t in tokens) / max(len(tokens), 1)
    score += 4 * min(len(tokens) / dur, 3.5)                             # speaking pace
    seg = loud[int(sents[0]["start"]): int(sents[-1]["end"]) + 1]
    score += 10 * (float(seg.mean()) - 1) if len(seg) else 0             # energy vs. rest of video
    fw = re.findall(r"[a-z']+", first)
    score -= 15 if fw and fw[0] in WEAK_STARTS else 0                    # starts mid-thought
    return max(0, min(99, round(score)))


def pick_heuristic(sentences, loud, n, min_len, max_len):
    cands = []
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            dur = sentences[j]["end"] - sentences[i]["start"]
            if dur > max_len:
                break
            if dur >= min_len:
                cands.append((score_window(sentences[i:j + 1], loud), i, j))
    cands.sort(reverse=True)
    chosen = []
    for sc, i, j in cands:
        s, e = sentences[i]["start"], sentences[j]["end"]
        if all(e <= c["start"] or s >= c["end"] for c in chosen):
            title = sentences[i]["text"][:60]
            chosen.append({"start": s, "end": e, "score": sc, "title": title, "reason": "heuristic"})
        if len(chosen) == n:
            break
    return chosen


def pick_with_claude(sentences, n, min_len, max_len, title):
    from typing import List

    import anthropic
    from pydantic import BaseModel

    class Clip(BaseModel):
        start: float
        end: float
        title: str
        hook_reason: str
        virality_score: int

    class ClipList(BaseModel):
        clips: List[Clip]

    transcript = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in sentences)
    prompt = (
        f"You are an expert short-form video editor. Below is the timestamped transcript of the video "
        f"\"{title}\". Pick the {n} best moments to post as standalone TikTok/Reels/Shorts clips.\n\n"
        f"Rules:\n- Each clip must be {min_len}-{max_len} seconds long.\n"
        f"- It must make sense on its own with no prior context, and open with a strong hook in the first "
        f"3 seconds (a bold claim, question, surprising fact or emotional moment).\n"
        f"- It should end on a payoff, not mid-thought.\n- Clips must not overlap.\n"
        f"- Use start/end times copied from the transcript line boundaries.\n"
        f"- title: a catchy caption-style title under 60 characters. virality_score: 0-99.\n\n"
        f"Transcript:\n{transcript}"
    )
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model="claude-opus-5-5",
        max_tokens=16000,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
        output_format=ClipList,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"Claude did not return clips (stop_reason={response.stop_reason})")
    return [{"start": c.start, "end": c.end, "score": c.virality_score, "title": c.title,
             "reason": c.hook_reason} for c in response.parsed_output.clips]


def snap_to_words(clip, words):
    """Snap clip edges to word boundaries so we never cut mid-word."""
    starts = [w["start"] for w in words]
    i = min(range(len(words)), key=lambda k: abs(starts[k] - clip["start"]))
    j = min(range(len(words)), key=lambda k: abs(words[k]["end"] - clip["end"]))
    clip["start"] = max(0.0, words[i]["start"] - 0.15)
    clip["end"] = words[j]["end"] + 0.35
    return clip


# ---------------------------------------------------------------- 4. reframe

def probe_size(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]


def read_frames(video, start, dur, w, h, fps, scale_w=None):
    """Yield BGR frames decoded by ffmpeg (frame-accurate seek, constant frame rate)."""
    vf = f"fps={fps}"
    if scale_w:
        h = int(round(h * scale_w / w / 2) * 2)
        w = scale_w
        vf += f",scale={w}:{h}"
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video),
                             "-t", f"{dur:.3f}", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


FACE_MODEL = str(ROOT / "models" / "face_detection_yunet_2023mar.onnx")


def track_speaker(video, start, dur, src_w, src_h, on_progress=lambda f: None):
    """Return per-output-frame crop centre x (in source pixels), following the main face."""
    aw = 480
    ah = int(round(src_h * aw / src_w / 2) * 2)
    sample_fps = 5
    detector = cv2.FaceDetectorYN.create(FACE_MODEL, "", (aw, ah), 0.7)
    xs, cuts, prev_hist, prev_thumb, target = [], set(), None, None, None
    for k, frame in enumerate(read_frames(video, start, dur, src_w, src_h, sample_fps, scale_w=aw)):
        on_progress(k / (dur * sample_fps))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
        # hard scene cut: jump the camera instead of panning. Real cuts score <=0.12 on histogram
        # correlation; lighting changes/fast motion within one shot can drop to ~0.4, so be strict.
        if (prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < 0.25
                and np.abs(thumb.astype(float) - prev_thumb).mean() > 8):
            cuts.add(k)
        prev_hist, prev_thumb = hist, thumb.astype(float)

        _, found = detector.detect(frame)
        faces = [f[:4] for f in found if f[3] >= ah / 12] if found is not None else []
        if len(faces):
            biggest = max(f[2] * f[3] for f in faces)
            big = [f for f in faces if f[2] * f[3] >= 0.5 * biggest]
            centres = [f[0] + f[2] / 2 for f in big]
            if target is not None and k not in cuts:
                target = min(centres, key=lambda c: abs(c - target))  # stick with the same person
            else:
                target = centres[int(np.argmax([f[2] * f[3] for f in big]))]
        xs.append(target)

    if not xs:
        return lambda i: src_w / 2
    first = next((x for x in xs if x is not None), aw / 2)
    filled, last = [], first
    for x in xs:
        last = x if x is not None else last
        filled.append(last)

    crop_w = min(src_w, int(round(src_h * 9 / 16 / 2) * 2))
    xs = np.array(filled, dtype=float) * src_w / aw
    path = camera_path(xs, cuts, crop_w, src_w, sample_fps)
    n = int(np.ceil(dur * FPS)) + 2
    t_frames = np.arange(n) / FPS * sample_fps  # frame index -> position in the sample timeline
    frames = np.empty(n)
    bounds = [0] + sorted(cuts) + [len(xs)]
    for a, b in zip(bounds, bounds[1:]):  # interpolate inside each shot only, so cuts stay hard cuts
        sel = (t_frames >= a) & (t_frames < b) if b < len(xs) else t_frames >= a
        frames[sel] = np.interp(t_frames[sel], np.arange(a, b), path[a:b])

    def centre(i):
        return float(frames[min(i, n - 1)])

    return centre


def camera_path(xs, cuts, crop_w, src_w, sample_fps):
    """Turn noisy per-sample face positions into a steady virtual camera, like a human operator:
    lock the shot when the speaker stays in frame, otherwise pan only when they leave a dead zone,
    with eased (low-pass) motion. Face-detector jitter never reaches the camera."""
    lo, hi = crop_w / 2, src_w - crop_w / 2
    xs = np.clip(xs, lo, hi)  # positions past the frame edge look identical, so ignore them
    out = np.empty_like(xs)
    bounds = [0] + sorted(cuts) + [len(xs)]
    alpha = 1 - np.exp(-1 / (0.8 * sample_fps))  # ~0.8s ease time constant
    for a, b in zip(bounds, bounds[1:]):
        seg = xs[a:b]
        if len(seg) == 0:
            continue
        # median filter (1s) removes one-off false detections
        k = 2
        pad = np.pad(seg, (k, k), mode="edge")
        med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(seg))])

        if med.max() - med.min() <= 0.35 * crop_w:
            out[a:b] = np.median(med)  # speaker stays put: locked-off shot, zero motion
            continue

        dz = 0.12 * crop_w  # dead zone: small moves inside the frame don't move the camera
        cam, target = med[0], np.empty_like(med)
        for i, x in enumerate(med):
            if x > cam + dz:
                cam = x - dz
            elif x < cam - dz:
                cam = x + dz
            target[i] = cam
        # forward + backward exponential smoothing = eased start and stop, no lag
        fwd = np.empty_like(target)
        fwd[0] = target[0]
        for i in range(1, len(target)):
            fwd[i] = fwd[i - 1] + alpha * (target[i] - fwd[i - 1])
        bwd = np.empty_like(fwd)
        bwd[-1] = fwd[-1]
        for i in range(len(fwd) - 2, -1, -1):
            bwd[i] = bwd[i + 1] + alpha * (fwd[i] - bwd[i + 1])
        out[a:b] = bwd
    return out


# ---------------------------------------------------------------- 5. captions

def build_chunks(words, start, end, max_words=3):
    ws = [dict(w, start=w["start"] - start, end=w["end"] - start)
          for w in words if w["start"] >= start - 0.05 and w["end"] <= end + 0.05]
    chunks, cur = [], []
    for i, w in enumerate(ws):
        cur.append(w)
        gap = ws[i + 1]["start"] - w["end"] if i + 1 < len(ws) else 99
        if len(cur) >= max_words or re.search(r"[.,?!]$", w["text"]) or gap > 0.6:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    for k, c in enumerate(chunks):
        nxt = chunks[k + 1][0]["start"] if k + 1 < len(chunks) else c[-1]["end"] + 0.5
        c_end = min(nxt, c[-1]["end"] + 0.6)
        for w in c:
            w["chunk_end"] = c_end
    return chunks


# Caption fonts: name -> (file, face index inside a .ttc collection, variable-font style name).
# Any .ttf/.otf dropped into the fonts/ folder is added automatically.
SYS = "/System/Library/Fonts/"
SUP = SYS + "Supplemental/"
BUILTIN_FONTS = {
    "Arial Black": (SUP + "Arial Black.ttf", 0, None),
    "Impact": (SUP + "Impact.ttf", 0, None),
    "SF Pro Heavy": (SYS + "SFNS.ttf", 0, "Heavy"),
    "SF Pro Rounded Heavy": (SYS + "SFNSRounded.ttf", 0, "Heavy"),
    "SF Pro Condensed Black": (SYS + "SFNS.ttf", 0, "Condensed Black"),
    "Avenir Next Heavy": (SYS + "Avenir Next.ttc", 8, None),
    "Avenir Next Condensed Heavy": (SYS + "Avenir Next Condensed.ttc", 8, None),
    "Futura Condensed ExtraBold": (SUP + "Futura.ttc", 4, None),
    "Futura Bold": (SUP + "Futura.ttc", 2, None),
    "Helvetica Bold": (SYS + "Helvetica.ttc", 1, None),
    "Arial Rounded Bold": (SUP + "Arial Rounded Bold.ttf", 0, None),
    "Trebuchet MS Bold": (SUP + "Trebuchet MS Bold.ttf", 0, None),
    "Verdana Bold": (SUP + "Verdana Bold.ttf", 0, None),
    "Tahoma Bold": (SUP + "Tahoma Bold.ttf", 0, None),
    "Georgia Bold": (SUP + "Georgia Bold.ttf", 0, None),
    "Rockwell Bold": (SUP + "Rockwell.ttc", 2, None),
    "Copperplate Bold": (SUP + "Copperplate.ttc", 2, None),
    "Phosphate Solid": (SUP + "Phosphate.ttc", 1, None),
    "Chalkboard SE Bold": (SUP + "ChalkboardSE.ttc", 2, None),
    "Noteworthy Bold": (SYS + "Noteworthy.ttc", 1, None),
    "Comic Sans MS Bold": (SUP + "Comic Sans MS Bold.ttf", 0, None),
    "Bradley Hand Bold": (SUP + "Bradley Hand Bold.ttf", 0, None),
    "Chalkduster": (SUP + "Chalkduster.ttf", 0, None),
    "Courier New Bold": (SUP + "Courier New Bold.ttf", 0, None),
}
DEFAULT_FONT = "Arial Black"


def available_fonts():
    fonts = {k: v for k, v in BUILTIN_FONTS.items() if os.path.exists(v[0])}
    for p in sorted((ROOT / "fonts").glob("*.[ot]tf")) if (ROOT / "fonts").exists() else []:
        try:
            family, style = ImageFont.truetype(str(p), 20).getname()
            fonts[f"{family} {style}".replace(" Regular", "")] = (str(p), 0, None)
        except OSError:
            pass
    return fonts


def hex_rgba(value):
    v = value.lstrip("#")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4)) + (255,)


@dataclass(frozen=True)
class CaptionStyle:
    font: str = DEFAULT_FONT
    size: int = 86               # font size in px on the 1080x1920 frame
    text_color: str = "#FFFFFF"
    highlight: str = "#FFE500"   # colour of the word being spoken
    outline_color: str = "#000000"
    outline: int = 9             # outline width in px (0 = none)
    position: int = 70           # vertical centre of the caption, % from the top
    words: int = 3               # max words on screen at once
    uppercase: bool = True
    pop: bool = True             # enlarge the spoken word briefly


@lru_cache(maxsize=32)
def font(name, size):
    path, index, variation = available_fonts().get(name) or BUILTIN_FONTS[DEFAULT_FONT]
    f = ImageFont.truetype(path, size, index=index)
    if variation:
        f.set_variation_by_name(variation)
    return f


@lru_cache(maxsize=256)
def render_caption(words, active, pop, style):
    """Render one caption state to a BGRA numpy image. words: tuple of strings."""
    f = font(style.font, style.size)
    space = f.getlength(" ")
    widths = [f.getlength(w) for w in words]
    lines, cur, cur_w = [], [], 0.0
    for i, wd in enumerate(widths):  # wrap to fit within 940px
        if cur and cur_w + space + wd > 940:
            lines.append(cur)
            cur, cur_w = [], 0.0
        cur_w += (space if cur else 0) + wd
        cur.append(i)
    lines.append(cur)

    pad = int(style.size * 0.5) + style.outline
    line_h = int(style.size * 1.25)
    canvas = Image.new("RGBA", (OUT_W, line_h * len(lines) + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    text_rgba, hi_rgba, out_rgba = hex_rgba(style.text_color), hex_rgba(style.highlight), hex_rgba(style.outline_color)
    for li, line in enumerate(lines):
        total = sum(widths[i] for i in line) + space * (len(line) - 1)
        x = (OUT_W - total) / 2
        cy = pad + li * line_h + line_h / 2
        for i in line:
            cx = x + widths[i] / 2
            is_active = i == active
            ff = font(style.font, int(style.size * pop)) if is_active and pop != 1 else f
            draw.text((cx, cy), words[i], font=ff, anchor="mm", fill=hi_rgba if is_active else text_rgba,
                      stroke_width=style.outline, stroke_fill=out_rgba)
            x += widths[i] + space
    arr = np.array(canvas)
    return arr[:, :, [2, 1, 0, 3]]  # RGBA -> BGRA


def caption_state(chunks, t, style):
    for c in chunks:
        if c[0]["start"] - 0.05 <= t < c[0]["chunk_end"]:
            active = 0
            for i, w in enumerate(c):
                if w["start"] <= t:
                    active = i
            since = t - c[active]["start"]
            pop = 1.0 + 0.18 * max(0.0, 1 - since / 0.15) if style.pop else 1.0  # quick "pop" on each word
            texts = tuple(re.sub(r"[.,!?;:]+$", "", w["text"]) for w in c)
            if style.uppercase:
                texts = tuple(t.upper() for t in texts)
            return texts, active, round(pop, 2)
    return None


def overlay(frame, bgra, centre_y):
    """Alpha-blend bgra onto frame, vertically centred on centre_y and clipped to the frame."""
    h, w = bgra.shape[:2]
    y = int(np.clip(centre_y - h / 2, 0, max(frame.shape[0] - h, 0)))
    bgra = bgra[: frame.shape[0] - y]
    h = bgra.shape[0]
    region = frame[y:y + h, :w].astype(np.float32)
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    frame[y:y + h, :w] = (region * (1 - alpha) + bgra[:, :, :3] * alpha).astype(np.uint8)


def caption_preview(style, background=None):
    """A 1080x1920 BGR frame showing the caption style, for the web page's live preview."""
    if background is None:
        grad = np.linspace(70, 25, OUT_H, dtype=np.uint8)[:, None, None]
        background = np.repeat(np.repeat(grad, OUT_W, axis=1), 3, axis=2)
    frame = background.copy()
    words = tuple("Make your captions pop like this".split())[: max(style.words, 1)]
    if style.uppercase:
        words = tuple(w.upper() for w in words)
    overlay(frame, render_caption(words, min(1, len(words) - 1), 1.0, style), OUT_H * style.position / 100)
    return frame


# ---------------------------------------------------------------- 6. render

def render_clip(video, clip, words, out_path, src_w, src_h, style, on_progress=lambda f: None):
    """on_progress(frac) reports 0..1 for this clip: face tracking ~30%, rendering ~70%."""
    start, dur = clip["start"], clip["end"] - clip["start"]
    centre = track_speaker(video, start, dur, src_w, src_h, lambda f: on_progress(0.3 * f))
    chunks = build_chunks(words, clip["start"], clip["end"], max_words=style.words)

    crop_w = min(src_w, int(round(src_h * 9 / 16 / 2) * 2))
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
         "-map", "0:v", "-map", "1:a?", "-c:v", "h264_videotoolbox", "-b:v", "10M",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
         "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE)
    cap_y = OUT_H * style.position / 100
    for i, frame in enumerate(read_frames(video, start, dur, src_w, src_h, FPS)):
        on_progress(0.3 + 0.7 * min(i / (dur * FPS), 1))
        x0 = int(np.clip(centre(i) - crop_w / 2, 0, src_w - crop_w))
        out = cv2.resize(frame[:, x0:x0 + crop_w], (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        state = caption_state(chunks, i / FPS, style)
        if state:
            overlay(out, render_caption(*state, style), cap_y)
        enc.stdin.write(out.tobytes())
    enc.stdin.close()
    enc.wait()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="YouTube URL or local video file")
    ap.add_argument("--num", type=int, default=5, help="number of clips (default 5)")
    ap.add_argument("--min", type=float, default=25, help="min clip length in seconds")
    ap.add_argument("--max", type=float, default=60, help="max clip length in seconds")
    ap.add_argument("--whisper", choices=WHISPER_MODELS, default="turbo")
    ap.add_argument("--no-llm", action="store_true", help="use the heuristic picker even if Claude is configured")
    cap = ap.add_argument_group("caption style")
    d = CaptionStyle()
    cap.add_argument("--font", default=d.font, help=f"caption font (default {d.font}); see --list-fonts")
    cap.add_argument("--font-size", type=int, default=d.size, help=f"px on the 1080x1920 frame (default {d.size})")
    cap.add_argument("--text-color", default=d.text_color, help=f"hex colour (default {d.text_color})")
    cap.add_argument("--highlight", default=d.highlight, help=f"spoken-word colour (default {d.highlight})")
    cap.add_argument("--outline-color", default=d.outline_color)
    cap.add_argument("--outline", type=int, default=d.outline, help=f"outline width px, 0 = none (default {d.outline})")
    cap.add_argument("--caption-pos", type=int, default=d.position, help=f"%% from top (default {d.position})")
    cap.add_argument("--words", type=int, default=d.words, help=f"max words on screen (default {d.words})")
    cap.add_argument("--no-uppercase", action="store_true")
    cap.add_argument("--no-pop", action="store_true", help="don't enlarge the spoken word")
    ap.add_argument("--list-fonts", action="store_true", help="print available caption fonts and exit")
    if "--list-fonts" in sys.argv:
        print("\n".join(available_fonts()))
        return
    args = ap.parse_args()

    fonts = available_fonts()
    if args.font not in fonts:
        sys.exit(f"Unknown font '{args.font}'. Available: {', '.join(fonts)}")
    for c in (args.text_color, args.highlight, args.outline_color):
        if not re.fullmatch(r"#?[0-9a-fA-F]{6}", c):
            sys.exit(f"Colours must be hex like #FFE500, got '{c}'")
    style = CaptionStyle(font=args.font, size=max(30, min(200, args.font_size)), text_color=args.text_color,
                         highlight=args.highlight, outline_color=args.outline_color,
                         outline=max(0, min(30, args.outline)), position=max(5, min(95, args.caption_pos)),
                         words=max(1, min(6, args.words)), uppercase=not args.no_uppercase, pop=not args.no_pop)

    repo = WHISPER_MODELS[args.whisper]
    use_llm = not args.no_llm and bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    model_state = {}

    def on_info(duration, work, download_bytes):
        cached = (work / f"transcript_{args.whisper}.json").exists()
        missing, total = (0, 0) if cached else model_missing_bytes(repo)
        model_state.update(missing=missing, total=total)
        clip_secs = min(duration or 600, args.num * (args.min + args.max) / 2)
        PROGRESS.plan = {  # pipeline order matters: the ETA sums the stages after the current one
            "download": download_bytes / EST_DOWNLOAD_BPS,
            "model": missing / EST_DOWNLOAD_BPS,
            "transcribe": 0 if cached else (duration or 600) / EST_TRANSCRIBE_X[args.whisper] + 5,
            "pick": 40 if use_llm else 3,
            "render": clip_secs * EST_RENDER_PER_SEC,
        }
        PROGRESS("download", 0, force=True)

    video, work, title = fetch_source(args.source, ROOT / "work", on_info)
    if model_state.get("missing"):
        download_model(repo, model_state["missing"], model_state["total"])
    wav = extract_audio(video, work)
    words = transcribe(wav, work, args.whisper)
    if not words:
        sys.exit("No speech found in this video.")
    sentences = split_sentences(words)

    PROGRESS("pick", 0, force=True)
    clips = None
    if use_llm:
        try:
            log("Asking Claude to pick the best moments")
            clips = pick_with_claude(sentences, args.num, args.min, args.max, title)
        except Exception as e:  # fall back rather than lose the whole run
            log(f"Claude picker failed ({e}); using heuristic")
    if not clips:
        log("Scoring moments with the built-in heuristic")
        clips = pick_heuristic(sentences, loudness_curve(wav), args.num, args.min, args.max)
    clips = sorted((snap_to_words(c, words) for c in clips), key=lambda c: -c["score"])

    out_dir = ROOT / "clips" / work.name
    out_dir.mkdir(parents=True, exist_ok=True)
    src_w, src_h = probe_size(video)
    total_secs = sum(c["end"] - c["start"] for c in clips)
    PROGRESS.plan["render"] = total_secs * EST_RENDER_PER_SEC  # now we know the real clip lengths
    done_secs = 0.0
    for n, clip in enumerate(clips, 1):
        path = out_dir / f"clip_{n:02d}.mp4"
        log(f"Rendering {path.name}  [{clip['start']:.1f}s - {clip['end']:.1f}s]  score {clip['score']}  "
            f"\"{clip['title']}\"")
        dur = clip["end"] - clip["start"]
        render_clip(video, clip, words, path, src_w, src_h, style,
                    lambda f, d=done_secs, dur=dur: PROGRESS("render", (d + f * dur) / total_secs))
        done_secs += dur
        clip["file"] = str(path)
    PROGRESS("render", 1, force=True)
    (out_dir / "clips.json").write_text(json.dumps(clips, indent=2))
    log(f"Done - {len(clips)} clips in {out_dir}")


if __name__ == "__main__":
    main()
