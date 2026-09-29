# Clipper AI

Paste a YouTube link, get vertical short clips (9:16) with animated word-by-word captions — a small, local take on tools like OpusClip. Everything runs on your Mac.

## How it works

1. **Download** the video with yt-dlp.
2. **Transcribe** with Whisper (MLX, optimised for Apple Silicon) to get a timestamp for every word.
3. **Pick the best moments** — a built-in scorer ranks windows by hook strength, pace and loudness. If `ANTHROPIC_API_KEY` is set, Claude picks them instead.
4. **Reframe to 9:16** — a YuNet face detector follows the speaker; the virtual camera holds still, pans smoothly and cuts only at real scene cuts.
5. **Caption** — 3 words at a time, the spoken word highlighted in yellow with a pop.
6. **Encode** with the Mac's hardware H.264 encoder.

## Requirements

- macOS on Apple Silicon (M1 or newer) — transcription uses MLX
- Python 3.11+
- [FFmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)
- Node.js or Deno — yt-dlp needs a JavaScript runtime for YouTube, otherwise downloads fail with HTTP 403

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The Whisper model (~1.6 GB, or ~0.5 GB in fast mode) downloads automatically on first use.

## Usage

### Web page

```bash
.venv/bin/python app.py
```

Open http://localhost:8765, paste a link and press **Get clips**. The page shows overall progress with an ETA, then lets you preview and download each clip.

### Command line

```bash
.venv/bin/python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID" --num 5
```

| Option | Default | Meaning |
|---|---|---|
| `--num` | 5 | number of clips |
| `--min` / `--max` | 25 / 60 | clip length range in seconds |
| `--whisper` | `turbo` | `small` is faster but less accurate |
| `--no-llm` | off | use the built-in scorer even if Claude is configured |

A local video file works in place of a URL. Output goes to `clips/<video-id>/` along with `clips.json` (titles, scores, timestamps). Downloads and transcripts are cached in `work/`, so re-running a video skips straight to rendering.

### Better clip picking with Claude (optional)

```bash
export ANTHROPIC_API_KEY=your-key
```

## Credits

Face detection model: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) from OpenCV Zoo (MIT licence), included in `models/`.
