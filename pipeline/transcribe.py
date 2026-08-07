#!/usr/bin/env python3
"""
transcribe.py — produce timestamped transcripts for every video in videos.csv.

Two sources, in order:
  1. YouTube captions (default, free, instant) — fetched with
     youtube-transcript-api when the video has them. Disable with --whisper-only.
  2. Whisper — downloads the audio with yt-dlp, compresses it with ffmpeg,
     splits long audio to fit the 25 MB API limit, transcribes each piece via
     any OpenAI-compatible endpoint (WHISPER_BASE_URL / WHISPER_API_KEY /
     WHISPER_MODEL), then stitches the timestamped segments back together.

Output: data/transcripts/<video_id>.json in the kit's canonical shape:
  {"language": "...", "duration": <sec>, "segments": [{"start","end","text"}]}

Resumable: data/transcribe.state.json tracks done/failed per video; re-running
only processes what's missing. Requires `yt-dlp` and `ffmpeg` on PATH for the
Whisper path.

Usage:
  set -a; . ./.env; set +a
  python pipeline/transcribe.py --estimate            # cost preview, no work
  python pipeline/transcribe.py                       # captions first, whisper fallback
  python pipeline/transcribe.py --whisper-only --workers 2
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

AUDIO_KBPS = 32          # opus mono — ~14 MB/hour, clean enough for ASR
PIECE_SECONDS = 1200     # 20-min pieces ≈ 5 MB each, well under the 25 MB cap


def ytdlp_bin() -> str:
    """Prefer a yt-dlp installed next to this Python (venv) — YouTube changes
    constantly and an outdated system-wide yt-dlp fails in confusing ways."""
    cand = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    return cand if os.path.exists(cand) else "yt-dlp"


def ytdlp_cookie_args() -> list:
    """
    YouTube increasingly answers datacenter/flagged IPs with 'Sign in to
    confirm you're not a bot'. The standard remedy is to reuse your browser's
    logged-in session:  set YTDLP_COOKIES_FROM_BROWSER=chrome (or firefox,
    edge, ...) in .env, with that browser logged into YouTube.
    """
    browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    return ["--cookies-from-browser", browser] if browser else []


# ------------------------------------------------------------------ state
def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"videos": {}}


def save_state(state, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


# ------------------------------------------------------------------ captions
class CaptionFetchBlocked(Exception):
    """Captions could not be FETCHED (IP block, network) — distinct from a
    video genuinely having no captions. Must not be silently treated as
    'no captions' or a blocked IP looks like an entire channel without them."""


def try_captions(video_id: str):
    """Canonical transcript dict from YouTube captions; None if the video has
    none; raises CaptionFetchBlocked when YouTube refuses the request."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            NoTranscriptFound, TranscriptsDisabled, VideoUnavailable)
    except ImportError:
        return None
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        return None                       # genuinely no captions to be had
    except Exception as e:
        raise CaptionFetchBlocked(f"{type(e).__name__}: {str(e)[:150]}")
    segments = [{"start": round(s.start, 2),
                 "end": round(s.start + s.duration, 2),
                 "text": s.text.replace("\n", " ").strip()}
                for s in fetched if s.text.strip()]
    if not segments:
        return None
    return {"language": getattr(fetched, "language_code", "") or "",
            "duration": segments[-1]["end"],
            "source": "captions",
            "segments": segments}


# ------------------------------------------------------------------ whisper
def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-300:]}")
    return p


def whisper_transcribe(video_id: str, cfg: dict) -> dict:
    """yt-dlp -> ffmpeg -> (split) -> Whisper API -> stitched canonical dict."""
    with tempfile.TemporaryDirectory(prefix=f"vsk-{video_id}-") as tmp:
        raw = os.path.join(tmp, "raw.m4a")
        run([ytdlp_bin(), "-f", "bestaudio", "-o", raw, "--no-playlist",
             *ytdlp_cookie_args(),
             f"https://www.youtube.com/watch?v={video_id}"])
        audio = os.path.join(tmp, "audio.ogg")
        run(["ffmpeg", "-y", "-i", raw, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libopus", "-b:a", f"{AUDIO_KBPS}k", audio])

        # split into fixed-length pieces so every upload fits the API limit
        pieces_dir = os.path.join(tmp, "pieces")
        os.makedirs(pieces_dir)
        run(["ffmpeg", "-y", "-i", audio, "-f", "segment",
             "-segment_time", str(PIECE_SECONDS), "-c", "copy",
             os.path.join(pieces_dir, "p%04d.ogg")])
        pieces = sorted(os.listdir(pieces_dir))

        segments, language = [], ""
        for n, piece in enumerate(pieces):
            offset = n * PIECE_SECONDS
            with open(os.path.join(pieces_dir, piece), "rb") as f:
                resp = httpx.post(
                    f"{cfg['base_url'].rstrip('/')}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {cfg['api_key']}"},
                    data={"model": cfg["model"], "response_format": "verbose_json"},
                    files={"file": (piece, f, "audio/ogg")},
                    timeout=600.0,
                )
            resp.raise_for_status()
            body = resp.json()
            language = language or body.get("language", "")
            for s in body.get("segments", []):
                text = (s.get("text") or "").strip()
                if text:
                    segments.append({"start": round(offset + s["start"], 2),
                                     "end": round(offset + s["end"], 2),
                                     "text": text})
        if not segments:
            raise RuntimeError("Whisper returned no segments")
        return {"language": language, "duration": segments[-1]["end"],
                "source": "whisper", "segments": segments}


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description="videos.csv -> timestamped transcripts")
    parser.add_argument("--csv", default=os.environ.get("VIDEOS_CSV", "data/videos.csv"))
    parser.add_argument("--out-dir", default=os.environ.get("TRANSCRIPTS_DIR",
                                                            "data/transcripts"))
    parser.add_argument("--state-file", default="data/transcribe.state.json")
    parser.add_argument("--whisper-only", action="store_true",
                        help="Skip the captions attempt")
    parser.add_argument("--captions-only", action="store_true",
                        help="Never call Whisper (videos without captions stay pending)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--estimate", action="store_true",
                        help="Print an audio-hours/cost estimate and exit")
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        videos = [r for r in csv.DictReader(f) if r.get("video_id")]

    if args.estimate:
        secs = sum(float(v.get("duration") or 0) for v in videos)
        hours = secs / 3600
        print(f"{len(videos)} videos, ~{hours:.1f} hours of audio "
              f"({sum(1 for v in videos if not v.get('duration'))} without a known duration).")
        print(f"Whisper cost if ALL of it is transcribed (captions usually cover much of it):")
        print(f"  OpenAI whisper-1        ~${hours*60*0.006:,.0f}")
        print(f"  Groq whisper-large-v3   ~${hours*0.111:,.0f}")
        return

    whisper_cfg = None
    if not args.captions_only:
        key = os.environ.get("WHISPER_API_KEY", "").strip()
        if key:
            whisper_cfg = {"base_url": os.environ.get("WHISPER_BASE_URL",
                                                      "https://api.openai.com/v1"),
                           "api_key": key,
                           "model": os.environ.get("WHISPER_MODEL", "whisper-1")}
        elif args.whisper_only:
            sys.exit("--whisper-only needs WHISPER_API_KEY in .env")
        for tool in ("yt-dlp", "ffmpeg"):
            if whisper_cfg and shutil.which(tool) is None:
                sys.exit(f"{tool} not found on PATH — required for the Whisper path")

    os.makedirs(args.out_dir, exist_ok=True)
    state = load_state(args.state_file)
    lock = threading.Lock()

    todo = [v for v in videos
            if state["videos"].get(v["video_id"], {}).get("status") != "done"
            and not os.path.exists(os.path.join(args.out_dir, f"{v['video_id']}.json"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(videos)} videos in CSV; {len(todo)} to transcribe.", flush=True)

    counts = {"captions": 0, "whisper": 0, "pending": 0, "failed": 0, "blocked": 0}
    blocked_note = []

    def work(v):
        vid = v["video_id"]
        result = None
        if not args.whisper_only:
            try:
                result = try_captions(vid)
            except CaptionFetchBlocked as e:
                if not blocked_note:
                    blocked_note.append(str(e))
                    print(f"  WARNING: caption fetch is being refused "
                          f"({e}) — falling back to Whisper where configured. "
                          f"This is usually an IP block, NOT missing captions.",
                          flush=True)
                if whisper_cfg is None or args.captions_only:
                    raise
        if result is None and whisper_cfg is not None and not args.captions_only:
            result = whisper_transcribe(vid, whisper_cfg)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, v): v for v in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            v = futures[fut]
            vid = v["video_id"]
            try:
                result = fut.result()
            except CaptionFetchBlocked as e:
                with lock:
                    counts["blocked"] += 1
                    state["videos"][vid] = {"status": "pending",
                                            "note": f"caption fetch blocked: {e}"}
                    save_state(state, args.state_file)
                continue
            except Exception as e:
                with lock:
                    counts["failed"] += 1
                    state["videos"][vid] = {"status": "failed", "error": str(e)[:300]}
                    save_state(state, args.state_file)
                print(f"  [{vid}] FAILED: {str(e)[:120]}", flush=True)
                continue
            with lock:
                if result is None:
                    counts["pending"] += 1
                    state["videos"][vid] = {"status": "pending",
                                            "note": "no captions; whisper not configured"}
                else:
                    path = os.path.join(args.out_dir, f"{vid}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False)
                    counts[result["source"]] += 1
                    state["videos"][vid] = {"status": "done", "source": result["source"]}
                save_state(state, args.state_file)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {counts}", flush=True)

    print(f"Done: {counts}", flush=True)


if __name__ == "__main__":
    main()
