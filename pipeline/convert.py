#!/usr/bin/env python3
"""
convert.py — turn SRT/VTT subtitle files into the kit's canonical transcripts.

If you already have subtitles per video, name each file after its video id
(<video_id>.srt or <video_id>.vtt), point this script at the directory, and it
writes data/transcripts/<video_id>.json ready for ingest.

Usage:
  python pipeline/convert.py path/to/subs/ --out-dir data/transcripts
"""

import argparse
import json
import os
import re
import sys

TS = r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
CUE = re.compile(rf"{TS}\s*-->\s*{TS}")
TAG = re.compile(r"<[^>]+>")          # strip inline styling tags


def ts_to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_subtitles(text: str) -> list:
    """Shared SRT/VTT cue parser — both formats are 'timestamp line + text'."""
    segments = []
    cur = None
    for raw in text.splitlines():
        line = raw.strip("﻿").strip()
        m = CUE.search(line)
        if m:
            if cur and cur["text"]:
                segments.append(cur)
            g = m.groups()
            cur = {"start": round(ts_to_seconds(*g[:4]), 2),
                   "end": round(ts_to_seconds(*g[4:]), 2), "text": ""}
        elif cur is not None:
            if not line:                       # blank line ends the cue
                if cur["text"]:
                    segments.append(cur)
                cur = None
            elif line.isdigit() or line.upper().startswith(("WEBVTT", "NOTE", "STYLE")):
                continue
            else:
                clean = TAG.sub("", line).strip()
                if clean:
                    cur["text"] = (cur["text"] + " " + clean).strip()
    if cur and cur["text"]:
        segments.append(cur)
    return segments


def main():
    parser = argparse.ArgumentParser(description="SRT/VTT -> canonical transcripts")
    parser.add_argument("subs_dir", help="Directory of <video_id>.srt / .vtt files")
    parser.add_argument("--out-dir", default=os.environ.get("TRANSCRIPTS_DIR",
                                                            "data/transcripts"))
    args = parser.parse_args()

    files = [f for f in sorted(os.listdir(args.subs_dir))
             if f.lower().endswith((".srt", ".vtt"))]
    if not files:
        sys.exit(f"No .srt/.vtt files in {args.subs_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    ok = bad = 0
    for fn in files:
        vid = os.path.splitext(fn)[0]
        with open(os.path.join(args.subs_dir, fn), encoding="utf-8", errors="replace") as f:
            segments = parse_subtitles(f.read())
        if not segments:
            print(f"  {fn}: no cues parsed — skipped")
            bad += 1
            continue
        out = {"language": "", "duration": segments[-1]["end"],
               "source": "subtitles", "segments": segments}
        with open(os.path.join(args.out_dir, f"{vid}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        ok += 1

    print(f"Converted {ok} file(s) -> {args.out_dir}" + (f", {bad} skipped" if bad else ""))


if __name__ == "__main__":
    main()
