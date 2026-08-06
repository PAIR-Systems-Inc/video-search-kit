#!/usr/bin/env python3
"""
validate.py — check a bring-your-own dataset before ingesting it.

Verifies:
  - videos.csv exists, has the required columns, no duplicate video_ids
  - every transcript file parses, has non-empty segments with numeric
    start < end and non-empty text
  - cross-references the two: transcripts without CSV metadata (cards would
    show a raw video id), CSV rows without transcripts (won't be searchable)

Exit code 0 = safe to ingest (warnings allowed), 1 = hard errors found.

Usage:
  python pipeline/validate.py --csv data/videos.csv --transcripts-dir data/transcripts
"""

import argparse
import csv
import json
import os
import sys

REQUIRED_COLS = {"video_id", "title", "url"}


def main():
    parser = argparse.ArgumentParser(description="Validate a BYO dataset")
    parser.add_argument("--csv", default=os.environ.get("VIDEOS_CSV", "data/videos.csv"))
    parser.add_argument("--transcripts-dir",
                        default=os.environ.get("TRANSCRIPTS_DIR", "data/transcripts"))
    args = parser.parse_args()

    errors, warnings = [], []

    # ---- videos.csv --------------------------------------------------------
    csv_ids = set()
    if not os.path.exists(args.csv):
        errors.append(f"{args.csv} not found")
    else:
        with open(args.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = REQUIRED_COLS - set(reader.fieldnames or [])
            if missing:
                errors.append(f"{args.csv}: missing required column(s): {sorted(missing)}")
            else:
                for n, row in enumerate(reader, 2):
                    vid = (row.get("video_id") or "").strip()
                    if not vid:
                        warnings.append(f"{args.csv}:{n}: empty video_id — row skipped at ingest")
                        continue
                    if vid in csv_ids:
                        errors.append(f"{args.csv}:{n}: duplicate video_id {vid}")
                    csv_ids.add(vid)
                    if not (row.get("title") or "").strip():
                        warnings.append(f"{args.csv}:{n}: {vid} has no title")
        print(f"videos.csv: {len(csv_ids)} unique videos")

    # ---- transcripts -------------------------------------------------------
    t_ids = set()
    if not os.path.isdir(args.transcripts_dir):
        errors.append(f"{args.transcripts_dir} not found")
    else:
        files = [f for f in sorted(os.listdir(args.transcripts_dir)) if f.endswith(".json")]
        ok = 0
        for fn in files:
            vid = fn[:-5]
            path = os.path.join(args.transcripts_dir, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    t = json.load(f)
            except Exception as e:
                errors.append(f"{fn}: invalid JSON ({str(e)[:80]})")
                continue
            segs = t.get("segments")
            if not isinstance(segs, list) or not segs:
                errors.append(f"{fn}: no segments[] — nothing to index")
                continue
            bad = 0
            for i, s in enumerate(segs):
                try:
                    st, en = float(s["start"]), float(s["end"])
                    if not (st < en) or not str(s.get("text", "")).strip():
                        bad += 1
                except (KeyError, TypeError, ValueError):
                    bad += 1
            if bad:
                if bad == len(segs):
                    errors.append(f"{fn}: all {bad} segments malformed")
                    continue
                warnings.append(f"{fn}: {bad}/{len(segs)} malformed segment(s) — they'll be skipped")
            t_ids.add(vid)
            ok += 1
        print(f"transcripts: {ok}/{len(files)} files valid")

    # ---- cross-reference ---------------------------------------------------
    if csv_ids and t_ids:
        no_meta = sorted(t_ids - csv_ids)
        no_transcript = sorted(csv_ids - t_ids)
        if no_meta:
            warnings.append(f"{len(no_meta)} transcript(s) missing from videos.csv "
                            f"(cards will show the raw id): {no_meta[:5]}{'...' if len(no_meta) > 5 else ''}")
        if no_transcript:
            print(f"note: {len(no_transcript)} CSV video(s) have no transcript yet "
                  f"— they just won't be searchable.")

    print()
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    if errors:
        print(f"\n{len(errors)} hard error(s) — fix these before ingesting.")
        sys.exit(1)
    print(f"OK — dataset is ingestable ({len(warnings)} warning(s)).")


if __name__ == "__main__":
    main()
