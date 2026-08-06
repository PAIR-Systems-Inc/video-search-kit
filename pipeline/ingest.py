#!/usr/bin/env python3
"""
ingest.py — load transcripts (and video titles) into GoodMem.

For each data/transcripts/<video_id>.json:
  1. Group its timestamped segments into ~CHUNK_WINDOW-second windows with
     CHUNK_OVERLAP seconds of overlap (chunker.py).
  2. Store one GoodMem memory per chunk:
       content  = the chunk's transcript text
       metadata = video_id, title, url, published_at, start, end, ...
  3. Also store one memory per VIDEO TITLE in the title space — a strong title
     match boosts every chunk of that video at search time.

Safe to re-run: memory IDs are deterministic (uuid5 of space/video/chunk), so
retries can never create duplicates; per-video progress is checkpointed in
data/ingest.state.json.

Deliberately throttled by default (--batch-size 50 --batch-delay 2): feeding a
GoodMem instance tens of thousands of memories at full speed can starve its
embedding worker. Slow and steady completes; fast wedges.

Usage:
  set -a; . ./.env; set +a
  python pipeline/ingest.py --dry-run          # chunk counts only, no writes
  python pipeline/ingest.py
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import uuid

from chunker import chunk_transcript


def chunk_memory_id(space_id: str, video_id: str, chunk_index) -> str:
    """Deterministic UUID so re-sending a chunk can never create a duplicate."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL,
                          f"video-search-kit/{space_id}/{video_id}/{chunk_index}"))


def load_video_meta(csv_path: str) -> dict:
    meta = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            if vid:
                meta[vid] = {
                    "title": row.get("title", ""),
                    "url": row.get("url") or f"https://www.youtube.com/watch?v={vid}",
                    "published_at": row.get("published_at", ""),
                }
    return meta


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"videos": {}, "titles": {}}


def save_state(state, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def build_chunk_requests(video_id, transcript, vmeta, window, overlap):
    chunks = chunk_transcript(transcript, window_seconds=window, overlap_seconds=overlap)
    for i, chunk in enumerate(chunks):
        yield chunk["text"], {
            "video_id": video_id,
            "title": vmeta.get("title", ""),
            "url": vmeta.get("url", f"https://www.youtube.com/watch?v={video_id}"),
            "published_at": vmeta.get("published_at", ""),
            "start": round(float(chunk["start"]), 2),
            "end": round(float(chunk["end"]), 2),
            "chunk_index": i,
            "chunk_count": len(chunks),
            "language": transcript.get("language", ""),
            "video_duration": transcript.get("duration", 0),
        }


def inspect_batch_response(resp):
    """Count real per-item failures; 'already exists' = idempotent retry = OK."""
    items = None
    for attr in ("results", "memories", "items", "responses"):
        candidate = getattr(resp, attr, None)
        if isinstance(candidate, list) and candidate:
            items = candidate
            break
    if items is None:
        return 0, "unverified-shape"
    failed, notes = 0, []
    for it in items:
        err = getattr(it, "error", None) or getattr(it, "error_message", None)
        status = str(getattr(it, "status", "") or "")
        blob = f"{err or ''} {status}".upper()
        has_memory = getattr(it, "memory", None) is not None or getattr(it, "memory_id", None)
        if err or ("FAIL" in blob or "ERROR" in blob) or (not has_memory and blob.strip()):
            if "ALREADY_EXISTS" in blob or "ALREADY EXISTS" in blob or "409" in blob:
                continue
            failed += 1
            if len(notes) < 3:
                notes.append(str(err or status)[:120])
    return failed, "; ".join(notes)


def main():
    parser = argparse.ArgumentParser(description="Transcripts + titles -> GoodMem")
    parser.add_argument("--csv", default=os.environ.get("VIDEOS_CSV", "data/videos.csv"))
    parser.add_argument("--transcripts-dir",
                        default=os.environ.get("TRANSCRIPTS_DIR", "data/transcripts"))
    parser.add_argument("--state-file", default="data/ingest.state.json")
    parser.add_argument("--space-id", default=os.environ.get("GOODMEM_SPACE_ID"))
    parser.add_argument("--title-space-id", default=os.environ.get("GOODMEM_TITLE_SPACE_ID"))
    parser.add_argument("--window", type=float,
                        default=float(os.environ.get("CHUNK_WINDOW", "60")))
    parser.add_argument("--overlap", type=float,
                        default=float(os.environ.get("CHUNK_OVERLAP", "15")))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-delay", type=float, default=2.0,
                        help="Seconds between batches — protects the embedding worker")
    parser.add_argument("--skip-titles", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    video_meta = load_video_meta(args.csv) if os.path.exists(args.csv) else {}
    state = load_state(args.state_file)
    state.setdefault("titles", {})

    todo = []
    for fn in sorted(os.listdir(args.transcripts_dir)):
        if fn.endswith(".json"):
            vid = fn[:-5]
            if state["videos"].get(vid, {}).get("status") not in ("ingested", "empty"):
                todo.append(vid)
    if args.limit:
        todo = todo[:args.limit]
    done_already = sum(1 for s in state["videos"].values()
                       if s.get("status") in ("ingested", "empty"))
    print(f"{len(todo)} video(s) to ingest ({done_already} already done).", flush=True)

    if args.dry_run:
        total = 0
        for vid in todo:
            try:
                with open(os.path.join(args.transcripts_dir, f"{vid}.json"),
                          encoding="utf-8") as f:
                    t = json.load(f)
                n = len(list(build_chunk_requests(vid, t, video_meta.get(vid, {}),
                                                  args.window, args.overlap)))
            except Exception as e:
                print(f"  {vid}: PARSE ERROR: {e}")
                continue
            total += n
        print(f"Total: {total} chunks across {len(todo)} videos. No writes (dry run).")
        return

    if not args.space_id:
        sys.exit("GOODMEM_SPACE_ID not set — run pipeline/setup.py first.")
    base_url = os.environ.get("GOODMEM_BASE_URL")
    api_key = os.environ.get("GOODMEM_API_KEY")
    if not base_url or not api_key:
        sys.exit("Set GOODMEM_BASE_URL and GOODMEM_API_KEY in .env first.")

    from goodmem import Goodmem
    from goodmem.api.memories import MemoryCreationRequest

    n_videos = n_chunks = 0
    with Goodmem(base_url=base_url, api_key=api_key, timeout=120.0) as client:

        def send_batches(pairs, space_id, vid, prev):
            """Send (content, metadata) pairs; resume from prev checkpoint."""
            bs = args.batch_size
            n_batches = math.ceil(len(pairs) / bs)
            start = min(int(prev.get("batches_done", 0)), n_batches)
            for b in range(start, n_batches):
                batch = pairs[b * bs:(b + 1) * bs]
                resp = client.memories.batch_create(requests=[
                    MemoryCreationRequest(
                        space_id=space_id,
                        memory_id=chunk_memory_id(space_id, vid, md["chunk_index"]),
                        original_content=content,
                        metadata=md,
                    ) for content, md in batch
                ])
                failed, notes = inspect_batch_response(resp)
                if failed:
                    raise RuntimeError(f"{failed} item(s) failed in batch {b}: {notes}")
                state["videos"][vid] = {"status": "partial", "batches_done": b + 1}
                save_state(state, args.state_file)
                if args.batch_delay:
                    time.sleep(args.batch_delay)

        # ---- transcript chunks --------------------------------------------
        for vid in todo:
            prev = state["videos"].get(vid, {})
            try:
                with open(os.path.join(args.transcripts_dir, f"{vid}.json"),
                          encoding="utf-8") as f:
                    transcript = json.load(f)
                vmeta = video_meta.get(vid)
                if vmeta is None:
                    print(f"  [{vid}] WARNING: not in {args.csv} — no title/date on cards",
                          flush=True)
                    vmeta = {}
                pairs = list(build_chunk_requests(vid, transcript, vmeta,
                                                  args.window, args.overlap))
            except Exception as e:
                state["videos"][vid] = {"status": "error", "stage": "parse",
                                        "message": str(e)[:300]}
                save_state(state, args.state_file)
                print(f"  [{vid}] PARSE ERROR: {e}", flush=True)
                continue

            if not pairs:
                state["videos"][vid] = {"status": "empty"}
                save_state(state, args.state_file)
                continue
            try:
                send_batches(pairs, args.space_id, vid, prev)
            except Exception as e:
                state["videos"][vid] = {"status": "error",
                                        "batches_done": state["videos"].get(vid, {}).get("batches_done", 0),
                                        "message": str(e)[:300]}
                save_state(state, args.state_file)
                print(f"  [{vid}] BATCH ERROR (will resume): {e}", flush=True)
                continue
            state["videos"][vid] = {"status": "ingested", "chunks": len(pairs)}
            save_state(state, args.state_file)
            n_videos += 1
            n_chunks += len(pairs)
            if n_videos % 25 == 0:
                print(f"  {n_videos}/{len(todo)} videos, {n_chunks} chunks", flush=True)

        # ---- titles --------------------------------------------------------
        if not args.skip_titles:
            if not args.title_space_id:
                print("GOODMEM_TITLE_SPACE_ID not set — skipping title boost ingest.")
            else:
                requests = []
                for vid, s in state["videos"].items():
                    if s.get("status") != "ingested":
                        continue
                    if state["titles"].get(vid) == "done":
                        continue
                    meta = video_meta.get(vid)
                    if not meta or not meta.get("title"):
                        continue
                    requests.append((vid, MemoryCreationRequest(
                        space_id=args.title_space_id,
                        memory_id=chunk_memory_id(args.title_space_id, vid, "title"),
                        original_content=meta["title"],
                        metadata={"video_id": vid, "title": meta["title"],
                                  "url": meta["url"],
                                  "published_at": meta.get("published_at", ""),
                                  "kind": "title"},
                    )))
                print(f"{len(requests)} title memories to create.", flush=True)
                bs = args.batch_size
                for i in range(0, len(requests), bs):
                    batch = requests[i:i + bs]
                    resp = client.memories.batch_create(requests=[r for _, r in batch])
                    failed, notes = inspect_batch_response(resp)
                    if failed:
                        print(f"  title batch {i//bs}: {failed} failed: {notes}", flush=True)
                    else:
                        for vid, _ in batch:
                            state["titles"][vid] = "done"
                        save_state(state, args.state_file)
                    if args.batch_delay:
                        time.sleep(args.batch_delay)

    print(f"Done: {n_videos} videos, {n_chunks} chunks ingested this run.", flush=True)
    print("GoodMem embeds asynchronously — give it a few minutes, then check "
          "`/api/health?deep=1` on the webapp (title_space_ready).")


if __name__ == "__main__":
    main()
