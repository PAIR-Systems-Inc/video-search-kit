#!/usr/bin/env python3
"""
keyword_index.py — local exact-keyword index over the transcript chunks.

Vector search + reranker rank by *meaning*, so a query like "osama bin laden"
retrieves the semantically-near "Usama ibn Zayd" and buries the actual
"bin laden" mentions (which live inside videos about other topics). This index
guarantees that any chunk containing ALL query terms verbatim can be surfaced,
regardless of what the reranker thinks — the direct-hit rule the product wants.

It rebuilds the SAME chunks the ingest used (deterministic chunker, same
window/overlap) so a keyword hit maps to the same timestamped card GoodMem
would return. Built once in a background thread so the server starts instantly.
"""

import csv
import json
import os
import re
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "pipeline"))
from chunker import chunk_transcript  # noqa: E402

_WORD = re.compile(r"[0-9a-z']+")


def load_video_meta(csv_path: str) -> dict:
    """video_id -> {title, published_at}. Mirrors pipeline/ingest.py."""
    meta = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = (row.get("video_id") or "").strip()
            if vid:
                meta[vid] = {"title": row.get("title", ""),
                             "published_at": row.get("published_at", "")}
    return meta


class KeywordIndex:
    def __init__(self, transcripts_dir: str, csv_path: str,
                 window: float, overlap: float):
        self.chunks = []        # parallel list of chunk dicts
        self.postings = {}      # term -> set of chunk indices
        self.ready = False
        self.error = None
        self._t = threading.Thread(
            target=self._build, args=(transcripts_dir, csv_path, window, overlap),
            daemon=True)
        self._t.start()

    def _build(self, tdir, csv_path, window, overlap):
        try:
            video_meta = load_video_meta(csv_path) if os.path.exists(csv_path) else {}
            for fn in sorted(os.listdir(tdir)):
                if not fn.endswith(".json"):
                    continue
                vid = fn[:-5]
                try:
                    with open(os.path.join(tdir, fn), encoding="utf-8") as f:
                        tr = json.load(f)
                    pieces = chunk_transcript(tr, window_seconds=window,
                                              overlap_seconds=overlap)
                except Exception:
                    continue
                meta = video_meta.get(vid, {})
                title = meta.get("title", "")
                for p in pieces:
                    idx = len(self.chunks)
                    self.chunks.append({
                        "score": None,           # no vector/rerank score
                        "video_id": vid,
                        "title": title,
                        "published_at": meta.get("published_at", ""),
                        "start": float(p["start"]),
                        "end": float(p["end"]),
                        "text": p["text"],
                        "thumbnail": f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg",
                    })
                    for w in set(_WORD.findall((title + " " + p["text"]).lower())):
                        self.postings.setdefault(w, set()).add(idx)
            self.ready = True
        except Exception as e:  # never let index failure take down search
            self.error = str(e)

    def search(self, terms: list, limit: int = 12) -> list:
        """Chunks containing EVERY term, ranked by total term occurrences."""
        if not self.ready or not terms:
            return []
        sets = [self.postings.get(t) for t in terms]
        if not all(sets):
            return []
        hits = set.intersection(*sets)
        scored = []
        for i in hits:
            blob = (self.chunks[i]["title"] + " " + self.chunks[i]["text"]).lower()
            occ = sum(len(re.findall(r"\b" + re.escape(t) + r"\b", blob)) for t in terms)
            scored.append((occ, i))
        scored.sort(reverse=True)
        return [dict(self.chunks[i]) for _, i in scored[:limit]]
