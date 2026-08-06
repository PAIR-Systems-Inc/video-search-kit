#!/usr/bin/env python3
"""
chunker.py

Group a transcript's sentence-level segments into retrieval-sized chunks while
preserving exact timestamps.

Each transcript JSON (transcripts/<video_id>.json) has:
    {"language", "duration", "text", "segments": [{"id", "start", "end", "text"}], ...}

A chunk is a run of consecutive segments. Its `start` is the first segment's
start and its `end` is the last segment's end, so every chunk maps to a precise
window of the video. Consecutive chunks overlap by `overlap_segments` segments
so a thought split across a boundary is still retrievable from either side.
"""


def _num(value, default: float = 0.0) -> float:
    """Coerce a segment timestamp to float; some payloads carry null/strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def chunk_segments(segments: list, max_chars: int = 1000, min_chars: int = 250,
                   overlap_segments: int = 1) -> list:
    """Return a list of {start, end, text, segment_ids} chunks."""
    runs = []
    cur = []
    cur_len = 0

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if cur and cur_len + len(text) + 1 > max_chars and cur_len >= min_chars:
            runs.append(cur)
            cur = cur[-overlap_segments:] if overlap_segments > 0 else []
            cur_len = sum(len((s.get("text") or "").strip()) + 1 for s in cur)
        cur.append(seg)
        cur_len += len(text) + 1

    if cur:
        # Avoid emitting a chunk that is only the overlap tail of the previous one.
        is_pure_overlap = runs and len(cur) <= overlap_segments and \
            all(s in runs[-1] for s in cur)
        if not is_pure_overlap:
            runs.append(cur)

    chunks = []
    for run in runs:
        chunks.append({
            "start": _num(run[0].get("start")),
            "end": _num(run[-1].get("end")),
            "text": " ".join((s.get("text") or "").strip() for s in run),
            "segment_ids": [s.get("id") for s in run],
        })
    return chunks


def chunk_segments_by_time(segments: list, window_seconds: float = 180.0,
                           overlap_seconds: float = 30.0) -> list:
    """
    Time-window chunking: each chunk covers up to `window_seconds` of video,
    and consecutive chunks overlap by ~`overlap_seconds` (realized at segment
    boundaries, so actual overlap is the closest achievable without splitting
    a sentence).
    """
    segs = [s for s in segments if (s.get("text") or "").strip()]
    if not segs:
        return []

    chunks = []
    i = 0
    n = len(segs)
    while i < n:
        chunk_start = _num(segs[i].get("start"))
        j = i
        while j < n and _num(segs[j].get("end")) - chunk_start <= window_seconds:
            j += 1
        if j == i:  # single segment longer than the window — take it whole
            j = i + 1
        run = segs[i:j]
        chunk_end = _num(run[-1].get("end"))
        chunks.append({
            "start": _num(run[0].get("start")),
            "end": chunk_end,
            "text": " ".join((s.get("text") or "").strip() for s in run),
            "segment_ids": [s.get("id") for s in run],
        })
        if j >= n:
            break
        # Next chunk begins at the first segment starting inside the overlap
        # tail; fall back to j (no overlap) so progress is always guaranteed.
        target = chunk_end - overlap_seconds
        i = next((idx for idx in range(i + 1, j)
                  if _num(segs[idx].get("start")) >= target), j)
    return chunks


def chunk_transcript(transcript: dict, **kwargs) -> list:
    """Chunk a full transcript payload; falls back to whole text if no segments."""
    segments = transcript.get("segments") or []
    if segments:
        return chunk_segments_by_time(segments, **kwargs)
    text = (transcript.get("text") or "").strip()
    if not text:
        return []
    # No segment timing available — one chunk spanning the whole video.
    return [{
        "start": 0.0,
        "end": _num(transcript.get("duration")),
        "text": text,
        "segment_ids": [],
    }]
