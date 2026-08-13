#!/usr/bin/env python3
"""
server.py — video-search-kit backend with title-weighted retrieval.

GET /api/search?q=...&k=10
  Runs TWO GoodMem queries in parallel:
    - transcript-chunk space (body matches)
    - title space (one memory per video, content = the video title)
  Scores are min-max normalized within each list and blended per chunk:
      combined = TITLE_WEIGHT * title_sim(video) + BODY_WEIGHT * body_sim(chunk)
  (defaults 0.6 / 0.4). Chunks are re-ranked by the blended score, then
  same-video adjacent/overlapping windows are merged into single results with
  timestamped YouTube links.

GET /api/answer?q=...&k=8
  Same fused + merged retrieval, then the top sources are sent to a small LLM
  via OpenRouter to synthesize an answer whose [n] citations refer directly to
  the returned `sources` list (numbering is identical by construction).

Failures return JSON with HTTP 200 ({"error": ...}) because proxies
(Cloudflare tunnel) replace 5xx bodies with their own HTML page.

Run:
  export GOODMEM_BASE_URL=... GOODMEM_API_KEY=gm_... GOODMEM_SPACE_ID=...
  export GOODMEM_TITLE_SPACE_ID=...   # optional: enables title boosting
  export OPENROUTER_API_KEY=sk-or-... # optional: enables /api/answer
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from goodmem import Goodmem
from pydantic import BaseModel

GOODMEM_BASE_URL = os.environ.get("GOODMEM_BASE_URL", "http://localhost:8080")
GOODMEM_API_KEY = os.environ.get("GOODMEM_API_KEY", "")
# Set GOODMEM_TLS_VERIFY=0 for local instances with self-signed certificates.
GOODMEM_TLS_VERIFY = os.environ.get("GOODMEM_TLS_VERIFY", "1") != "0"
GOODMEM_SPACE_ID = os.environ.get("GOODMEM_SPACE_ID", "")
GOODMEM_TITLE_SPACE_ID = os.environ.get("GOODMEM_TITLE_SPACE_ID", "")
TITLE_WEIGHT = float(os.environ.get("TITLE_WEIGHT", "0.6"))
BODY_WEIGHT = float(os.environ.get("BODY_WEIGHT", "0.4"))
# Titles below this absolute similarity give no boost — vector search always
# returns the 10 "nearest" titles even when none genuinely relate to the query.
TITLE_SIM_FLOOR = float(os.environ.get("TITLE_SIM_FLOOR", "0.4"))
# Optional Voyage reranker registered in GoodMem. When set, both retrievals
# rerank their vector candidates and relevance_score becomes a calibrated
# 0-1 relevance (higher = better) instead of a vector distance.
GOODMEM_RERANKER_ID = os.environ.get("GOODMEM_RERANKER_ID", "")
# Results whose best signal (max of body/title relevance) falls below the floor
# are dropped — vector search always returns SOMETHING, and off-topic queries
# otherwise render confident-looking cards. Reranker and raw-vector scores live
# on different scales, so there are two floors and the active one is chosen by
# whether the reranker is enabled. (Both measured to cleanly separate off-topic
# ≤~0.56 from genuine matches ≥~0.68.)
RELEVANCE_FLOOR = float(os.environ.get("RELEVANCE_FLOOR", "0.6"))                    # reranker on
RELEVANCE_FLOOR_NORERANK = float(os.environ.get("RELEVANCE_FLOOR_NORERANK", "0.6"))  # reranker off
# Retrieve a WIDE pool from the embedder and let the reranker rerank all of it
# (don't cut to a small top-k) — a genuinely relevant chunk that the embedder
# ranks deep still gets a fair reranker judgement.
RETRIEVE_POOL = int(os.environ.get("RETRIEVE_POOL", "200"))
TITLE_RETRIEVE = int(os.environ.get("TITLE_RETRIEVE", "20"))
# Local exact-keyword index: guarantees chunks containing ALL query terms
# surface regardless of vector/rerank ranking (the direct-match rule).
KEYWORD_INJECT_MAX = int(os.environ.get("KEYWORD_INJECT_MAX", "8"))
ENABLE_KEYWORD_INDEX = os.environ.get("ENABLE_KEYWORD_INDEX", "1") == "1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "openai/gpt-4.1-mini")
# Surfaced in the page header (and /api/health) so tester feedback can be tied
# to a known build. Bump APP_VERSION whenever retrieval behaviour changes
# (weights, floors, pool size, keyword rules).
APP_STAGE = os.environ.get("APP_STAGE", "Alpha")
APP_VERSION = os.environ.get("APP_VERSION", "0.1")
# Site branding — the page fetches these from /api/health and fills itself in,
# so the static HTML stays generic.
SITE_NAME = os.environ.get("SITE_NAME", "Video Search")
SITE_TAGLINE = os.environ.get(
    "SITE_TAGLINE",
    "Ask a question — every result links to the exact moment in the video it came from.")
EXAMPLE_QUERY = os.environ.get("EXAMPLE_QUERY",
                               "e.g. What does the speaker say about patience?")

STATIC_DIR = Path(__file__).parent / "static"
# Data paths in .env are written relative to the repo root; the server runs
# from webapp/, so resolve non-absolute paths against the repo root.
_ROOT = Path(__file__).resolve().parent.parent


def _rooted(p: str) -> str:
    return p if os.path.isabs(p) else str(_ROOT / p)

# Build the local keyword index once, in a background thread (see keyword_index).
# It re-chunks data/transcripts with the same window/overlap the ingest used,
# so a keyword hit maps to the same timestamped card GoodMem would return.
_KW_INDEX = None
if ENABLE_KEYWORD_INDEX:
    try:
        from keyword_index import KeywordIndex
        _KW_INDEX = KeywordIndex(
            _rooted(os.environ.get("TRANSCRIPTS_DIR", "data/transcripts")),
            _rooted(os.environ.get("VIDEOS_CSV", "data/videos.csv")),
            float(os.environ.get("CHUNK_WINDOW", "60")),
            float(os.environ.get("CHUNK_OVERLAP", "15")),
        )
    except Exception:
        _KW_INDEX = None

app = FastAPI(title=SITE_NAME)


def _to_plain_dict(obj) -> dict:
    """Metadata may arrive as a dict or a pydantic-style object — normalize."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                pass
    return dict(getattr(obj, "__dict__", {}))


def _fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _overlap_join(a: str, b: str, min_overlap_chars: int = 10) -> str:
    """
    Concatenate two chunk texts, deduplicating the shared tail/head.
    Consecutive overlapping chunks are built from the same segment texts, so
    the overlap region is an exact string match: the longest suffix of `a`
    that is a prefix of `b`. Matches shorter than min_overlap_chars are
    treated as coincidence (e.g. a shared word or letter) and NOT deduped —
    stripping them would delete genuinely spoken text.
    """
    a, b = a.strip(), b.strip()
    for k in range(min(len(a), len(b)), min_overlap_chars - 1, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + " " + b


def _merge_adjacent(results: list, max_gap: float = 1.0) -> list:
    """
    Merge chunks from the same video whose time windows overlap or touch
    (within max_gap seconds) into one result spanning the combined window.
    Groups keep the ranking slot of their best-ranked member (`orig_pos` =
    1-based position in the fused ranking).
    """
    by_video = {}
    for r in results:
        by_video.setdefault(r["video_id"], []).append(r)

    merged_all = []
    for members in by_video.values():
        members.sort(key=lambda r: r["start"])
        clusters = []
        for r in members:
            if clusters and r["start"] <= clusters[-1]["end"] + max_gap:
                c = clusters[-1]
                if r["end"] > c["end"]:
                    if r["start"] < c["end"]:
                        # True time overlap — shared segments, safe to dedup.
                        c["text"] = _overlap_join(c["text"], r["text"])
                    else:
                        # Windows only touch — no shared text, plain join.
                        c["text"] = c["text"].strip() + " " + r["text"].strip()
                    c["end"] = r["end"]
                c["members"].append(r["orig_pos"])
                # If any window of this video is a direct keyword hit, the card is.
                c["keyword_match"] = c.get("keyword_match") or r.get("keyword_match")
                if r["orig_pos"] < c["best_rank"]:
                    c["best_rank"] = r["orig_pos"]
                    for f in ("score", "combined_score", "title_sim", "body_sim", "source"):
                        if f in r:
                            c[f] = r[f]
            else:
                clusters.append({**r, "members": [r["orig_pos"]],
                                 "best_rank": r["orig_pos"]})
        merged_all.extend(clusters)

    merged_all.sort(key=lambda c: c["best_rank"])
    out = []
    for c in merged_all:
        start_s = max(0, int(c["start"]))
        video_id = c["video_id"]
        c.update({
            "start_label": _fmt_ts(c["start"]),
            "end_label": _fmt_ts(c["end"]),
            "watch_url": f"https://www.youtube.com/watch?v={video_id}&t={start_s}s",
            "embed_url": f"https://www.youtube.com/embed/{video_id}?start={start_s}&autoplay=1",
            "merged_count": len(c["members"]),
        })
        c.pop("members")
        c.pop("best_rank")
        c.pop("orig_pos", None)
        out.append(c)
    return out


def _group_by_video(merged: list) -> list:
    """
    One card per video. The best-ranked window leads the card; other windows
    of the same video become small "also at" alternate moments (max 3).
    A TITLE-DRIVEN video (title contribution to the blend >= body contribution)
    is presented as the whole video: timestamps are dropped and links play
    from the start.
    """
    cards = []
    by_vid = {}
    for m in merged:  # merged is already in fused-rank order
        card = by_vid.get(m["video_id"])
        if card is None:
            card = {**m, "moments": []}
            by_vid[m["video_id"]] = card
            cards.append(card)
        elif len(card["moments"]) < 3:
            card["moments"].append({
                "start": m["start"],
                "start_label": m["start_label"],
                "end_label": m["end_label"],
                "watch_url": m["watch_url"],
            })

    for c in cards:
        title_part = TITLE_WEIGHT * c.get("title_sim", 0.0)
        body_part = BODY_WEIGHT * c.get("body_sim", 0.0)
        c["title_match"] = c.get("title_sim", 0.0) > 0 and title_part >= body_part
        if c["title_match"]:
            vid = c["video_id"]
            c["watch_url"] = f"https://www.youtube.com/watch?v={vid}"
            c["embed_url"] = f"https://www.youtube.com/embed/{vid}?autoplay=1"
            c["moments"] = []
    return cards


def _retrieve(q: str, k: int, space_id: str, use_reranker: bool = False):
    """One GoodMem retrieval from `space_id`; returns joined results."""
    kwargs = dict(message=q, space_ids=[space_id], fetch_memory=True, stream=False)
    if use_reranker:
        # Embedder returns k candidates; the reranker rescores ALL of them and
        # we return ALL (max_results=k) — don't cut to a small top slice.
        kwargs.update(reranker_id=GOODMEM_RERANKER_ID,
                      requested_size=k,
                      max_results=k,
                      chronological_resort=False)
    else:
        kwargs.update(requested_size=k)
    try:
        with Goodmem(base_url=GOODMEM_BASE_URL, api_key=GOODMEM_API_KEY,
                     timeout=60.0, verify=GOODMEM_TLS_VERIFY) as client:
            events = client.memories.retrieve(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GoodMem retrieval failed: {e}")

    meta_by_memory = {}
    for ev in events:
        mem = getattr(ev, "memory_definition", None) or getattr(ev, "memory", None)
        if mem is not None:
            mem_id = str(getattr(mem, "memory_id", "") or "")
            if mem_id:
                meta_by_memory[mem_id] = _to_plain_dict(getattr(mem, "metadata", None))

    results = []
    for ev in events:
        item = getattr(ev, "retrieved_item", None)
        if item is None:
            continue
        ref = getattr(item, "chunk", None)
        if ref is None:
            continue
        chunk = getattr(ref, "chunk", None) or ref
        text = getattr(chunk, "chunk_text", None) or getattr(ref, "chunk_text", "") or ""
        mem_id = str(getattr(chunk, "memory_id", "") or getattr(ref, "memory_id", "") or "")
        score = getattr(ref, "relevance_score", None)

        md = meta_by_memory.get(mem_id, {})
        video_id = md.get("video_id", "")
        if not video_id:
            continue  # metadata join failed — can't build a timestamped link
        start = float(md.get("start", 0) or 0)
        end = float(md.get("end", 0) or 0)

        results.append({
            "text": text,
            "score": score,
            "memory_id": mem_id,
            "video_id": video_id,
            "title": md.get("title", ""),
            "published_at": md.get("published_at", ""),
            "start": start,
            "end": end,
            "thumbnail": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
        })
    return results


# English function words dropped from keyword matching so a question like
# "how do I..." doesn't force spurious matches. Name particles like "bin",
# "ibn", "al", "abu" are deliberately NOT here — they carry meaning in names.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "how", "do", "does", "did", "i", "we", "what", "why", "when",
    "who", "my", "me", "your", "you", "it", "that", "this", "with", "about",
    "can", "should", "would", "as", "at", "by", "be", "so",
}


def _query_terms(q: str) -> list:
    """Meaningful lowercase tokens from the query for exact keyword matching."""
    toks = re.findall(r"[0-9a-z']+", q.lower())
    terms = [t for t in toks if t not in _STOPWORDS and len(t) > 1]
    return terms or toks  # if a query is ALL stopwords, keep the raw tokens


def _keyword_match(terms: list, *texts) -> bool:
    """
    True only if EVERY query term appears as a whole word in the combined
    title/text. A direct hit like all of {osama, bin, laden} present outranks
    whatever the reranker thinks (it may prefer the semantically-near but wrong
    "Usama bin Zayd"). Requiring ALL terms keeps this from firing on stray
    single-word overlaps.
    """
    if not terms:
        return False
    blob = " ".join(t.lower() for t in texts if t)
    return all(re.search(r"\b" + re.escape(t) + r"\b", blob) for t in terms)


def _abs_sim(score, reranked: bool = False) -> float:
    """
    Convert a GoodMem relevance_score to an absolute [0,1] similarity.
    - Vector path: score is a pgvector negative-inner-product distance and
      Voyage embeddings are normalized, so -score is the cosine similarity.
    - Reranker path: score is already a calibrated relevance (higher=better).
    Absolute values (not per-list min-max) keep the title/body blend honest:
    an unrelated "nearest" match keeps a low sim instead of being forced to 1.
    """
    value = (score or 0.0) if reranked else -(score or 0.0)
    return min(1.0, max(0.0, value))


def _fused_search(q: str):
    """
    Retrieve a wide body pool + title matches (reranked, not truncated), blend
    scores (TITLE_WEIGHT / BODY_WEIGHT), inject local exact-keyword hits, gate
    on the relevance floor (keyword hits bypass it), and rank by the blend.
    """
    rerank = bool(GOODMEM_RERANKER_ID)
    terms = _query_terms(q)

    # Retrieve a WIDE pool from the embedder and rerank all of it (don't cut).
    with ThreadPoolExecutor(max_workers=2) as pool:
        body_f = pool.submit(_retrieve, q, RETRIEVE_POOL, GOODMEM_SPACE_ID, rerank)
        title_f = (pool.submit(_retrieve, q, TITLE_RETRIEVE, GOODMEM_TITLE_SPACE_ID, rerank)
                   if GOODMEM_TITLE_SPACE_ID else None)
        body = body_f.result()
        titles = title_f.result() if title_f else []


    def _scale_is_reranked(results):
        """
        Trust the DATA, not the request: some server versions silently skip
        the reranker above a pool-size cap and return raw vector scores.
        Reranker scores are calibrated [0,1]; pgvector similarity scores are
        negative inner products — the sign of the first score tells which
        scale actually came back.
        """
        for r in results:
            if r.get("score") is not None:
                return r["score"] >= 0
        return False

    body_reranked = rerank and _scale_is_reranked(body)
    titles_reranked = rerank and _scale_is_reranked(titles)

    title_sim = {}
    for r in titles:
        sim = _abs_sim(r["score"], titles_reranked)
        if sim >= TITLE_SIM_FLOOR:
            title_sim[r["video_id"]] = max(title_sim.get(r["video_id"], 0.0), sim)

    def annotate(r, reranked):
        r["body_sim"] = round(_abs_sim(r["score"], reranked), 4)
        r["title_sim"] = round(title_sim.get(r["video_id"], 0.0), 4)
        r["combined_score"] = round(BODY_WEIGHT * r["body_sim"]
                                    + TITLE_WEIGHT * r["title_sim"], 4)
        r["keyword_match"] = _keyword_match(terms, r.get("title", ""), r.get("text", ""))
        return r

    for r in body:
        annotate(r, body_reranked)
        r["source"] = "reranker" if rerank else "embedder"

    # Local exact-keyword index runs BEFORE the threshold cut: any chunk with
    # every query term is injected and flagged, so the floor never removes a
    # direct match even if the embedder/reranker never surfaced it.
    seen = {(r["video_id"], round(r["start"])) for r in body}
    if _KW_INDEX is not None and terms:
        injected = 0
        for r in _KW_INDEX.search(terms, limit=KEYWORD_INJECT_MAX * 2):
            if injected >= KEYWORD_INJECT_MAX:
                break
            key = (r["video_id"], round(r["start"]))
            if key in seen:
                continue
            annotate(r, False)          # no score → body_sim 0, but flagged match
            r["source"] = "index"
            body.append(r)
            seen.add(key)
            injected += 1

    if not body:
        return []

    # Confidence gate: drop chunks that aren't genuinely relevant — UNLESS every
    # query term appears verbatim (keyword_match), which bypasses the floor.
    floor = RELEVANCE_FLOOR if body_reranked else RELEVANCE_FLOOR_NORERANK
    body = [r for r in body
            if r["keyword_match"] or max(r["body_sim"], r["title_sim"]) >= floor]
    if not body:
        return []

    # Direct keyword matches lead (a query-term match beats any reranker score),
    # then order by the blended title/body score.
    body.sort(key=lambda r: (r["keyword_match"], r["combined_score"]), reverse=True)
    for i, r in enumerate(body, start=1):
        r["orig_pos"] = i
    return body


def _generate_answer(q: str, sources: list) -> str:
    excerpts = []
    for i, s in enumerate(sources, 1):
        # Be honest with the LLM about scope: a title-matched video is cited
        # whole, but the excerpt itself is still one window of it.
        if s.get("title_match"):
            where = (f"video matched by title; excerpt from "
                     f"{s.get('start_label', '?')}–{s.get('end_label', '?')}")
        else:
            where = f"{s['start_label']}–{s['end_label']}"
        excerpts.append(f"[{i}] \"{s['title']}\" ({where})\n{s['text'][:1500]}")
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": ANSWER_MODEL,
            "temperature": 0.3,
            "max_tokens": 700,
            "messages": [
                {"role": "system", "content":
                    f"You answer questions about {SITE_NAME}'s video content using ONLY "
                    "the provided video excerpts. Cite the excerpts you draw on inline with their "
                    "bracketed numbers, e.g. [2] or [1, 3]. Be concise and faithful to the excerpts; "
                    "if they are insufficient to answer, say so briefly."},
                {"role": "user", "content":
                    f"Question: {q}\n\nVideo excerpts:\n\n" + "\n\n".join(excerpts)},
            ],
        },
        timeout=90.0,
    )
    resp.raise_for_status()
    data = resp.json()
    # OpenRouter can return HTTP 200 with an error body, or a null message.
    if data.get("error"):
        raise RuntimeError(data["error"].get("message") or str(data["error"]))
    choices = data.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    if not content:
        raise RuntimeError("model returned no text")
    return content.strip()


def _require_config():
    if not GOODMEM_API_KEY or not GOODMEM_SPACE_ID:
        raise HTTPException(
            status_code=500,
            detail="Server not configured: set GOODMEM_BASE_URL, GOODMEM_API_KEY, GOODMEM_SPACE_ID.",
        )


@app.get("/api/search")
def search(q: str = Query(..., min_length=2), k: int = Query(10, ge=1, le=50)):
    t0 = time.perf_counter()
    try:
        _require_config()
        fused = _fused_search(q)
    except HTTPException as e:
        return {"query": q, "count": 0, "results": [],
                "error": f"{e.detail} — the GoodMem backend may be restarting; try again shortly."}
    results = _group_by_video(_merge_adjacent(fused))[:k]
    return {"query": q, "count": len(results), "results": results,
            "took_ms": int((time.perf_counter() - t0) * 1000)}


class AnswerRequest(BaseModel):
    q: str
    sources: list


@app.get("/api/answer")
def answer_get(q: str = Query(..., min_length=2), k: int = Query(8, ge=1, le=20)):
    """
    Backwards-compatible variant that does its own retrieval — kept so pages
    loaded before the POST flow shipped (stale tabs on the shared tunnel URL)
    still get answers.
    """
    t0 = time.perf_counter()
    if not OPENROUTER_API_KEY:
        return {"query": q, "answer": None, "disabled": True, "sources": []}
    try:
        _require_config()
        fused = _fused_search(q)
    except HTTPException as e:
        return {"query": q, "answer": None, "sources": [],
                "error": f"{e.detail} — the GoodMem backend may be restarting; try again shortly."}
    merged = _group_by_video(_merge_adjacent(fused))[:k]
    if not merged:
        return {"query": q, "answer": None, "sources": [], "error": "No relevant video moments found."}
    try:
        answer_text = _generate_answer(q, merged)
    except Exception as e:
        return {"query": q, "answer": None, "sources": merged,
                "error": f"Answer generation failed: {e}"}
    return {"query": q, "answer": answer_text, "error": None, "sources": merged,
            "took_ms": int((time.perf_counter() - t0) * 1000)}


@app.post("/api/answer")
def answer(req: AnswerRequest):
    """
    Generate the AI summary from sources the client ALREADY retrieved and is
    displaying — no second retrieval, so the results never wait on the LLM
    and [n] citations match the visible cards by construction.
    """
    t0 = time.perf_counter()
    if not OPENROUTER_API_KEY:
        return {"query": req.q, "answer": None, "disabled": True}
    sources = [s for s in req.sources
               if isinstance(s, dict) and s.get("text") and s.get("title") is not None][:10]
    if not sources:
        return {"query": req.q, "answer": None, "error": "No sources provided."}
    try:
        answer_text = _generate_answer(req.q, sources)
    except Exception as e:
        return {"query": req.q, "answer": None, "error": f"Answer generation failed: {e}"}
    return {"query": req.q, "answer": answer_text, "error": None,
            "took_ms": int((time.perf_counter() - t0) * 1000)}


@app.get("/api/health")
def health(deep: bool = Query(False)):
    payload = {
        "ok": True,
        "stage": APP_STAGE,
        "version": APP_VERSION,
        "site_name": SITE_NAME,
        "site_tagline": SITE_TAGLINE,
        "example_query": EXAMPLE_QUERY,
        "configured": bool(GOODMEM_API_KEY and GOODMEM_SPACE_ID),
        "title_boost_configured": bool(GOODMEM_TITLE_SPACE_ID),
        "weights": {"title": TITLE_WEIGHT, "body": BODY_WEIGHT,
                    "title_sim_floor": TITLE_SIM_FLOOR,
                    "relevance_floor": RELEVANCE_FLOOR if GOODMEM_RERANKER_ID
                                       else RELEVANCE_FLOOR_NORERANK},
        "reranker": bool(GOODMEM_RERANKER_ID),
        "answer_enabled": bool(OPENROUTER_API_KEY),
        "answer_model": ANSWER_MODEL if OPENROUTER_API_KEY else None,
        "goodmem_base_url": GOODMEM_BASE_URL,
    }
    if deep and GOODMEM_TITLE_SPACE_ID:
        # Config presence doesn't prove the boost works (memories may be
        # unembedded) — probe with a real retrieval.
        try:
            payload["title_space_ready"] = bool(_retrieve("test probe", 1, GOODMEM_TITLE_SPACE_ID))
        except HTTPException as e:
            payload["title_space_ready"] = False
            payload["title_space_error"] = str(e.detail)
    return payload


@app.get("/healthz")
def healthz():
    """
    Liveness probe required by the pairhub contract (docs/pairhub-contract.md):
    unauthenticated, returns 200. Deliberately does NOT touch GoodMem or wait on
    the keyword index — the container is up and serving as soon as uvicorn binds,
    and a transient GoodMem outage must not get the site restart-looped.
    Use /api/health for the configuration/readiness detail.
    """
    return {"ok": True}


# Must stay LAST: this mount catches every unmatched path, so any route
# declared after it would be shadowed by the static handler.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
