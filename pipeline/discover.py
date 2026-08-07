#!/usr/bin/env python3
"""
discover.py — turn a YouTube channel (or playlist) into data/videos.csv.

Accepts any of:
  https://www.youtube.com/@SomeHandle          @SomeHandle
  https://www.youtube.com/channel/UCxxxx       UCxxxx
  https://www.youtube.com/playlist?list=PL...  PL...

Requires a (free) YouTube Data API v3 key — set YOUTUBE_API_KEY in .env.
Get one: Google Cloud Console -> enable "YouTube Data API v3" -> Credentials.
Cheap on quota: 1 unit per 50 videos (default daily quota is 10,000 units).

Output CSV columns: video_id,title,published_at,url,duration,description

Usage:
  set -a; . ./.env; set +a
  python pipeline/discover.py "https://www.youtube.com/@SomeChannel" --out data/videos.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, parse_qs
from urllib.request import urlopen

API_BASE = "https://www.googleapis.com/youtube/v3"


# ---------------------------------------------------------------- input parse
def parse_source(src: str) -> dict:
    """Normalize a channel/playlist reference into {kind, value}."""
    src = src.strip()
    if src.startswith("@"):
        return {"kind": "handle", "value": src}
    if re.fullmatch(r"UC[0-9A-Za-z_-]{22}", src):
        return {"kind": "channel_id", "value": src}
    if re.fullmatch(r"(PL|UU|OL)[0-9A-Za-z_-]+", src):
        return {"kind": "playlist", "value": src}
    if "youtube.com" in src or "youtu.be" in src:
        u = urlparse(src if "://" in src else "https://" + src)
        qs = parse_qs(u.query)
        if "list" in qs:
            return {"kind": "playlist", "value": qs["list"][0]}
        parts = [p for p in u.path.split("/") if p]
        if parts:
            if parts[0].startswith("@"):
                return {"kind": "handle", "value": parts[0]}
            if parts[0] == "channel" and len(parts) > 1:
                return {"kind": "channel_id", "value": parts[1]}
            if parts[0] in ("c", "user") and len(parts) > 1:
                return {"kind": "name", "value": parts[1]}
    sys.exit(f"Could not understand source: {src!r} — pass a channel URL, @handle, "
             f"channel ID (UC...), or playlist ID/URL.")


# ---------------------------------------------------------------- API mode
def api_get(key, endpoint, params, retries=5):
    params = {**params, "key": key}
    url = f"{API_BASE}/{endpoint}?{urlencode(params)}"
    delay = 1.0
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code in (403, 429, 500, 503) and attempt < retries - 1:
                time.sleep(delay); delay *= 2; continue
            body = e.read().decode(errors="ignore")[:300]
            sys.exit(f"YouTube API error {e.code} on {endpoint}: {body}")
        except URLError:
            if attempt < retries - 1:
                time.sleep(delay); delay *= 2; continue
            raise


def iso8601_duration_to_seconds(d: str) -> float:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m:
        return 0.0
    h, mi, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mi * 60 + s


def discover_api(source: dict, key: str) -> list:
    if source["kind"] == "playlist":
        playlist_id = source["value"]
    else:
        params = {"part": "contentDetails"}
        if source["kind"] == "handle":
            params["forHandle"] = source["value"]
        elif source["kind"] == "channel_id":
            params["id"] = source["value"]
        else:
            params["forUsername"] = source["value"]
        items = api_get(key, "channels", params).get("items", [])
        if not items:
            sys.exit(f"No channel found for {source['value']!r}")
        playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    videos, token = [], None
    while True:
        params = {"part": "snippet,contentDetails", "playlistId": playlist_id,
                  "maxResults": 50}
        if token:
            params["pageToken"] = token
        data = api_get(key, "playlistItems", params)
        for it in data.get("items", []):
            sn = it["snippet"]
            vid = it["contentDetails"]["videoId"]
            videos.append({
                "video_id": vid,
                "title": sn.get("title", ""),
                "published_at": it["contentDetails"].get("videoPublishedAt",
                                                         sn.get("publishedAt", "")),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "duration": "",
                "description": sn.get("description", ""),
            })
        token = data.get("nextPageToken")
        print(f"  {len(videos)} videos...", flush=True)
        if not token:
            break

    # durations, batched 50 at a time (needed for transcription cost estimates)
    by_id = {v["video_id"]: v for v in videos}
    ids = list(by_id)
    for i in range(0, len(ids), 50):
        data = api_get(key, "videos", {"part": "contentDetails",
                                       "id": ",".join(ids[i:i + 50])})
        for it in data.get("items", []):
            by_id[it["id"]]["duration"] = \
                iso8601_duration_to_seconds(it["contentDetails"].get("duration"))
    return videos


def main():
    parser = argparse.ArgumentParser(description="Channel/playlist -> videos.csv")
    parser.add_argument("source", help="Channel URL, @handle, channel ID, or playlist")
    parser.add_argument("--out", default="data/videos.csv")
    args = parser.parse_args()

    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        sys.exit("YOUTUBE_API_KEY is not set — add it to .env (free key: enable "
                 "'YouTube Data API v3' in Google Cloud Console).")
    videos = discover_api(parse_source(args.source), key)

    if not videos:
        sys.exit("No videos found.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "title", "published_at",
                                          "url", "duration", "description"])
        w.writeheader()
        w.writerows(videos)
    total = sum(float(v["duration"] or 0) for v in videos)
    print(f"\nWrote {len(videos)} videos to {args.out}"
          + (f" (~{total/3600:.1f} hours of content)" if total else ""))


if __name__ == "__main__":
    main()
