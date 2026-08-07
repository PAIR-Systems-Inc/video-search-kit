# video-search-kit

Turn any YouTube channel (or your own transcripts) into a searchable site where
**every answer links to the exact moment in the video it came from** — backed by
[GoodMem](https://goodmem.ai) semantic search, a reranker, an exact-keyword
override, and an optional AI summary.

```
discover ─▶ videos.csv ─▶ transcribe ─▶ transcripts/*.json ─▶ ingest ─▶ GoodMem ─▶ webapp
(channel URL)            (captions/Whisper)  (canonical shape)     (chunks+titles)
```

You can enter at **any stage**: bring a channel URL and let the pipeline do
everything, or bring your own timestamped transcripts and jump straight to
[ingest](#3-bring-your-own-transcripts).

---

## 0. Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r webapp/requirements.txt -r pipeline/requirements.txt

cp .env.example .env        # fill in the (required) values — see comments inside
set -a; . ./.env; set +a    # load it into the shell (repeat after edits)

python pipeline/setup.py --prefix mychannel
# prints GOODMEM_SPACE_ID / GOODMEM_TITLE_SPACE_ID / GOODMEM_RERANKER_ID
# -> paste them into .env
```

What goes in `.env` (details in `.env.example`):

| Key | What it's for | Required |
|---|---|---|
| `GOODMEM_BASE_URL`, `GOODMEM_API_KEY` | your GoodMem instance | yes |
| `EMBEDDER_API_KEY` (+ model) | registered into GoodMem by setup.py | yes |
| `RERANKER_API_KEY` (+ model) | registered into GoodMem by setup.py | recommended |
| `OPENROUTER_API_KEY` (+ model) | the AI summary (any OpenAI-compatible API) | optional |
| `WHISPER_API_KEY` (+ base URL, model) | transcription | only for transcribe |
| `YOUTUBE_API_KEY` | channel discovery | required for discover |

## 1. Discover — channel ➜ video list

```bash
python pipeline/discover.py "https://www.youtube.com/@SomeChannel"
# accepts @handle, channel URL/ID, or a playlist URL/ID
```

Writes `data/videos.csv`: `video_id,title,published_at,url,duration,description`.

**Requires a free `YOUTUBE_API_KEY`**
([get one](https://console.cloud.google.com/apis/library/youtube.googleapis.com):
enable *YouTube Data API v3* → Credentials → API key, then put it in `.env`).
Discovery uses the official API only — it's stable, fast (1 quota unit per 50
videos against a 10,000/day free quota), and returns the publish dates and
durations that the result cards and the `--estimate` cost preview depend on.

## 2. Transcribe — videos ➜ timestamped transcripts

```bash
python pipeline/transcribe.py --estimate     # audio hours + Whisper cost preview
python pipeline/transcribe.py                # captions first, Whisper fallback
```

- **Captions first** (free, instant) for videos that have them; **Whisper** for
  the rest using *your* key. **A plain OpenAI API key works as-is** — the
  defaults already target OpenAI's `whisper-1`; just set `WHISPER_API_KEY=sk-...`.
  Any other OpenAI-compatible endpoint works too (e.g. Groq: set
  `WHISPER_BASE_URL=https://api.groq.com/openai/v1`,
  `WHISPER_MODEL=whisper-large-v3-turbo`). `--whisper-only` / `--captions-only`
  narrow the behaviour.
**Why does this need anything installed — isn't transcription an API call?**
The transcription itself *is* API-side. But the Whisper API accepts an **audio
file upload, not a URL** — you cannot send it a YouTube link — and YouTube has
no API that returns audio (the Data API is metadata-only). So the machine
running the pipeline is the middleman: it downloads the audio (`yt-dlp`),
shrinks it to speech-grade mono and splits anything long into 20-minute pieces
because the API rejects files over 25 MB (`ffmpeg`), uploads the pieces, and
stitches the timestamped results back together.

```
YouTube ──(yt-dlp)──▶ this machine ──(ffmpeg: shrink+split)──▶ Whisper API ──▶ transcript
```

**Install requirements, per path:**

| Path | Needs installed | Needs a key |
|---|---|---|
| Captions | nothing | nothing |
| Whisper (YouTube) | `ffmpeg` (e.g. `sudo apt install ffmpeg`) + `yt-dlp` (pip) | `WHISPER_API_KEY` |
| Whisper (local files, `--media`) | `ffmpeg` only | `WHISPER_API_KEY` |
| Bring-your-own transcripts | nothing | nothing |

These are **pipeline-machine dependencies only** — the deployed website needs
none of them.
- Resumable: re-running skips what's done (`data/transcribe.state.json`).
- Note: downloading audio from YouTube is generally only appropriate for
  content you own or have rights to — the intended use is your own channel.

**Local files instead of YouTube** — transcribe any audio/video files ffmpeg
can read (MP4, MKV, MP3, WAV, …), no YouTube involved:

```bash
python pipeline/transcribe.py --media path/to/talk.mp4 path/to/interview.mp3
```

Each file becomes `data/transcripts/<filename-slug>.json`. Add a matching row
to `videos.csv` (`video_id` = the filename slug) if you want a title on its
search card.

## 3. Bring your own transcripts

Skip stages 1–2 entirely. Provide:

**`data/videos.csv`** — one row per video:

```csv
video_id,title,published_at,url,duration,description
abc123,"My Video",2024-03-01T12:00:00Z,https://youtube.com/watch?v=abc123,1368.5,""
```

`video_id`, `title`, `url` are required; the rest are optional.

**`data/transcripts/<video_id>.json`** — one file per video:

```json
{
  "language": "en",
  "duration": 635.9,
  "segments": [
    {"start": 0.0,  "end": 7.0,  "text": "First sentence of the video."},
    {"start": 7.0,  "end": 14.5, "text": "Second sentence..."}
  ]
}
```

Only `segments[].start/end/text` is mandatory (seconds, start < end, non-empty
text) — that's what makes jump-to-the-moment links possible. Segment
granularity: sentence-level or a few seconds each works best.

Already have SRT/VTT subtitles instead? `python pipeline/convert.py path/to/subs/`
converts `<video_id>.srt|.vtt` files into the shape above.

Then check the dataset before ingesting:

```bash
python pipeline/validate.py     # exit 0 = safe to ingest; prints every problem
```

## 4. Ingest — transcripts ➜ GoodMem

```bash
python pipeline/ingest.py --dry-run   # chunk counts, no writes
python pipeline/ingest.py
```

Chunks each transcript into ~60-second overlapping windows, stores one memory
per chunk (text + `{video_id, title, url, start, end}` metadata) plus one
memory per video title. Idempotent — deterministic IDs mean re-runs can't
duplicate. **Deliberately throttled by default**: bulk-feeding a GoodMem
instance at full speed can starve its embedding worker (we reproduced this on
two independent instances); 50-per-batch with a 2s pause completes reliably.

GoodMem embeds asynchronously — give it a few minutes after ingest, then:

```bash
curl localhost:8080/api/health?deep=1     # "title_space_ready": true = all set
```

## 5. Run the website

**Locally:**

```bash
set -a; . ./.env; set +a
cd webapp && uvicorn server:app --host 0.0.0.0 --port 8080
# open http://localhost:8080
```

**Docker (recommended for deployment):**

```bash
docker compose -f deploy/compose.yaml --env-file .env up -d --build
# open http://localhost:8080
```

**Fly.io (public URL in ~5 minutes):**

```bash
cp deploy/fly.toml.example fly.toml       # set your app name inside
fly launch --no-deploy
cat .env | fly secrets import
fly deploy
```

Any container host (Render, Railway, a VM with Docker) works the same way:
build `deploy/Dockerfile`, supply `.env` as environment, expose port 8080.
`GET /healthz` is the liveness probe.

Site name, tagline, example query, and the `Alpha · v0.1` badge are all `.env`
values (`SITE_NAME`, `SITE_TAGLINE`, `EXAMPLE_QUERY`, `APP_STAGE`,
`APP_VERSION`) — no code edits to rebrand. Bump `APP_VERSION` whenever you
retune retrieval so tester feedback maps to a build.

## How search works (short version)

Two GoodMem queries run in parallel — transcript chunks and video titles —
each reranked (Voyage rerank-2.5) into calibrated 0–1 relevance. Scores blend
as `0.6 × title + 0.4 × body`, a local exact-keyword index injects any chunk
containing **all** query terms verbatim (a direct hit outranks whatever the
models think), a relevance floor drops everything below 0.68 so off-topic
queries return an honest "no results" instead of confident-looking noise, and
same-video adjacent windows merge into one card per video. The AI summary is
generated from exactly the shown results and stays collapsed until clicked.

All the knobs (`TITLE_WEIGHT`, `RELEVANCE_FLOOR`, `RETRIEVE_POOL`,
`CHUNK_WINDOW`, …) live in `.env` with tested defaults.
