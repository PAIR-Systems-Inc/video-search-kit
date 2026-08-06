# video-search-kit — FastAPI backend + static site.
# Build from the REPO ROOT:  docker build -f deploy/Dockerfile -t video-search .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY webapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp ./webapp
COPY pipeline/chunker.py ./pipeline/chunker.py

# Corpus for the local exact-keyword index. If your transcripts are large and
# you'd rather have a slim image, set ENABLE_KEYWORD_INDEX=0 and drop this COPY
# (search still works; you lose the exact-match override).
COPY data/videos.csv ./data/videos.csv
COPY data/transcripts ./data/transcripts

ENV TRANSCRIPTS_DIR=/app/data/transcripts \
    VIDEOS_CSV=/app/data/videos.csv

WORKDIR /app/webapp
EXPOSE 8080

# Single worker on purpose: each worker would build its own in-memory keyword
# index; concurrency is I/O-bound (GoodMem + LLM), which async uvicorn handles.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
