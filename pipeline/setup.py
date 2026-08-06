#!/usr/bin/env python3
"""
setup.py — one-time provisioning of your GoodMem instance.

Creates (or reuses, if run twice):
  1. The embedder      (EMBEDDER_MODEL, registered with EMBEDDER_API_KEY)
  2. The reranker      (RERANKER_MODEL, if RERANKER_API_KEY is set)
  3. Two spaces:       "<prefix>-transcripts" and "<prefix>-titles",
                       chunking disabled — every memory we ingest is already
                       one timestamped chunk and must not be re-split.

Then prints the .env lines to copy in (GOODMEM_SPACE_ID etc.).

Usage:
  set -a; . ./.env; set +a
  python pipeline/setup.py --prefix mychannel
"""

import argparse
import os
import sys

from goodmem import Goodmem
from goodmem.errors import ConflictError


def require(name):
    v = os.environ.get(name, "").strip()
    if not v or v.endswith("..."):
        sys.exit(f"{name} is not set — fill it in .env first (see .env.example).")
    return v


def get_or_create_space(client, name, embedder_id):
    try:
        space = client.spaces.create(
            name=name,
            space_embedders=[{"embedderId": embedder_id}],
            default_chunking_config={"none": {}},
            labels={"kit": "video-search-kit"},
        )
        print(f"created space '{name}'")
    except ConflictError:
        space = next(iter(client.spaces.list(name_filter=name)), None)
        if space is None:
            sys.exit(f"space '{name}' exists but is not visible to this API key")
        print(f"space '{name}' already exists — reusing")
    return str(space.space_id)


def main():
    parser = argparse.ArgumentParser(description="Provision GoodMem for the kit.")
    parser.add_argument("--prefix", default="videosearch",
                        help="Space name prefix (e.g. your channel slug)")
    args = parser.parse_args()

    base_url = require("GOODMEM_BASE_URL")
    api_key = require("GOODMEM_API_KEY")
    embedder_model = require("EMBEDDER_MODEL")
    embedder_key = require("EMBEDDER_API_KEY")
    reranker_model = os.environ.get("RERANKER_MODEL", "").strip()
    reranker_key = os.environ.get("RERANKER_API_KEY", "").strip()

    with Goodmem(base_url=base_url, api_key=api_key, timeout=60.0) as client:
        # --- embedder (reuse an existing registration of the same model) ----
        embedder_id = None
        for e in client.embedders.list():
            if getattr(e, "model_identifier", None) == embedder_model:
                embedder_id = str(e.embedder_id)
                print(f"reusing embedder {embedder_id} ({embedder_model})")
                break
        if embedder_id is None:
            e = client.embedders.create(display_name=embedder_model,
                                        model_identifier=embedder_model,
                                        api_key=embedder_key)
            embedder_id = str(e.embedder_id)
            print(f"created embedder {embedder_id} ({embedder_model})")

        # --- reranker (optional) -------------------------------------------
        reranker_id = ""
        if reranker_key and reranker_model:
            for r in client.rerankers.list():
                if getattr(r, "model_identifier", None) == reranker_model:
                    reranker_id = str(r.reranker_id)
                    print(f"reusing reranker {reranker_id} ({reranker_model})")
                    break
            if not reranker_id:
                r = client.rerankers.create(display_name=reranker_model,
                                            model_identifier=reranker_model,
                                            api_key=reranker_key)
                reranker_id = str(getattr(r, "reranker_id", None) or r.reranker.reranker_id)
                print(f"created reranker {reranker_id} ({reranker_model})")
        else:
            print("no RERANKER_API_KEY — skipping reranker (search will use raw vector scores)")

        # --- spaces --------------------------------------------------------
        space_id = get_or_create_space(client, f"{args.prefix}-transcripts", embedder_id)
        title_space_id = get_or_create_space(client, f"{args.prefix}-titles", embedder_id)

    print("\nAdd these lines to your .env:\n")
    print(f"GOODMEM_SPACE_ID={space_id}")
    print(f"GOODMEM_TITLE_SPACE_ID={title_space_id}")
    print(f"GOODMEM_RERANKER_ID={reranker_id}")


if __name__ == "__main__":
    main()
