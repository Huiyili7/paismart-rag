"""Seed the running PaiSmart knowledge base with the evaluation corpus.

For each document in ``eval/corpus/`` it performs the production upload flow:
  1. POST /api/v1/upload/chunk   (single chunk; these docs are small)
  2. POST /api/v1/upload/merge   -> enqueues a Kafka task that parses, chunks,
                                    embeds and indexes the document into ES.

Documents are uploaded as PUBLIC so every evaluation query can retrieve them.

Usage:
    export PAISMART_USER=admin PAISMART_PASS=admin123
    python scripts/seed_corpus.py --base-url http://localhost:8080 --corpus corpus
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import requests


def login(base_url: str, username: str, password: str) -> str:
    resp = requests.post(f"{base_url}/api/v1/users/login",
                         json={"username": username, "password": password}, timeout=30)
    resp.raise_for_status()
    token = (resp.json().get("data") or {}).get("token")
    if not token:
        raise SystemExit(f"login failed: {resp.text}")
    return token


def upload_doc(base_url: str, token: str, path: str, org_tag: str | None) -> None:
    name = os.path.basename(path)
    with open(path, "rb") as f:
        content = f.read()
    file_md5 = hashlib.md5(content).hexdigest()
    headers = {"Authorization": f"Bearer {token}"}

    # multipart form fields must be strings (requests rejects raw ints here)
    data = {
        "fileMd5": file_md5,
        "chunkIndex": "0",
        "totalSize": str(len(content)),
        "fileName": name,
        "totalChunks": "1",
        "isPublic": "true",
    }
    if org_tag:
        data["orgTag"] = org_tag

    chunk = requests.post(
        f"{base_url}/api/v1/upload/chunk",
        headers=headers,
        data=data,
        files={"file": (name, content, "text/markdown")},
        timeout=60,
    )
    chunk.raise_for_status()

    merge = requests.post(
        f"{base_url}/api/v1/upload/merge",
        headers=headers,
        json={"fileMd5": file_md5, "fileName": name},
        timeout=60,
    )
    merge.raise_for_status()
    print(f"  uploaded + merged: {name}  (md5={file_md5[:8]}…)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8081")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--org-tag", default=None, help="optional org tag; defaults to user's primary org")
    p.add_argument("--user", default=os.environ.get("PAISMART_USER", "admin"))
    p.add_argument("--password", default=os.environ.get("PAISMART_PASS", "admin123"))
    args = p.parse_args()

    docs = sorted(f for f in os.listdir(args.corpus) if f.endswith((".md", ".txt")))
    if not docs:
        raise SystemExit(f"no documents found in {args.corpus}")

    print(f"Logging in as {args.user} …")
    token = login(args.base_url, args.user, args.password)

    print(f"Seeding {len(docs)} documents …")
    for name in docs:
        upload_doc(args.base_url, token, os.path.join(args.corpus, name), args.org_tag)
        time.sleep(0.3)

    print("\nDone. Indexing is asynchronous (Kafka -> parse -> embed -> ES).")
    print("Wait ~10-30s before running the live evaluation so all chunks are searchable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
