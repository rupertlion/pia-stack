#!/usr/bin/env python3
"""
PST Email → Qdrant ETL Pipeline
Extracts Outlook PST archives, chunks emails, embeds via TEI, stores in Qdrant.

RUN FROM: WSL2 terminal with venv activated
  cd ~/pia/etl && source .venv/bin/activate
  python scripts/ingest_pst.py
"""

import subprocess, os, json, hashlib, re
from pathlib import Path
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    PayloadSchemaType, TextIndexParams
)
import httpx
from tqdm import tqdm

# ─── CONFIG — UPDATE THESE PATHS ───
QDRANT_URL = "http://localhost:6333"
TEI_URL = "http://localhost:8081"
COLLECTION = "emails"
EMBED_DIM = 768  # nomic-embed-text-v2
BATCH_SIZE = 32

# Point this to your PST files.
# If they're on your Windows drive, use: /mnt/c/Users/Rupert/path/to/pst/
PST_DIR = Path("/mnt/c/Users/Rupert/Documents/Outlook Files")
EXTRACT_DIR = Path("/tmp/pst_extract")

client = QdrantClient(url=QDRANT_URL)


def setup_collection():
    """Create Qdrant collection if it doesn't already exist."""
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' already exists "
              f"({info.points_count} vectors, dim={info.config.params.vectors.size}).")

        # Check if vector dimension matches our embedding model
        if info.config.params.vectors.size != EMBED_DIM:
            print(f"  ⚠  Dimension mismatch! Collection has {info.config.params.vectors.size}, "
                  f"we need {EMBED_DIM}.")
            print(f"  Options: delete collection with client.delete_collection('{COLLECTION}') "
                  f"or use a different collection name.")
            exit(1)
        return

    print(f"Creating collection '{COLLECTION}'...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=EMBED_DIM, distance=Distance.COSINE
        ),
    )
    client.create_payload_index(
        COLLECTION, "sender",
        field_schema=PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        COLLECTION, "date",
        field_schema=PayloadSchemaType.DATETIME
    )
    client.create_payload_index(
        COLLECTION, "subject",
        field_schema=TextIndexParams(
            type="text", tokenizer="word",
            min_token_len=2, max_token_len=20
        )
    )
    print(f"Collection '{COLLECTION}' created.")


def extract_pst(pst_path: Path) -> Path:
    """Extract PST to individual email files using readpst."""
    out_dir = EXTRACT_DIR / pst_path.stem
    if out_dir.exists() and any(out_dir.rglob("*")):
        print(f"  → Already extracted to {out_dir}, skipping extraction.")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["readpst", "-e", "-o", str(out_dir), str(pst_path)],
        check=True, capture_output=True
    )
    return out_dir


def parse_email_file(filepath: Path) -> dict | None:
    """Parse a single extracted email file into structured data."""
    try:
        text = filepath.read_text(errors="ignore")
        headers = {}
        body_start = text.find("\n\n")
        if body_start == -1:
            return None
        header_text = text[:body_start]
        body = text[body_start + 2:].strip()

        for line in header_text.split("\n"):
            if ": " in line:
                key, val = line.split(": ", 1)
                headers[key.strip().lower()] = val.strip()

        if not body or len(body) < 20:
            return None

        return {
            "subject": headers.get("subject", "No Subject"),
            "sender": headers.get("from", "unknown"),
            "to": headers.get("to", ""),
            "date": headers.get("date", ""),
            "body": body,
            "source_file": str(filepath),
        }
    except Exception:
        return None


def chunk_email(email: dict, max_tokens: int = 400) -> list[dict]:
    """Chunk email body into overlapping segments."""
    body = email["body"]
    words = body.split()
    chunks = []
    step = max_tokens - 50

    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_tokens]
        chunk_text = " ".join(chunk_words)
        enriched = f"Subject: {email['subject']}\n\n{chunk_text}"
        chunk_id = hashlib.md5(
            f"{email['source_file']}:{i}".encode()
        ).hexdigest()

        chunks.append({
            "id": chunk_id,
            "text": enriched,
            "metadata": {
                "subject": email["subject"],
                "sender": email["sender"],
                "to": email["to"],
                "date": email["date"],
                "source_file": email["source_file"],
                "chunk_index": i // step,
            }
        })
    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings from TEI server."""
    resp = httpx.post(
        f"{TEI_URL}/embed",
        json={"inputs": texts, "truncate": True},
        timeout=60.0
    )
    resp.raise_for_status()
    return resp.json()


def ingest_chunks(chunks: list[dict]):
    """Embed and upsert chunks into Qdrant in batches."""
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_batch(texts)

        points = [
            PointStruct(
                id=c["id"],
                vector=emb,
                payload={**c["metadata"], "content": c["text"]}
            )
            for c, emb in zip(batch, embeddings)
        ]
        client.upsert(COLLECTION, points)


def main():
    setup_collection()

    if not PST_DIR.exists():
        print(f"✗  PST directory not found: {PST_DIR}")
        print(f"   Update PST_DIR in this script to your actual PST location.")
        print(f"   Windows paths use: /mnt/c/Users/YourName/...")
        exit(1)

    pst_files = list(PST_DIR.glob("*.pst"))
    print(f"Found {len(pst_files)} PST files.")

    for pst_file in pst_files:
        print(f"\nProcessing: {pst_file.name}")
        extract_dir = extract_pst(pst_file)

        all_chunks = []
        email_files = list(extract_dir.rglob("*.eml")) + \
                      list(extract_dir.rglob("*.txt"))

        for ef in tqdm(email_files, desc="Parsing"):
            email = parse_email_file(ef)
            if email:
                all_chunks.extend(chunk_email(email))

        print(f"  → {len(all_chunks)} chunks from {len(email_files)} emails")
        if all_chunks:
            ingest_chunks(all_chunks)

    count = client.count(COLLECTION).count
    print(f"\nDone. Total vectors in '{COLLECTION}': {count}")


if __name__ == "__main__":
    main()