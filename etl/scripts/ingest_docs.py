#!/usr/bin/env python3
"""
Document (PDF/DOCX/TXT) → Qdrant ETL Pipeline

RUN FROM: WSL2 terminal with venv activated
  cd ~/pia/etl && source .venv/bin/activate
  python scripts/ingest_docs.py
"""

from pathlib import Path
from unstructured.partition.auto import partition
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import httpx, hashlib
from tqdm import tqdm

# ─── CONFIG — UPDATE THESE PATHS ───
QDRANT_URL = "http://localhost:6333"
TEI_URL = "http://localhost:8081"
COLLECTION = "documents"
EMBED_DIM = 768
BATCH_SIZE = 32

# Point to your documents folder
DOC_DIR = Path("/mnt/c/Users/Ruper/Documents")

client = QdrantClient(url=QDRANT_URL)


def setup_collection():
    """Create or verify the documents collection."""
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' exists ({info.points_count} vectors).")
        if info.config.params.vectors.size != EMBED_DIM:
            print(f"  ⚠  Dimension mismatch! Has {info.config.params.vectors.size}, need {EMBED_DIM}.")
            exit(1)
        return

    print(f"Creating collection '{COLLECTION}'...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def process_file(filepath: Path) -> list[dict]:
    """Extract text from PDF/DOCX/TXT using unstructured."""
    try:
        elements = partition(str(filepath))
        full_text = "\n".join([str(el) for el in elements])
    except Exception as e:
        print(f"  ⚠ Skipping {filepath.name}: {e}")
        return []

    words = full_text.split()
    chunks = []
    for i in range(0, len(words), 350):
        chunk = " ".join(words[i:i+400])
        chunks.append({
            "id": hashlib.md5(f"{filepath}:{i}".encode()).hexdigest(),
            "text": chunk,
            "metadata": {
                "filename": filepath.name,
                "filepath": str(filepath),
                "doc_type": filepath.suffix,
            }
        })
    return chunks


def embed_batch(texts):
    resp = httpx.post(
        f"{TEI_URL}/embed",
        json={"inputs": texts, "truncate": True},
        timeout=60.0
    )
    resp.raise_for_status()
    return resp.json()


def main():
    setup_collection()

    if not DOC_DIR.exists():
        print(f"✗  Document directory not found: {DOC_DIR}")
        print(f"   Update DOC_DIR in this script. Windows paths: /mnt/c/Users/...")
        exit(1)

    files = list(DOC_DIR.rglob("*.pdf")) + \
            list(DOC_DIR.rglob("*.docx")) + \
            list(DOC_DIR.rglob("*.txt"))
    print(f"Found {len(files)} documents.")

    all_chunks = []
    for f in tqdm(files, desc="Processing"):
        all_chunks.extend(process_file(f))

    print(f"Total chunks: {len(all_chunks)}")

    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding"):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_batch(texts)
        points = [
            PointStruct(id=c["id"], vector=emb,
                        payload={**c["metadata"], "content": c["text"]})
            for c, emb in zip(batch, embeddings)
        ]
        client.upsert(COLLECTION, points)

    count = client.count(COLLECTION).count
    print(f"\nDone. Total vectors in '{COLLECTION}': {count}")


if __name__ == "__main__":
    main()