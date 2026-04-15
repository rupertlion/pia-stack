import os
import hashlib
import sys
from pathlib import Path
try:
    import httpx
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "qdrant-client", "tqdm"])
    import httpx
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
    from tqdm import tqdm

# Configuration
QDRANT_URL = "http://localhost:6333"
TEI_URL = "http://localhost:8081"
COLLECTION = "codebase"
EMBED_DIM = 768
BATCH_SIZE = 8

# What files do we want to remember?
TARGET_EXTENSIONS = {".py", ".md", ".yml", ".yaml", ".json", ".sh", ".bat"}
IGNORE_DIRS = {".git", ".venv", "__pycache__", "node_modules", "models"}

# Pass the directory you want to index as an argument, default to ~/pia
TARGET_DIR = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "pia")

def setup_qdrant(client):
    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' exists ({info.points_count} vectors)")
    else:
        print(f"Creating collection '{COLLECTION}'...")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        client.create_payload_index(COLLECTION, "filepath", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, "extension", field_schema=PayloadSchemaType.KEYWORD)

def get_code_files(directory):
    filepaths = []
    for root, dirs, files in os.walk(directory):
        # Remove ignored directories so we don't traverse them
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            if Path(file).suffix in TARGET_EXTENSIONS:
                filepaths.append(os.path.join(root, file))
    return filepaths

def chunk_code(filepath, max_tokens=400):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return []

    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    # Very basic chunking by line to preserve code structure
    for line in lines:
        words = len(line.split())
        if current_length + words > max_tokens and current_chunk:
            chunk_text = "\n".join(current_chunk)
            enriched = f"File: {filepath}\n\n{chunk_text}"
            chunk_id = hashlib.md5(f"{filepath}:{chunk_index}".encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": enriched,
                "metadata": {
                    "filepath": filepath,
                    "filename": Path(filepath).name,
                    "extension": Path(filepath).suffix,
                    "chunk_index": chunk_index
                }
            })
            current_chunk = [line]
            current_length = words
            chunk_index += 1
        else:
            current_chunk.append(line)
            current_length += words

    # Grab the last chunk
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        enriched = f"File: {filepath}\n\n{chunk_text}"
        chunk_id = hashlib.md5(f"{filepath}:{chunk_index}".encode()).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": enriched,
            "metadata": {
                "filepath": filepath,
                "filename": Path(filepath).name,
                "extension": Path(filepath).suffix,
                "chunk_index": chunk_index
            }
        })
    return chunks

def embed_batch(texts):
    resp = httpx.post(f"{TEI_URL}/embed", json={"inputs": texts, "truncate": True}, timeout=600.0)
    resp.raise_for_status()
    return resp.json()

def ingest_chunks(client, chunks):
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"Embedding -> {COLLECTION}"):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        try:
            embeddings = embed_batch(texts)
        except Exception as e:
            print(f"  Embedding failed: {e}")
            continue
        points = [
            PointStruct(id=c["id"], vector=emb, payload={**c["metadata"], "content": c["text"]})
            for c, emb in zip(batch, embeddings)
        ]
        client.upsert(COLLECTION, points)

def main():
    print("=" * 60)
    print(f"  Codebase -> Qdrant ({TARGET_DIR})")
    print("=" * 60)
    
    qclient = QdrantClient(url=QDRANT_URL)
    setup_qdrant(qclient)

    files = get_code_files(TARGET_DIR)
    print(f"Found {len(files)} target files.")

    all_chunks = []
    for f in files:
        all_chunks.extend(chunk_code(f))
    
    print(f"Generated {len(all_chunks)} chunks.")
    
    if all_chunks:
        ingest_chunks(qclient, all_chunks)
    
    count = qclient.count(COLLECTION).count
    print(f"\nDone. Total vectors in '{COLLECTION}': {count}")

if __name__ == "__main__":
    main()
