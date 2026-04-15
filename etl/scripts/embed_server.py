from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    cache_folder="/models"
)
app = FastAPI()


class EmbedRequest(BaseModel):
    inputs: list[str]
    truncate: bool = True


@app.post("/embed")
def embed(req: EmbedRequest):
    results = model.encode(
        req.inputs,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False
    )
    return results.tolist()


@app.get("/health")
def health():
    return {"status": "ok"}


uvicorn.run(app, host="0.0.0.0", port=8081, workers=1, timeout_keep_alive=600)