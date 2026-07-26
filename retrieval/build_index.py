# retrieval/build_index.py
"""Build retrieval index over corpus/chunks.jsonl.
Caches embeddings + chunk metadata to retrieval/index/. Run once."""
import json, pathlib, numpy as np
from sentence_transformers import SentenceTransformer

ROOT   = pathlib.Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "corpus" / "chunks.jsonl"
OUT    = ROOT / "retrieval" / "index"
MODEL  = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, local

def main():
    chunks = [json.loads(l) for l in CHUNKS.open() if l.strip()]
    print(f"loaded {len(chunks)} chunks")
    model = SentenceTransformer(MODEL)
    emb = model.encode([c["text"] for c in chunks],
                       normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)        # (n, 384), L2-normalized
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "embeddings.npy", emb)
    (OUT / "chunks_meta.json").write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"saved embeddings {emb.shape} + {len(chunks)} meta -> {OUT}")

if __name__ == "__main__":
    main()
