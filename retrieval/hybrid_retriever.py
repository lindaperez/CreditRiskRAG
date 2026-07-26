# retrieval/hybrid_retriever.py
"""Hybrid BM25 + dense retriever over the cached index.
Fuses the two rankings with Reciprocal Rank Fusion (RRF)."""
import json, pathlib, re, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT  = pathlib.Path(__file__).resolve().parent.parent
INDEX = ROOT / "retrieval" / "index"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _tok(s):  # simple lexical tokenizer: lowercase words + section numbers
    return re.findall(r"[a-z0-9§]+", s.lower())

class HybridRetriever:
    def __init__(self, rrf_k=60):
        self.rrf_k  = rrf_k
        self.chunks = json.loads((INDEX / "chunks_meta.json").read_text())
        self.emb    = np.load(INDEX / "embeddings.npy")          # (n,384) normalized
        self.bm25   = BM25Okapi([_tok(c["text"]) for c in self.chunks])
        self.model  = SentenceTransformer(MODEL)

    def _ranks(self, scores):
        # map doc_index -> rank (0 = best); higher score = better
        order = np.argsort(scores)[::-1]
        return {int(idx): r for r, idx in enumerate(order)}

    def search(self, query, k=5):
        # dense: cosine == dot product (vectors are normalized)
        q = self.model.encode([query], normalize_embeddings=True)[0]
        dense_scores = self.emb @ q
        # lexical
        bm25_scores = self.bm25.get_scores(_tok(query))
        # fuse by rank: RRF score = sum 1/(rrf_k + rank) across both
        dr, br = self._ranks(dense_scores), self._ranks(bm25_scores)
        fused = {i: 1/(self.rrf_k+dr[i]) + 1/(self.rrf_k+br[i])
                 for i in range(len(self.chunks))}
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        out = []
        for i in top:
            c = self.chunks[i]
            out.append({"chunk_id": c["chunk_id"], "citation": c["citation"],
                        "source": c["source"], "text": c["text"],
                        "score": round(fused[i], 5)})
        return out

if __name__ == "__main__":
    r = HybridRetriever()
    for hit in r.search("high debt-to-income ratio", k=3):
        print(f"[{hit['score']}] {hit['citation']} ({hit['chunk_id']})")
        print("   ", hit["text"][:120].replace("\n", " "), "...\n")
