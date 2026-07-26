# retrieval/diagnose.py — show dense-only vs bm25-only vs fused, for one query
import numpy as np
from retrieval.hybrid_retriever import HybridRetriever, _tok

r = HybridRetriever()
q = "high debt-to-income ratio"

qv = r.model.encode([q], normalize_embeddings=True)[0]
dense = r.emb @ qv
bm25  = np.array(r.bm25.get_scores(_tok(q)))

def top(scores, n=5):
    for i in np.argsort(scores)[::-1][:n]:
        c = r.chunks[i]
        print(f"   {scores[i]:.3f}  {c['citation']} ({c['chunk_id']})  {c['text'][:70].strip()!r}")

print("DENSE-only top5:");  top(dense)
print("\nBM25-only top5:");  top(bm25)
