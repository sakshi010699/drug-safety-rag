import numpy as np
from typing import Optional
from schema import Chunk, ChunkStore

DRUG_KEYWORDS = {
    "atorvastatin": ["atorvastatin", "lipitor", "statin", "cholesterol", "ldl"],
    "diclofenac":   ["diclofenac", "voltaren", "arthrotec", "cataflam",
                     "nsaid", "arthritis", "inflammation"],
}

def detect_drug(query: str) -> Optional[str]:
    query = query.lower()
    for drug, keywords in DRUG_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            return drug
    return None

def retrieve(
    query_vector: np.ndarray,
    query_text: str,
    store: ChunkStore,
    embeddings: np.ndarray,
    index,
    k: int = 5
) -> list:
    detected_drug = detect_drug(query_text)

    if detected_drug:
        candidate_chunks = store.filter_by_drug(detected_drug)
        candidate_indices = [c.vector_index for c in candidate_chunks]
        candidate_vectors = embeddings[candidate_indices].astype(np.float32)
        scores = candidate_vectors @ query_vector[0]
        top_k_local = np.argsort(scores)[::-1][:k]
        results = []
        for local_idx in top_k_local:
            global_idx = candidate_indices[local_idx]
            chunk = store.get(global_idx)
            results.append({
                "chunk": chunk,
                "score": float(scores[local_idx]),
                "drug": detected_drug
            })
        return results

    else:
        distances, indices = index.search(query_vector, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = store.get(idx)
            results.append({
                "chunk": chunk,
                "score": 1 / (1 + float(dist)),
                "drug": None
            })
        return results

def show_results(
    query_text: str,
    store: ChunkStore,
    embeddings: np.ndarray,
    index,
    encode_fn,
    k: int = 5
):
    query_vector = encode_fn([query_text], batch_size=32).astype(np.float32)
    results = retrieve(query_vector, query_text, store, embeddings, index, k)

    print(f"Query: '{query_text}'")
    print("=" * 60)
    for i, r in enumerate(results):
        chunk = r["chunk"]
        filter_note = f" [filtered to {r['drug']}]" if r['drug'] else ""
        print(f"\nRank {i+1} | score={r['score']:.4f} | drug={chunk.drug_name}{filter_note}")
        print(f"\n{chunk.text[:150]}")