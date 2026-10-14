def retrieval_scores(ranked_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    retrieved = list(dict.fromkeys(ranked_ids))[:k]
    hits = set(retrieved) & relevant_ids
    return {
        "k": k,
        "precision_at_k": len(hits) / k,
        "retrieved_precision": len(hits) / len(retrieved) if retrieved else 0.0,
        "recall_at_k": len(hits) / len(relevant_ids) if relevant_ids else None,
        "relevant_retrieved": len(hits),
        "relevant_total": len(relevant_ids),
        "retrieved_total": len(retrieved),
    }
