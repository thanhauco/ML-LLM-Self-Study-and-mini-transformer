import numpy as np

def cosine_similarity(v1, v2):
    """
    Computes the cosine similarity between two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return dot_product / (norm_v1 * norm_v2)

def evaluate_retrieval(query_embedding, context_embeddings):
    """
    Simulates finding the most relevant context and returning its score.
    """
    similarities = [cosine_similarity(query_embedding, ctx) for ctx in context_embeddings]
    best_score = max(similarities) if similarities else 0.0
    return best_score, similarities

if __name__ == "__main__":
    # Mock embeddings
    query = np.array([0.1, 0.2, 0.8])
    contexts = [
        np.array([0.1, 0.2, 0.7]), # Relevant
        np.array([0.9, 0.1, 0.1]), # Irrelevant
        np.array([0.2, 0.3, 0.9])  # Very Relevant
    ]
    
    score, all_scores = evaluate_retrieval(query, contexts)
    print(f"Best retrieval score: {score:.4f}")
    print(f"All context scores: {[round(s, 4) for s in all_scores]}")
