import json
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding

documents_dir = Path("task2-rag/documents")
results_dir = Path("task2-rag/results")

# Load embedding model
model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def cosine_similarity(query_embedding, embeddings):

    query_norm = np.linalg.norm(query_embedding)
    document_norms = np.linalg.norm(
        embeddings, axis=1
    )

    similarities = np.dot(
        embeddings, query_embedding
    ) / (document_norms * query_norm + 1e-10)

    return similarities


def search_documents(query, chunk_size=200, top_k=5):

    # Load chunks
    with open(
        documents_dir / f"chunks_{chunk_size}.json",
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)
        print("Chunk size:", chunk_size)
        print("Total chunks loaded:", len(chunks))

    # Load embeddings
    embeddings = np.load(
        results_dir / f"embeddings_{chunk_size}.npy"
    )

    # Generate query embedding
    query_embedding = np.array(
        list(model.query_embed(query))[0],
        dtype=np.float32
    )

    # Calculate similarities
    scores = cosine_similarity(
        query_embedding, embeddings
    )

    # Get top matching chunks
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return results


# Test semantic search
if __name__ == "__main__":

    question = "Human oversight processes are defined, assessed, and documented, including the roles and responsibilities of human-AI teams."

    results = search_documents(
        question,
        chunk_size=200,
        top_k=5
    )

    print("\nQUESTION:", question)

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print("Page:", result["page"])
        print("Chunk ID:", result["chunk_id"])
        print("Similarity:", round(result["score"], 4))
        print("Text:", result["text"][:400])