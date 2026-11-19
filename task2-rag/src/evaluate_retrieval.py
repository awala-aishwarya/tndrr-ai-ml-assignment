
import json
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding

documents_dir = Path("task2-rag/documents")
results_dir = Path("task2-rag/results")
evaluation_dir = Path("task2-rag/evaluation")

# Load the 15 original evaluation questions
with open(evaluation_dir / "questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)

# Expanded queries for the two previously missed questions
expanded_queries = {
    13: (
        "After instituting GOVERN, organizations usually start with MAP "
        "and continue to MEASURE or MANAGE. The functions are integrated "
        "iteratively throughout the AI lifecycle."
    ),
    14: (
        "Human oversight processes are defined, assessed, and documented, "
        "including the roles and responsibilities of human-AI teams."
    )
}

print("Loading embedding model...")
model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def cosine_similarity(query_embedding, embeddings):
    query_norm = np.linalg.norm(query_embedding)
    document_norms = np.linalg.norm(embeddings, axis=1)

    return np.dot(embeddings, query_embedding) / (
        document_norms * query_norm + 1e-10
    )


def evaluate_chunk_size(chunk_size, use_expansion=False):
    with open(
        documents_dir / f"chunks_{chunk_size}.json",
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    embeddings = np.load(
        results_dir / f"embeddings_{chunk_size}.npy"
    )

    hits = 0
    total_questions = len(questions)

    mode = "WITH QUERY EXPANSION" if use_expansion else "ORIGINAL QUERIES"

    print("\n" + "=" * 60)
    print(f"EVALUATING {chunk_size}-WORD CHUNKS")
    print(f"Mode: {mode}")
    print(f"Total chunks: {len(chunks)}")
    print("=" * 60)

    for item in questions:
        question_id = item["id"]
        original_query = item["question"]
        expected_page = item["expected_page"]

        # Expand only Questions 13 and 14 when enabled
        if use_expansion and question_id in expanded_queries:
            query = expanded_queries[question_id]
            print(f"\nQuestion {question_id}: {original_query}")
            print("Expanded query:", query)
        else:
            query = original_query
            print(f"\nQuestion {question_id}: {original_query}")

        query_embedding = np.array(
            list(model.query_embed(query))[0],
            dtype=np.float32
        )

        scores = cosine_similarity(query_embedding, embeddings)
        top_indices = np.argsort(scores)[::-1][:5]

        retrieved_pages = [
            chunks[index]["page"] for index in top_indices
        ]

        hit = expected_page in retrieved_pages

        if hit:
            hits += 1

        print("Expected page:", expected_page)
        print("Retrieved pages:", retrieved_pages)
        print("Result:", "HIT" if hit else "MISS")

    recall_at_5 = hits / total_questions

    print(f"\nHits: {hits}/{total_questions}")
    print(f"Recall@5: {recall_at_5:.4f}")
    print(f"Recall@5 percentage: {recall_at_5 * 100:.2f}%")

    return recall_at_5


if __name__ == "__main__":

    # Original baseline evaluations
    recall_200_original = evaluate_chunk_size(
        200,
        use_expansion=False
    )

    recall_400_original = evaluate_chunk_size(
        400,
        use_expansion=False
    )

    # Improved 200-word evaluation
    recall_200_improved = evaluate_chunk_size(
        200,
        use_expansion=True
    )

    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 60)

    print(
        f"200-word original:  "
        f"{recall_200_original * 100:.2f}%"
    )

    print(
        f"400-word original:  "
        f"{recall_400_original * 100:.2f}%"
    )

    print(
        f"200-word improved:  "
        f"{recall_200_improved * 100:.2f}%"
    )

    improvement = (
        recall_200_improved - recall_200_original
    ) * 100

    print(f"Improvement: {improvement:+.2f} percentage points")
