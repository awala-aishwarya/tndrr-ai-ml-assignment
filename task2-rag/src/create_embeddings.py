import json
import numpy as np
from pathlib import Path
from fastembed import TextEmbedding

# Project paths
documents_dir = Path("task2-rag/documents")
results_dir = Path("task2-rag/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Load embedding model
print("Loading embedding model...")

model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

# Generate embeddings for both chunk sizes
for chunk_size in [200, 400]:

    input_path = documents_dir / f"chunks_{chunk_size}.json"

    with open(input_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    texts = [chunk["text"] for chunk in chunks]

    print(f"\nGenerating embeddings for {chunk_size}-word chunks...")

    embeddings = list(model.embed(texts))

    embeddings = np.array(embeddings, dtype=np.float32)

    # Save numerical vectors
    output_path = results_dir / f"embeddings_{chunk_size}.npy"
    np.save(output_path, embeddings)

    print("Number of chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("Saved to:", output_path)

print("\nALL EMBEDDINGS GENERATED SUCCESSFULLY!")