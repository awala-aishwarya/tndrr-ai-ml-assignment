
import os
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from fastembed import TextEmbedding


# --------------------------------------------------
# 1. Project paths and Gemini configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK2_DIR = PROJECT_ROOT / "task2-rag"

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your project-root .env file.")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.5-flash-lite"


# --------------------------------------------------
# 2. Load chunks and embeddings
# --------------------------------------------------

chunks_path = TASK2_DIR / "documents" / "chunks_200.json"
embeddings_path = TASK2_DIR / "results" / "embeddings_200.npy"

with open(chunks_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

embeddings = np.load(embeddings_path)

print(f"Loaded {len(chunks)} text chunks.")
print(f"Embedding matrix shape: {embeddings.shape}")


# --------------------------------------------------
# 3. Load the same embedding model used for retrieval
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Retrieve the most relevant passages
# --------------------------------------------------

def retrieve_context(question, top_k=5):

    query_embedding = np.array(
        list(embedding_model.query_embed(question)),
        dtype=np.float32
    )[0]

    query_embedding = query_embedding / (
        np.linalg.norm(query_embedding) + 1e-10
    )

    normalized_embeddings = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    )

    similarities = normalized_embeddings @ query_embedding

    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_passages = []

    for index in top_indices:
        chunk = chunks[int(index)]

        retrieved_passages.append({
            "page": chunk.get("page"),
            "text": chunk.get("text", ""),
            "similarity": float(similarities[index])
        })

    return retrieved_passages


# --------------------------------------------------
# 5. Generate an answer using Gemini
# --------------------------------------------------

def generate_answer(question):

    passages = retrieve_context(question, top_k=5)

    context = "\n\n".join(
        f"[Source: Page {item['page']}]\n{item['text']}"
        for item in passages
    )

    prompt = f"""
You are a question-answering assistant for the NIST AI Risk
Management Framework (AI RMF 1.0).

Answer the user's question using ONLY the retrieved context below.

Instructions:
1. Do not invent facts or use outside knowledge.
2. If the context does not contain the answer, clearly say so.
3. Cite the relevant source page numbers in your answer.
4. Give a clear and concise answer.

Retrieved context:
{context}

User question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    print("\n" + "=" * 65)
    print("QUESTION:", question)
    print("=" * 65)

    print("\nRETRIEVED SOURCES:")

    for item in passages:
        print(
            f"Page {item['page']} | "
            f"Similarity: {item['similarity']:.4f}"
        )

    print("\nGENERATED ANSWER:")
    print(response.text)

    print("=" * 65)

    return response.text


# --------------------------------------------------
# 6. Interactive question-answering loop
# --------------------------------------------------

if __name__ == "__main__":

    print("\nNIST AI RMF - RAG QUESTION ANSWERING SYSTEM")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("Enter your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("RAG system stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            generate_answer(question)

        except Exception as error:
            print(f"\nError generating answer: {error}")
