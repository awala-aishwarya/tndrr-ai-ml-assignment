import json
from pathlib import Path

# Load extracted pages
input_path = Path("task2-rag/documents/extracted_pages.json")

with open(input_path, "r", encoding="utf-8") as file:
    pages = json.load(file)


def create_chunks(pages, chunk_size, overlap):
    chunks = []

    for page in pages:
        words = page["text"].split()

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunks.append({
                "chunk_id": len(chunks),
                "page": page["page"],
                "text": " ".join(chunk_words),
                "word_count": len(chunk_words)
            })

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# Create two chunk sizes
small_chunks = create_chunks(
    pages,
    chunk_size=200,
    overlap=40
)

large_chunks = create_chunks(
    pages,
    chunk_size=400,
    overlap=40
)

# Save both versions
output_dir = Path("task2-rag/documents")

with open(output_dir / "chunks_200.json", "w", encoding="utf-8") as file:
    json.dump(small_chunks, file, indent=4, ensure_ascii=False)

with open(output_dir / "chunks_400.json", "w", encoding="utf-8") as file:
    json.dump(large_chunks, file, indent=4, ensure_ascii=False)


print("CHUNKING COMPLETED!")
print("200-word chunks:", len(small_chunks))
print("400-word chunks:", len(large_chunks))

print("\nFirst small chunk:")
print(small_chunks[0]["text"][:300])

print("\nFirst large chunk:")
print(large_chunks[0]["text"][:300])