from pypdf import PdfReader
from pathlib import Path
import json

# Locate the PDF
pdf_path = Path("task2-rag/documents/nist_ai_rmf.pdf")

# Load PDF
reader = PdfReader(pdf_path)

pages = []

# Extract text page by page
for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text() or ""

    # Clean unnecessary whitespace
    text = " ".join(text.split())

    if text:
        pages.append({
            "page": page_number,
            "text": text
        })

# Save extracted text
output_path = Path(
    "task2-rag/documents/extracted_pages.json"
)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(pages, file, indent=4, ensure_ascii=False)

# Display extraction information
total_characters = sum(
    len(page["text"]) for page in pages
)

print("TEXT EXTRACTION COMPLETED!")
print("Total PDF pages:", len(reader.pages))
print("Pages containing text:", len(pages))
print("Total characters:", total_characters)
print("\nFirst page preview:")
print(pages[0]["text"][:500])

print("\nSaved to:", output_path)