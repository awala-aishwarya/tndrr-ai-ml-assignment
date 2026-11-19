from pypdf import PdfReader
from pathlib import Path

pdf_path = Path("task2-rag/documents/nist_ai_rmf.pdf")

reader = PdfReader(pdf_path)

print("PDF loaded successfully!")
print("Total pages:", len(reader.pages))

first_page_text = reader.pages[0].extract_text()

print("\nFirst page text:")
print(first_page_text[:1000])