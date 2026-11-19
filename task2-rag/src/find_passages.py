
import json
from pathlib import Path

path = Path("task2-rag/documents/extracted_pages.json")

with open(path, "r", encoding="utf-8") as file:
    pages = json.load(file)

keywords = [
    "trustworthy AI",
    "valid and reliable",
    "safe",
    "secure and resilient",
    "accountable and transparent",
    "explainable and interpretable",
    "privacy enhanced",
    "fairness",
    "GOVERN",
    "MAP",
    "MEASURE",
    "MANAGE",
    "risk management",
    "impact assessment",
    "human oversight"
]

for keyword in keywords:
    print(f"\n{'=' * 65}")
    print("KEYWORD:", keyword)

    matches = 0

    for item in pages:
        text = item["text"]
        position = text.lower().find(keyword.lower())

        if position != -1 and item["page"] >= 10:
            print("\nPage:", item["page"])
            print(text[max(0, position - 150):position + 350])
            matches += 1

            if matches == 3:
                break
