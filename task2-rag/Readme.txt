# Task 2: Retrieval-Augmented Generation (RAG) from Scratch

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline from scratch using a public document.

The system extracts document text, divides it into chunks, generates embeddings, retrieves relevant passages, and uses a Large Language Model (LLM) to generate context-aware answers.

The implementation uses Python, NumPy, FastEmbed, and Gemini without using LangChain or LlamaIndex.

## 2. Source Document

- **Document:** NIST AI Risk Management Framework (AI RMF 1.0)
- **Publication:** NIST AI 100-1
- **Published:** January 2023
- **Document Length:** 48 pages

## 3. System Architecture

The RAG pipeline follows these steps:

1. PDF text extraction with page-number preservation.
2. Text preprocessing and cleaning.
3. Chunk creation using overlapping word windows.
4. Embedding generation.
5. Similarity-based document retrieval.
6. Context-based answer generation using an LLM.

## 4. Chunking Strategy

Two chunk sizes were evaluated:

| Configuration | Chunk Size | Overlap |
|---|---:|---:|
| Configuration 1 | 200 words | 40 words |
| Configuration 2 | 400 words | 40 words |

The extracted document generated:
- 114 chunks using 200-word chunking.
- 66 chunks using 400-word chunking.

## 5. Embedding and Retrieval

- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **Embedding Library:** FastEmbed
- **Embedding Dimensions:** 384
- **Similarity Method:** Cosine Similarity
- **Retrieval Method:** NumPy-based dense retrieval
- **Retrieved Results:** Top 5 passages

The system retrieves relevant chunks based on similarity between the query embedding and document embeddings.

## 6. LLM Answer Generation

Gemini (`gemini-3.5-flash-lite`) is used to generate answers using the retrieved document context.

The prompt instructs the model to answer based on the retrieved passages and include page references.

The API key is loaded from a local `.env` file and is not included in the repository.

## 7. Retrieval Evaluation

Fifteen manually written questions were used for evaluation.

The evaluation metric is page-level Recall@5: whether the expected source page appears among the top five retrieved results.

| Chunk Configuration | Recall@5 |
|---|---:|
| 200-word chunks | 86.67% |
| 400-word chunks | 73.33% |
| 200-word chunks with targeted query expansion | 100% |

The query expansion experiment improved the two questions that initially failed retrieval.

The 100% result applies only to this 15-question evaluation set and should not be considered a general performance guarantee.

## 8. Retrieval Failures and Improvements

### Failure 1: AI RMF Core Functions

The original query did not retrieve the expected page in the top five results.

**Improvement:** Expanded the query with the terms GOVERN, MAP, MEASURE, and MANAGE.

### Failure 2: Human Oversight

The original query did not retrieve the expected page.

**Improvement:** Rephrased the query to include the wording "human oversight processes are defined, assessed, and documented".

These changes improved retrieval for both questions.

## 9. BM25 and Reranking

**BM25:**

BM25 is a sparse information retrieval algorithm that ranks documents using term frequency, inverse document frequency, and document-length normalization.

It can complement dense retrieval when queries contain specific keywords, technical terms, or acronyms.

**Reranking:**

A reranker is a second-stage model that evaluates query-passage relevance more precisely and reorders retrieved passages. It can improve retrieval quality at the cost of additional latency and computation.

## 10. Technologies Used

- Python
- NumPy
- FastEmbed
- Sentence Transformers
- Gemini API
- PyMuPDF
- JSON

## 11. Project Structure

```text
task2-rag/
├── documents/
│   ├── nist_ai_rmf.pdf
│   ├── extracted_pages.json
│   ├── chunks_200.json
│   └── chunks_400.json
├── evaluation/
│   └── questions.json
├── results/
│   ├── embeddings_200.npy
│   └── embeddings_400.npy
├── src/
│   ├── check_pdf.py
│   ├── extract_text.py
│   ├── chunk_text.py
│   ├── create_embeddings.py
│   ├── retrieval.py
│   ├── evaluate_retrieval.py
│   ├── find_passages.py
│   └── generate_answer.py
└── README.md
```

## 12. Conclusion

The project demonstrates a complete RAG pipeline built without high-level RAG frameworks. The evaluation shows that chunk size and query formulation affect retrieval performance. Query expansion helped resolve two retrieval failures in the test set.