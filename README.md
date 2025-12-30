# 📌 Retrieval-Augmented Generation (RAG)

<p align="center">
  <img src="image.png" width="100%">
</p>

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant external knowledge and injecting it into the prompt before generation.  
This approach improves factual accuracy, grounding, and significantly reduces hallucinations.

---

## 🔹 Architecture
---

## 🔹 Ingestion

Raw data is processed, transformed, and stored to enable fast and accurate semantic search.

### Ingestion Pipeline
* Load documents (PDF / Text / CSV)
* Clean and preprocess text
* Chunk documents into semantically meaningful segments
* Generate dense vector embeddings
* Store embeddings in FAISS for efficient similarity search

---

## 🔹 Retrieval

At query time, the system retrieves the most relevant knowledge chunks.

### Retrieval Flow
* Convert user query into an embedding
* Perform similarity search using FAISS
* Retrieve Top-K most relevant document chunks
* Inject retrieved context into the LLM prompt

---

## 🔹 Tech Stack

### Embeddings
- Sentence-Transformers
  - BGE
  - E5
  - SBERT

### Vector Store
- FAISS (Facebook AI Similarity Search)

---

## 🔹 Core Files

- `data_ingestion.py`  
  Handles document loading, cleaning, chunking, and embedding generation

- `vector_store_faiss.py`  
  Manages FAISS indexing and similarity search

- `data/`  
  Contains raw input documents

- `requirements.txt`  
  Project dependencies

---
