#  Laptop Recommendation System with Chatbot using RAG + Streamlit

This project is an end-to-end **Laptop Recommendation System** enhanced by a **chatbot assistant** that uses a **RAG (Retrieval-Augmented Generation)** pipeline powered by **SentenceTransformers**, **BM25**, and **Large Language Models (LLMs)**. Built with a modular design and a clean Streamlit UI, it allows users to search for laptops using natural language and interact via a smart chatbot.

---

##  Features

-  **Hybrid Laptop Search**: Combines keyword matching (BM25) + semantic similarity (embeddings)
-  **Chatbot Assistant**: Conversational UI for refining and narrowing search results
-  **LLM Integration**: Uses pre-trained transformer model to semantically match user queries
-  **RAG Pipeline**: Retrieval + generation-based approach
-  **Streamlit Frontend**: Clean, interactive UI with real-time updates

---

##  RAG Pipeline Overview


- **Retrieval**: BM25 + embeddings (MiniLM) used to fetch top-K relevant documents
- **Augmentation**: Result passed to chatbot layer for refinement (if query continues)
- **Generation**: Response generated from selected relevant results

---

##  Project Structure

```text
├── laptop_data.csv          # Raw laptop dataset (used in preprocessing)
├── cleaned_laptops.pkl      # Cleaned + embedded laptop data (generated)
├── preprocessing.py         # Clean, normalize, and generate sentence embeddings
├── search_engine.py         # Hybrid RAG retrieval logic (BM25 + embeddings)
├── streamlit_app.py         # UI interface and chatbot logic using Streamlit
```

---

## ⚙️ Tech Stack & Models

| Component        | Details |
|------------------|---------|
| **LLM Model**    | `all-MiniLM-L6-v2` (from `sentence-transformers`) |
| **Retrieval**    | BM25 (via `rank_bm25`) + Cosine Similarity |
| **Interface**    | Streamlit |
| **Format**       | RAG-style hybrid search pipeline |
| **Storage**      | `.pkl` (cleaned + embedded data) |

---

## Workflow

### 1. `preprocessing.py`
- Cleans raw data
- Normalizes text columns
- Extracts nested info (e.g., specifications)
- Combines fields into a search corpus
- Embeds using SentenceTransformer
- Saves to `cleaned_laptops.pkl`

### 2. `search_engine.py`
- Loads `cleaned_laptops.pkl`
- Computes BM25 + semantic similarity
- Combines rankings for hybrid score
- Returns top K relevant laptops

### 3. `streamlit_app.py`
- Accepts user query
- Triggers hybrid search
- Optional: further query refinement by chatbot
- Displays results interactively

---

##  How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run preprocessing (if `cleaned_laptops.pkl` not present):
```bash
python preprocessing.py
```

3. Launch the app:
```bash
streamlit run streamlit_app.py
```

---

##  Example Queries

- “I want a lightweight laptop with good battery backup under ₹60,000”
- “Show gaming laptops with SSD and minimum 8GB RAM”
- “Filter those with Ryzen processors only”

---


