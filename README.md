# Nexus: Enterprise AI Knowledge Base

Nexus is a modular **Retrieval-Augmented Generation (RAG)** system built on a **FastAPI** backbone. It utilizes **Supabase** for relational and vector data persistence (`pgvector`) and **LangChain** for LLM orchestration, enabling enterprises to turn static documents into interactive, verifiable intelligence.



## System Architecture

### The Tech Stack
| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Vanilla JS, HTML5, CSS3 | Minimal, responsive SPA with Material 3 / Gemini aesthetics. |
| **Backend** | Python (FastAPI) | Handles RAG logic, AI routing, and batch metadata lookups. |
| **Database** | Supabase (PostgreSQL) | Stores relational data and Vector embeddings. |
| **AI Models** | Gemini 1.5 & AWS Bedrock | Powers embeddings and high-level reasoning. |
| **Data Viz** | Chart.js | Dynamically renders AI-generated JSON into interactive charts. |

---

## Core Engines

### 1. The Ingestion Pipeline (`/upload`)
The ingestion engine follows a strict **Load-Transform-Embed** pattern:
* **Parsing:** Multi-format support via `PyPDF2` (PDF), `pandas` (XLSX/CSV), and `python-pptx` (PPTX).
* **Chunking:** Utilizes `RecursiveCharacterTextSplitter` (2,000 character chunks with 200 character overlap to preserve semantic context).
* **Metadata Baking:** Injects the `project_id`, `source` filename, and the Supabase Storage `file_url` directly into the chunk's metadata before vectorization.
* **Embedding:** Converts chunks into vectors using Google `gemini-embedding-001` or Amazon `titan-embed-text-v2:0`.



### 2. The Retrieval & Synthesis Engine (`/chat`)
* **Vector Search:** Converts the user query into a vector and calls the `match_project_documents` RPC in Supabase to perform a Cosine Similarity Search (`<=>`).
* **Dynamic Routing:** Routes prompts to either Google Gemini or AWS Bedrock (OSS Models) based on user selection.
* **Citation Extraction:** Uses post-processing regex to extract a forced `SOURCES: [File.ext]` tag from the LLM, mapping it to exact public URLs via a batch relational database lookup.

---

## API Reference

### `POST /chat`
Generates a RAG-based response with verifiable citations and optional chart configurations.

**Payload (Form Data):**
* `message`: (string) The user question.
* `project_id`: (int) Target project silo for scoped retrieval.
* `model`: (string) The LLM identifier (e.g., `gemini-2.5-flash`).

**Response:**
```json
{
  "answer": "Calculated revenue for Q3 is $4.2M...",
  "sources": [
    {
      "name": "Q3_Report.pdf",
      "url": "https://[PROJECT].supabase.co/storage/v1/object/public/project_files/project_1/Q3_Report.pdf"
    }
  ]
}