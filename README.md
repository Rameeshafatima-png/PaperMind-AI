# PaperMind AI — Research Paper Assistant using RAG

> **GitHub Repository Description:** A document-grounded AI research assistant that uses Retrieval-Augmented Generation, FAISS, and Google Gemini to answer questions from uploaded research papers.

## Overview

**PaperMind AI** is a Retrieval-Augmented Generation (RAG) based research paper assistant designed to help users interact with academic research papers through natural-language questions.

Users can upload one or multiple research paper PDFs. PaperMind AI extracts their text, divides the content into searchable chunks, generates vector embeddings, and stores those embeddings in a FAISS vector index. When a user asks a question, the system performs semantic similarity search to retrieve relevant sections from the uploaded papers and provides those sections to Google Gemini for grounded response generation.

The assistant is designed to keep its answers grounded in the uploaded research papers rather than relying on outside knowledge. Its response pipeline also preserves document and page information so users can identify where retrieved information originated.

The application is implemented with FastAPI and provides a web-based interface using HTML, CSS, JavaScript, and Jinja2. The backend exposes endpoints for document upload, system status, question answering, and document management.

---

## Problem Statement

Research papers often contain large amounts of technical information distributed across multiple sections and pages. Finding specific information manually can be time-consuming, especially when working with several papers simultaneously.

Traditional keyword-based searching may also fail when the wording of a question differs from the wording used in the document.

PaperMind AI addresses this problem by combining document processing, semantic embeddings, vector similarity search, and generative AI into a single research-focused workflow.

The objective is to allow users to ask questions naturally while keeping generated responses connected to the actual content of their uploaded papers.

---

## Solution

PaperMind AI implements a complete document-grounded RAG pipeline:

1. Upload research paper PDFs.
2. Extract text page by page.
3. Split extracted text into manageable chunks.
4. Generate embeddings for each chunk.
5. Store normalized embeddings in a FAISS vector index.
6. Store document, page, chunk, and text metadata.
7. Convert each user question into an embedding.
8. Perform similarity search against the FAISS index.
9. Retrieve the most relevant document chunks.
10. Send the retrieved context to Google Gemini.
11. Generate an answer using only the retrieved research-paper context.
12. Return the answer together with source document, page, and similarity information.

The implementation uses FAISS inner-product search with normalized vectors, allowing the search to approximate cosine similarity.

---

## How It Works / RAG Pipeline

```text
PDF Upload
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Embedding Generation
     ↓
FAISS Vector Store
     ↓
Similarity Search
     ↓
Relevant Context Retrieval
     ↓
Gemini Response Generation
     ↓
Source-Aware Answer
```

### 1. PDF Upload

Users can upload one or multiple PDF research papers through the web application.

The backend validates file extensions, checks for empty files, applies a configurable file-size limit, and stores uploaded files using generated safe filenames.

### 2. Text Extraction

PaperMind AI uses **PyMuPDF** to open PDF documents and extract text from each page.

Page numbers are preserved alongside the extracted text, allowing later retrieval results to identify the relevant page.

### 3. Text Chunking

Extracted page text is divided into smaller overlapping chunks.

The default implementation uses:

* Chunk size: `1200`
* Chunk overlap: `180`

These values are configurable through environment variables.

### 4. Embedding Generation

Each text chunk is converted into a vector embedding using the configured Gemini embedding model.

The implementation uses an embedding dimension of `768` and normalizes vectors before storing them in FAISS.

### 5. FAISS Vector Store

The generated embeddings are stored in a FAISS `IndexFlatIP` index.

Alongside the vectors, PaperMind AI stores metadata containing:

* Document name
* Document ID
* Page number
* Chunk number
* Original chunk text

The FAISS index and metadata are persisted locally.

### 6. Similarity Search

When a user asks a question, the question is converted into an embedding and searched against the FAISS index.

The system retrieves up to the configured `TOP_K` results and filters results using the configured minimum similarity threshold.

### 7. Relevant Context Retrieval

The retrieved chunks are assembled into a context containing the source document and page information.

This context is then supplied to Gemini rather than asking the model to answer without document evidence.

### 8. Gemini Response Generation

Google Gemini receives the user's question together with the retrieved research-paper context.

The generation prompt explicitly instructs the model to:

* Use only the uploaded-paper context.
* Avoid outside knowledge.
* Avoid inventing information.
* State when the information is unavailable.
* Mention the relevant paper and page when useful.
* Support important claims using the provided context.

### 9. Source-Aware Answer

The API returns the generated answer together with source metadata including:

* Source document
* Page
* Similarity score

This makes the response more transparent and easier to verify.

---

## Key Features

* Upload one or multiple research paper PDFs
* PDF text extraction using PyMuPDF
* Page-aware document processing
* Configurable text chunking
* Overlapping document chunks
* Gemini-powered embeddings
* FAISS vector storage
* Semantic similarity search
* Configurable retrieval count
* Configurable similarity threshold
* Google Gemini answer generation
* Document-grounded responses
* Source document information
* Source page information
* Similarity/relevance information
* Persistent FAISS index
* Document metadata persistence
* Document statistics
* Document management
* Professional web-based chat interface
* Empty-question validation
* Unsupported-file validation
* Empty-document handling
* Missing-document handling
* Gemini retry and backup-model handling

---

## Technology Stack

| Technology        | Purpose                                 |
| ----------------- | --------------------------------------- |
| Python            | Core application and RAG implementation |
| FastAPI           | Backend web framework and API           |
| Google Gemini API | Embeddings and response generation      |
| FAISS             | Vector storage and similarity search    |
| PyMuPDF           | PDF text extraction                     |
| NumPy             | Vector and numerical operations         |
| Jinja2            | HTML template rendering                 |
| HTML              | Frontend structure                      |
| CSS               | Frontend styling                        |
| JavaScript        | Frontend interaction                    |
| python-dotenv     | Environment variable management         |

The backend imports and integrates FastAPI, Jinja2, python-dotenv, and the `RAGEngine` implementation.

The RAG engine integrates FAISS, PyMuPDF, NumPy, python-dotenv, and the Google Gemini SDK.

---

## Project Architecture

```text
                    ┌─────────────────────────┐
                    │       Web Interface     │
                    │     HTML / CSS / JS     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        FastAPI          │
                    │     Application API     │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
        ┌───────────────────┐        ┌────────────────────┐
        │   PDF Processing  │        │    Chat Request    │
        │    PyMuPDF        │        │      Question      │
        └─────────┬─────────┘        └─────────┬──────────┘
                  │                            │
                  ▼                            ▼
        ┌───────────────────┐        ┌────────────────────┐
        │   Text Chunks     │        │ Question Embedding │
        └─────────┬─────────┘        └─────────┬──────────┘
                  │                            │
                  ▼                            ▼
        ┌───────────────────┐        ┌────────────────────┐
        │ Gemini Embeddings │        │   FAISS Search     │
        └─────────┬─────────┘        └─────────┬──────────┘
                  │                            │
                  ▼                            ▼
        ┌─────────────────────────────────────────────────┐
        │              FAISS Vector Store                 │
        │          + Document/Page Metadata               │
        └───────────────────────┬─────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ Relevant Context        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Google Gemini       │
                    │   Grounded Generation  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Source-Aware Response   │
                    └─────────────────────────┘
```

---

## Project Structure

A typical project structure for PaperMind AI is:

```text
PaperMind-AI/
│
├── main.py
├── rag.py
├── requirements.txt
├── .env
├── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── uploads/
│
└── faiss_db/
    ├── research_papers.index
    └── metadata.json
```

The application defines the upload directory, static files, Jinja2 templates, and persistent FAISS storage directly in the backend configuration.

The exact frontend filenames may vary depending on the final project files.

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd PaperMind-AI
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key
```

Optional configuration variables supported by the RAG engine include:

```env
GEMINI_MODEL=gemini-3.8-flash
GEMINI_BACKUP_MODEL=gemini-3.7-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

TOP_K=6
CHUNK_SIZE=1200
CHUNK_OVERLAP=180
MIN_SIMILARITY=0.22

MAX_FILE_SIZE_MB=25
```

The application requires `GEMINI_API_KEY`. The RAG engine also supports configurable generation models, embedding model, retrieval count, chunk size, chunk overlap, and similarity threshold.

The API uses `MAX_FILE_SIZE_MB` for the upload size limit and defaults it to `25 MB`.

Do not commit your `.env` file or API keys to GitHub.

---

## How to Run Locally

After activating the virtual environment and configuring the `.env` file, start the FastAPI application with:

```bash
uvicorn main:app --reload
```

The application can then be accessed through the local FastAPI server.

For a production-oriented run without auto-reload:

```bash
uvicorn main:app
```

---

## How to Use

### Step 1 — Upload Research Papers

Upload one or more research paper PDFs through the web interface.

PaperMind AI validates the uploaded files before indexing them.

### Step 2 — Wait for Indexing

For every valid PDF, the system:

```text
PDF
 ↓
Page Text
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS Index
```

The system also records document and page metadata.

### Step 3 — Ask a Question

Enter a question about the uploaded research papers.

For example:

```text
What problem does this paper address?
```

### Step 4 — Retrieve Relevant Information

The question is embedded and compared with the stored document vectors.

### Step 5 — Generate the Answer

Relevant chunks are passed to Google Gemini with instructions to answer using only the uploaded-paper context.

### Step 6 — Review Sources

The response includes source information such as the document name, page number, and similarity score.

---

## Example Questions

PaperMind AI can be used for questions such as:

```text
Summarize this research paper.
```

```text
What problem does the paper address?
```

```text
Explain the methodology used.
```

```text
Which dataset was used?
```

```text
What algorithms or models were used?
```

```text
What are the main findings?
```

```text
What are the limitations?
```

```text
What future work is suggested?
```

Users can also ask more specific questions about information contained in the uploaded research papers.

---

## RAG Workflow

The complete retrieval-augmented generation workflow is:

```text
1. PDF Upload
       ↓
2. Text Extraction
       ↓
3. Text Chunking
       ↓
4. Embedding Generation
       ↓
5. FAISS Vector Store
       ↓
6. Question Embedding
       ↓
7. Similarity Search
       ↓
8. Relevant Context Retrieval
       ↓
9. Gemini Response Generation
       ↓
10. Source-Aware Answer
```

### Indexing Workflow

```text
Research Paper PDF
        ↓
PyMuPDF
        ↓
Page-Level Text
        ↓
Chunking
        ↓
Gemini Embedding Model
        ↓
Normalized Vectors
        ↓
FAISS Index
        +
Metadata
```

### Question-Answering Workflow

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Top Relevant Chunks
      ↓
Similarity Threshold
      ↓
Research-Paper Context
      ↓
Google Gemini
      ↓
Grounded Answer
      +
Source Information
```

The implementation searches the FAISS index using the question embedding and returns only results that meet the configured similarity threshold.

---

## Grounding and Hallucination Control

A central design principle of PaperMind AI is **document grounding**.

The assistant is explicitly instructed to answer only from the uploaded research-paper context.

Its generation rules include:

```text
Do not use outside knowledge.
Do not invent information.
If the answer is not available in the uploaded papers, clearly say so.
Mention the relevant paper and page when useful.
Support important claims using the provided context.
```

If semantic search does not find sufficiently relevant information, PaperMind AI returns a response indicating that it could not find enough relevant information in the uploaded research papers instead of generating an unsupported answer.

This behavior is particularly important for research-oriented applications where unsupported claims can reduce the reliability of an answer.

---

## API Endpoints

| Method   | Endpoint         | Purpose                                         |
| -------- | ---------------- | ----------------------------------------------- |
| `GET`    | `/`              | Loads the main research assistant interface     |
| `GET`    | `/api/status`    | Returns uploaded documents and chunk statistics |
| `POST`   | `/api/upload`    | Uploads and indexes research paper PDFs         |
| `POST`   | `/api/chat`      | Answers a question using indexed papers         |
| `DELETE` | `/api/documents` | Clears uploaded documents and indexed data      |

The implemented FastAPI application defines these routes directly in `main.py`.

### `GET /`

Loads the main interface and provides document and statistics information to the Jinja2 template.

### `GET /api/status`

Returns:

```json
{
  "documents": [],
  "document_count": 0,
  "chunk_count": 0
}
```

The actual values depend on the currently indexed documents.

### `POST /api/upload`

Accepts multiple uploaded files and indexes valid PDFs.

The response contains information about successfully uploaded documents, errors, current documents, and chunk count.

### `POST /api/chat`

Accepts a question in the request body:

```json
{
  "question": "What methodology does the paper use?"
}
```

The endpoint validates the question, verifies that documents are available, and passes the question to the RAG engine.

### `DELETE /api/documents`

Clears the indexed documents and removes uploaded files from the application's upload directory.

---

## Error Handling

PaperMind AI includes validation and error handling for common application scenarios.

### Unsupported Files

Only PDF files are accepted.

```text
only PDF files are supported.
```

### Empty Files

Empty uploaded files are rejected.

```text
the file is empty.
```

### File Size Limit

Files exceeding the configured maximum size are rejected.

The default limit is `25 MB`.

### PDFs Without Extractable Text

PDFs without readable text are not indexed.

```text
no extractable text was found.
```

### Empty Questions

The chat endpoint rejects empty questions.

```text
Please enter a question about the uploaded papers.
```

### No Uploaded Documents

Questions cannot be processed until at least one research paper has been indexed.

### Gemini API Failures

The RAG engine supports retries for transient API-related failures and can attempt a configured backup Gemini model if the primary generation model fails.

---

## Future Improvements

Potential future improvements include:

* More advanced document-aware chunking strategies
* Improved retrieval and reranking techniques
* Additional document formats
* Citation links that navigate directly to source pages
* More detailed retrieval transparency
* Conversation history
* User-specific document collections
* Authentication and access control
* Cloud-based vector storage
* Background document processing
* Batch embedding optimization
* Advanced research-paper metadata extraction
* Improved evaluation and retrieval benchmarking
* Additional model providers
* Production deployment configuration

These are potential future directions and are not currently represented as implemented features.

---

## Deployment

PaperMind AI is designed around a FastAPI backend and can be run locally using an ASGI server such as Uvicorn.

For deployment, the application would require:

1. A Python runtime.
2. Project dependencies installed from `requirements.txt`.
3. A valid Gemini API key.
4. Appropriate environment-variable configuration.
5. Persistent storage for uploaded documents and the FAISS index.

The current implementation stores uploads and FAISS data locally, so any production deployment should account for persistent storage requirements. The configured FAISS index and metadata are stored under the application's `faiss_db` directory.

No specific cloud provider or production infrastructure is assumed by this project.

---

## GitHub / Project Information

**Project Name**

```text
PaperMind AI
```

**Project Type**

```text
AI / Machine Learning
Retrieval-Augmented Generation
Research Paper Assistant
Document Question Answering
```

**Primary Focus**

```text
Document-grounded question answering using RAG
```

**Core Technologies**

```text
Python
FastAPI
Google Gemini
FAISS
PyMuPDF
NumPy
Jinja2
HTML
CSS
JavaScript
python-dotenv
```

**Repository Description**

> A document-grounded AI research assistant that uses Retrieval-Augmented Generation, FAISS, and Google Gemini to answer questions from uploaded research papers.

---

## License

This project does not specify a license in the provided project requirements.

If the repository is intended for public distribution, add an appropriate license file and update this section accordingly.

---
