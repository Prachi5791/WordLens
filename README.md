# WordLens

WordLens is a technical terminology detection and explanation platform for PDFs and text. It combines:

- a curated technical vocabulary dataset
- semantic retrieval with FAISS
- a FastAPI backend for term lookup and explanation
- a Vite + React frontend for upload and interactive inspection

The project is designed to answer two primary questions:

1. Which technical terms appear in a document or text snippet?
2. What do those terms mean, grounded in retrieved evidence and local knowledge?

---

## High-level flow

The project follows this end-to-end pipeline:

1. Source glossaries are ingested and normalized by the dataset builder.
2. The curated vocabulary is stored in `data/processed/technical_terms.json` and exported to CSV/JSON artifacts.
3. A FAISS vectorstore is built from the cleaned terminology records for semantic retrieval.
4. The FastAPI app loads the dataset and vectorstore at startup and validates that the runtime dependencies exist.
5. A user uploads a PDF or provides text.
6. The backend detects technical terms using local vocabulary + semantic matching.
7. Term lookup and RAG-style explanation APIs return grounded definitions and evidence.
8. The React frontend renders the document, detects terms, and shows popover explanations.

---

## Project structure

```text
WordLens/
├── .env.example
├── README.md
├── requirements-day1.txt
├── backend/
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── core/config.py
│   │   ├── db/database.py
│   │   ├── db/import_terms.py
│   │   ├── repositories/terminology.py
│   │   ├── services/
│   │   │   ├── detector.py
│   │   │   ├── document.py
│   │   │   ├── knowledge.py
│   │   │   ├── rag.py
│   │   │   ├── retrieval.py
│   │   │   └── terminology.py
│   │   └── main.py
│   └── tests/
├── data/
│   ├── processed/
│   ├── raw/
│   ├── evaluation/
│   └── uploads/
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   ├── build_technical_dataset.py
│   ├── dynamic_knowledge.py
│   ├── export_runtime_dataset.py
│   ├── rag.py
│   ├── retrieval.py
│   ├── scientific_term_discovery.py
│   └── technical_term_detector.py
├── vectorstore/
├── models/
├── notebooks/
├── tests/
└── docs/data artifacts and evaluation outputs
```

---

## Technical flow in detail

### 1) Data source and vocabulary build

The dataset builder lives in `scripts/build_technical_dataset.py`.

It does the following:

- reads source glossary pages from public sources such as Google ML docs and NIST glossary entries
- normalizes terminology strings and aliases
- classifies domain/subdomain using heuristic rules
- validates candidate terms against quality checks
- saves output to:
  - `data/processed/technical_terms.json`
  - `data/processed/technical_terms.csv`
  - `data/evaluation/validation_report.json`

This is the curated seed vocabulary that powers the app.

Important project behavior:

- the project keeps a reproducible seed dataset in `data/processed/technical_terms.json`
- the dataset is used as a baseline, not a hardcoded fixed-size list
- new or enriched knowledge can be added without replacing the seed dataset wholesale

### 2) Semantic retrieval index

The vectorstore is built by `scripts/retrieval.py`.

This script:

- loads the processed technical terms dataset
- normalizes records
- generates embeddings using `BAAI/bge-small-en-v1.5`
- stores them in FAISS format under `vectorstore/`
- emits metadata.json, records.json, embeddings.npy, and build_report.json

The index is essential for the semantic matching layer used by the backend. It lets the app retrieve conceptually similar terms even when exact lexical matches are not present.

### 3) Backend startup and runtime validation

The app entry point is `backend/app/main.py`.

At startup, it:

- loads settings from environment variables
- configures logging
- checks that the dataset exists
- checks that all required FAISS files are present
- initializes the database
- creates runtime serving tables

This is the guardrail that prevents the API from starting in a partially configured state.

### 4) API and request flow

The main API router is in `backend/app/api/routes.py`.

Core endpoints:

- `GET /health` — checks backend, DB, and vectorstore readiness
- `POST /upload` — uploads a PDF and extracts document statistics
- `GET /term/{term}` — resolves a single term from local knowledge
- `POST /explain` — explains a term using retrieved evidence and contextual grounding
- `POST /explain-selection` — extracts terms from raw text and explains them

The application flow for a term explanation usually looks like this:

1. incoming term or text is received by the API
2. `detect_text` or `detect_pdf` identifies candidate technical terms
3. `lookup_term` checks the local terminology repository and matching logic
4. relevant records are retrieved from the vectorstore
5. `explain` builds a grounded explanation using retrieved evidence and contextual text
6. the response includes canonical definition, generated explanation, evidence, and warnings

### 5) PDF processing and detection

PDF upload processing is handled by `backend/app/services/document.py`.

This file:

- validates PDF type and size
- ensures the uploaded file is safe
- extracts text from PDF pages using `PyMuPDF`
- calls the term detection service
- returns metadata and detected terms for the frontend

The actual detection logic is in `backend/app/services/detector.py`.

This part combines:

- exact term matching against the dataset
- phrase-level candidate extraction
- semantic similarity against the vectorstore
- confidence and evidence scoring

### 6) Term resolution and grounding

The terminology service (`backend/app/services/terminology.py`) is the business logic that decides whether a term is:

- known and canonical
- partially matched
- inferred through retrieval
- not confidently resolvable

The RAG layer (`backend/app/services/rag.py`) turns these lookup results into a human-readable explanation using context-aware evidence rather than a raw dictionary definition alone.

---

## Frontend flow

The frontend is in `frontend/` and is built with Vite + React.

The main UI in `frontend/src/App.jsx`:

- checks the backend health on mount
- lets the user upload a PDF
- extracts technical terms from the document
- shows hover or selection-based explanations
- combines canonical meaning with retrieval-grounded explanation when available

The frontend calls the backend endpoints defined in `frontend/src/api/wordlensApi.js`.

---

## Setup and run

### 1) Python environment

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-day1.txt
```

### 2) Environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

The key values are similar to:

```env
WORDLENS_APP_ENV=development
WORDLENS_HOST=127.0.0.1
WORDLENS_PORT=8000
WORDLENS_DATABASE_URL=mysql+pymysql://wordlens:CHANGE_ME@127.0.0.1:3306/wordlens
WORDLENS_UPLOAD_DIR=data/uploads
WORDLENS_VECTORSTORE_DIR=vectorstore
WORDLENS_DATASET_PATH=data/processed/technical_terms.json
WORDLENS_MAX_UPLOAD_BYTES=10485760
WORDLENS_LOG_LEVEL=INFO
WORDLENS_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

### 3) Start the backend

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

The API will validate the dataset and vectorstore before starting successfully.

### 4) Start the frontend

From `frontend/`:

```bash
npm install
npm run dev
```

The app is typically served on:

- frontend: `http://localhost:5173`
- backend: `http://127.0.0.1:8000`

---

## Useful commands

### Build or refresh the terminology dataset

```bash
python scripts/build_technical_dataset.py --offline
```

For a source refresh:

```bash
python scripts/build_technical_dataset.py --refresh
```

### Rebuild vectorstore

```bash
python scripts/retrieval.py build
python scripts/retrieval.py validate
```

### Import runtime definitions into MySQL

```bash
python -m backend.app.db.import_terms
```

### Health-check the service

```bash
curl http://127.0.0.1:8000/health
```

---

## Example API usage

### Check readiness

```bash
curl http://127.0.0.1:8000/health
```

### Resolve a known term

```bash
curl http://127.0.0.1:8000/term/attention
```

### Explain a term using context

```bash
curl -X POST http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "term": "transformer",
    "context": "The transformer uses self-attention to model long-range dependencies in text.",
    "top_k": 5,
    "include_retrieval_evidence": true
  }'
```

### Explain a selection from text

```bash
curl -X POST http://127.0.0.1:8000/explain-selection \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A transformer model uses self-attention and positional encoding.",
    "top_k_per_term": 3,
    "max_terms": 10
  }'
```

---

## Data and knowledge lifecycle

The project separates three layers:

1. Curated seed vocabulary
   - stored in `data/processed/technical_terms.json`
   - intended as the reproducible source of truth

2. Semantic retrieval layer
   - stored in `vectorstore/`
   - used to find likely related terminology from embeddings

3. Runtime knowledge / MySQL layer
   - populated via `backend/app/db/import_terms.py`
   - supports runtime serving and enriched knowledge persistence

This design keeps the project stable: the locally curated vocabulary remains the baseline, while retrieval and runtime database records enrich the system without destroying the original dataset.

---

## Notes and expected behavior

- The app expects the processed dataset and vectorstore to exist before the backend starts.
- The backend is designed to fail fast if the dataset or vectorstore is missing or incomplete.
- The terminology detection system is open-world and tries to resolve terms even when they are not exact dataset matches.
- RAG explanations are grounded in retrieved context, rather than returning a generic dictionary-style answer.
- The frontend is meant to be an interactive document-analysis layer over the backend API.

---

## Common troubleshooting

### Backend says vectorstore is incomplete

Rebuild the index:

```bash
python scripts/retrieval.py build
```

### Backend says dataset not found

Check that the file exists:

```bash
ls data/processed/technical_terms.json
```

### Frontend cannot reach backend

Ensure the backend is running on port 8000 and CORS is configured correctly in `.env`.

### Database errors

Make sure MySQL is running and `WORDLENS_DATABASE_URL` points to a valid database.

---

## Summary

WordLens is a document-aware technical terminology system that combines:

- glossary data collection
- semantic embedding retrieval
- knowledge lookup and grounding
- PDF processing and term detection
- a FastAPI API + React frontend

Together, these components create a full retrieval-and-explanation pipeline for technical language in documents.
