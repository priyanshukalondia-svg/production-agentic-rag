# Production Agentic RAG

Production Agentic RAG is a production-oriented retrieval-augmented generation system built for offline experimentation and local deployment. It combines hybrid retrieval, agentic planning, grounding checks, guardrails, self-correction, source-aware citations, and a browser-based dashboard into a single, testable architecture.

The project is designed to work in three modes:

- Default handbook mode using the built-in knowledge base
- Optional custom knowledge mode with user-uploaded manuals and knowledge documents
- Optional custom Q&A knowledge that is treated as a high-priority retrieval source

Knowledge priority is structured as:

1. User Q&A
2. User custom handbook
3. Default handbook

This allows the system to answer from a user-specific knowledge base while keeping the original handbook available as a fallback and reference source.

## Overview

The system includes:

- Hybrid retrieval using BM25 and dense vector search
- Reciprocal Rank Fusion (RRF) to combine lexical and semantic rankings
- Optional custom knowledge ingestion for uploaded documentation
- A custom Q&A layer for high-priority answer lookup
- Agentic query decomposition and multi-hop retrieval
- Grounding and faithfulness evaluation
- Guardrails for unsafe or suspicious user input
- Self-correction when retrieved context is insufficient
- FastAPI endpoints for local serving
- A React dashboard for chat and knowledge management
- Offline deterministic behavior for local testing and CI

## Core architecture

```text
User question
    ↓
Input guardrails
    ↓
Query planning / decomposition
    ↓
Retrieval over default + custom knowledge sources
    ↓
Hybrid ranking + reranking
    ↓
Grounded synthesis
    ↓
Faithfulness check + self-correction if needed
    ↓
Answer + citations + metadata
```

## Feature set

### Default knowledge base

The default corpus remains intact and is always available. It serves as the baseline knowledge source and fallback for situations where custom knowledge is empty, irrelevant, or insufficient.

### Custom knowledge base

Users can optionally add a custom handbook or knowledge document through the dashboard or API. The uploaded document is processed through the same ingestion and retrieval pipeline used by the default handbook, but stored separately so the system can distinguish between sources.

Custom knowledge includes:

- Manual or handbook upload
- Custom document indexing
- Optional Q&A pair management
- Enable / disable switch for custom knowledge
- Custom knowledge removal
- Source-aware traceability through retrieved citations

### Knowledge hierarchy

The project implements a clear resolution order:

- User Q&A entries are treated as the highest-priority knowledge
- Custom handbook content is next in priority
- Default handbook is retained as the final fallback and standard source

This hierarchy does not bypass existing guardrails, faithfulness checks, or retrieval quality evaluation. The system still decides based on grounding and relevance, rather than forcing a custom answer when the context does not justify it.

## Dashboard

The frontend is a lightweight but production-styled chat interface with optional knowledge management tools built directly into the same experience.

It includes:

- Chat input and assistant responses
- Suggested prompts
- Citation display
- Faithfulness and safety status
- Optional custom knowledge panel
- Upload custom handbook
- Toggle custom knowledge on/off
- Add, edit, and delete custom Q&A entries
- Remove custom knowledge entirely
- Clear source indication between default and custom knowledge

## Project structure

```text
production-agentic-rag-main/
├── api/
│   └── main.py                  # FastAPI app and custom knowledge endpoints
├── data/
│   ├── corpus/                  # Default handbook corpus
│   └── eval/                   # Evaluation dataset
├── docs/
│   └── ARCHITECTURE.md         # Project architecture notes
├── examples/
│   ├── quickstart.py
│   └── with_integrations.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── src/
│   └── production_agentic_rag/
│       ├── agents/
│       ├── evaluation/
│       ├── ingestion/
│       ├── providers/
│       ├── retrieval/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── graph.py
│       ├── guardrails.py
│       ├── observability.py
│       └── pipeline.py
├── tests/
│   ├── test_agent.py
│   ├── test_custom_knowledge.py
│   ├── test_guardrails.py
│   ├── test_integrations.py
│   ├── test_langgraph.py
│   ├── test_pipeline.py
│   ├── test_retrieval.py
│   └── test_text.py
├── Dockerfile
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
└── .gitignore
```

## Technical stack

### Backend

- Python 3.10+
- FastAPI
- Pydantic
- Uvicorn
- Pure Python BM25 implementation
- Hashing-based dense embeddings for offline use

### Retrieval and reasoning

- BM25 lexical retrieval
- Dense vector search
- Hybrid retrieval with reciprocal rank fusion
- Reranking
- Retrieval tool abstraction for agent calls
- Multi-hop planning and self-correction

### Frontend

- React
- Vite
- CSS modules / custom styling
- Lucide React icons

## Local setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd production-agentic-rag-main
```

### 2. Create a virtual environment

On Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -e ".[api,dev]"
```

If you are using the upload API route, install the multipart parser as well:

```bash
pip install python-multipart
```

### 4. Start the backend

From the project root:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Available endpoints include:

- `/health`
- `/ask`
- `/knowledge/custom`
- `/knowledge/custom/enable`
- `/knowledge/custom/index`
- `/knowledge/custom/upload`
- `/knowledge/custom/qa`
- `/knowledge/custom/qa/{qa_id}`
- `/knowledge/custom` (DELETE)

### 5. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Then open:

```text
http://localhost:5173
```

## API usage

### Ask a question

```http
POST /ask
Content-Type: application/json
```

```json
{
  "question": "What is the return policy?"
}
```

Example response:

```json
{
  "answer": "Customers may return unused items within 30 days.",
  "citations": ["handbook:0"],
  "faithfulness": 1.0,
  "iterations": 1,
  "blocked": false
}
```

### Custom knowledge example

```http
POST /knowledge/custom/index
Content-Type: application/json
```

```json
{
  "name": "internal-policy",
  "text": "# Refund Policy\nCustomers can request a refund within 14 days."
}
```

```http
POST /knowledge/custom/qa
Content-Type: application/json
```

```json
{
  "question": "What is our refund policy?",
  "answer": "Customers can request a refund within 14 days."
}
```

```http
POST /knowledge/custom/enable
Content-Type: application/json
```

```json
{
  "enabled": true
}
```

## Testing and evaluation

Run the project test suite:

```bash
pytest -q
```

The repository includes retrieval and answer evaluation harnesses:

```bash
prag eval --k 3
```

The evaluation system checks quality metrics such as:

- retrieval hit rate
- MRR
- recall@k
- nDCG
- answer faithfulness
- answer similarity

## Offline mode and provider flexibility

The project is built to run without external API keys for local development and CI. The default deterministic mock LLM and hashing-based embeddings allow the system to run end-to-end offline.

It also supports optional provider integrations for:

- OpenAI
- Azure OpenAI
- LangGraph execution mode
- MCP-connected integrations

## Notes

This project deliberately keeps the default handbook and custom knowledge logically separated. Retrieval, citations, and evaluation can identify the source of a retrieved chunk, while preserving the standard fallback semantics of the default corpus.

The system is designed to be modular, testable, and operationally realistic without depending on a single monolithic storage layer.

## License

MIT
