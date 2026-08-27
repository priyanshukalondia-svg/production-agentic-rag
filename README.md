# Production Agentic RAG

A production-style Agentic RAG application built with Python, FastAPI, and React.

I built the frontend for this project and connected it with the existing RAG backend to turn the project into a complete interactive application that can be run locally.

The system takes a user's question, retrieves relevant information from the available knowledge base, and generates an answer along with citations and evaluation details.

## What it does

The backend uses an agentic RAG pipeline with:

- Query planning and decomposition
- Hybrid retrieval using BM25 and vector search
- Reciprocal Rank Fusion (RRF)
- Multi-hop retrieval
- Reranking
- Guardrails for unsafe or suspicious input
- Faithfulness checking
- Self-correction when the answer is not sufficiently grounded
- Evaluation metrics for retrieval and answers

The project can also run in an offline mode using deterministic mock components, so it doesn't require an API key just to test the basic pipeline.

## Frontend

I added a React frontend to make the RAG system easier to interact with.

The frontend includes:

- A chat-based interface
- Suggested questions
- Conversation history during the session
- Answer citations
- Faithfulness score
- Iteration information
- New conversation option
- Direct communication with the FastAPI backend

## Tech used

### Backend

- Python
- FastAPI
- Pydantic
- Uvicorn

### RAG pipeline

- BM25 retrieval
- Vector retrieval
- Hybrid search
- Reciprocal Rank Fusion
- Reranking
- Agentic query planning
- Guardrails
- Evaluation

### Frontend

- React
- Vite
- CSS
- Lucide React

## Project structure

```text
production-agentic-rag/
│
├── api/                    # FastAPI application
├── data/                   # Sample corpus and evaluation data
├── docs/                   # Architecture documentation
├── examples/               # Example scripts
├── frontend/               # React frontend
├── src/
│   └── production_agentic_rag/
│       ├── agents/
│       ├── evaluation/
│       ├── ingestion/
│       ├── providers/
│       └── retrieval/
│
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Running the project locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### 2. Create and activate a virtual environment

On Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install the backend dependencies

```bash
pip install -e ".[api,dev]"
```

### 4. Start the backend

From the project root:

```bash
uvicorn api.main:app --reload
```

The backend should now be available at:

```text
http://127.0.0.1:8000
```

You can check:

- `/health` — API health check
- `/docs` — FastAPI interactive documentation
- `/ask` — RAG question endpoint

### 5. Start the frontend

Open another terminal and go into the frontend folder:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

Then start the app:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

## API example

Send a POST request to:

```text
/ask
```

With:

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

## Running tests

```bash
pytest -q
```

The project also includes an evaluation dataset and harness:

```bash
prag eval --k 3
```

This evaluates things like retrieval quality, MRR, recall, nDCG, faithfulness, and answer similarity.

## A few things I want to improve

Some things I would like to add next:

- Document upload from the frontend
- Chat history persistence
- Streaming responses
- Better vector database support
- Authentication
- Deployment of the frontend and backend
- Support for real LLM providers in the deployed version

## Notes

The core RAG system supports offline/mock components, OpenAI, and Azure OpenAI configurations. The offline setup makes it easier to test the project without requiring API keys.

The frontend and backend are currently configured to run locally together using React/Vite and FastAPI.

## License

MIT
