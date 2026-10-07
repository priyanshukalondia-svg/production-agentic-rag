"""FastAPI service exposing the RAG pipeline (optional; needs fastapi+uvicorn)."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from production_agentic_rag import ChatAgent, RAGPipeline, Settings
from production_agentic_rag.ingestion.loaders import load_directory

app = FastAPI(title="Production Agentic RAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
        "https://frontend-five-phi-44.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(Settings.from_env())
        corpus = Path(__file__).resolve().parents[1] / "data" / "corpus"
        if corpus.exists():
            _pipeline.add_documents(load_directory(corpus))
    return _pipeline


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    faithfulness: float
    iterations: int
    blocked: bool


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    intent: str
    used_retrieval: bool
    citations: list[str] = Field(default_factory=list)
    grounding: float | None = None
    iterations: int = 0
    blocked: bool = False
    debug: dict[str, object] | None = None


class CustomToggleRequest(BaseModel):
    enabled: bool


class CustomQAPairRequest(BaseModel):
    question: str
    answer: str


class CustomIndexRequest(BaseModel):
    text: str
    name: str = "custom-handbook"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Production Agentic RAG",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    agent = ChatAgent(get_pipeline(), settings=get_pipeline().settings)
    cleaned_history = [
        {"role": item.role, "content": item.content}
        for item in req.history
    ]
    result = agent.chat(req.message, cleaned_history, conversation_id=req.conversation_id)
    return ChatResponse(
        message=result["message"],
        conversation_id=result["conversation_id"],
        intent=result["intent"],
        used_retrieval=result["used_retrieval"],
        citations=result.get("citations", []),
        grounding=result.get("grounding"),
        iterations=result.get("iterations", 0),
        blocked=result.get("blocked", False),
        debug=result.get("reason"),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Backward-compatible wrapper around the original single-question API."""
    res = get_pipeline().query(req.question)
    return AskResponse(answer=res.answer, citations=res.citations,
                       faithfulness=res.faithfulness, iterations=res.iterations,
                       blocked=res.blocked)


@app.get("/knowledge/custom")
def list_custom_knowledge() -> dict[str, object]:
    return get_pipeline().list_custom_knowledge()


@app.post("/knowledge/custom/enable")
def set_custom_knowledge(req: CustomToggleRequest) -> dict[str, object]:
    enabled = get_pipeline().enable_custom_knowledge(req.enabled)
    return {"enabled": enabled, "status": "custom knowledge enabled" if enabled else "default-only mode"}


@app.post("/knowledge/custom/index")
def index_custom_knowledge(req: CustomIndexRequest) -> dict[str, object]:
    try:
        added = get_pipeline().add_custom_document(req.text, req.name)
        return {"status": "indexed", "name": req.name, "chunks": added, "enabled": get_pipeline().custom_enabled}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/knowledge/custom/upload")
async def upload_custom_handbook(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload requires a file name.")

    allowed = {".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a .txt or .md file.")

    if file.size and file.size > 2_000_000:
        raise HTTPException(status_code=413, detail="Uploaded file is too large. Keep it below 2 MB.")

    try:
        raw = await file.read()
        if not raw:
            raise ValueError("Uploaded file is empty.")
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    doc_name = Path(file.filename).stem or "custom-handbook"
    try:
        added = get_pipeline().add_custom_document(text, doc_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "uploaded", "name": file.filename, "chunks": added, "enabled": get_pipeline().custom_enabled}


@app.post("/knowledge/custom/qa")
def add_custom_qa(req: CustomQAPairRequest) -> dict[str, object]:
    try:
        qa_id = get_pipeline().add_custom_qa(req.question, req.answer)
        return {"status": "added", "id": qa_id, "question": req.question, "answer": req.answer}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/knowledge/custom/qa/{qa_id}")
def update_custom_qa(qa_id: str, req: CustomQAPairRequest) -> dict[str, object]:
    try:
        item = get_pipeline().update_custom_qa(qa_id, req.question, req.answer)
        return {"status": "updated", "id": qa_id, **item}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/knowledge/custom/qa/{qa_id}")
def delete_custom_qa(qa_id: str) -> dict[str, str]:
    try:
        get_pipeline().delete_custom_qa(qa_id)
        return {"status": "deleted", "id": qa_id}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/knowledge/custom")
def delete_custom_knowledge() -> dict[str, str]:
    get_pipeline().remove_custom_knowledge()
    return {"status": "removed"}
