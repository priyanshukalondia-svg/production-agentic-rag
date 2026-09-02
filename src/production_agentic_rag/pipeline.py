"""Top-level RAG pipeline: ingest → guardrails → agentic retrieval+generation → report."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .agents.rag_agent import AgenticRAG
from .agents.tools import RetrievalTool
from .config import Settings
from .guardrails import InputBlocked, check_input, grounding_score, redact_pii
from .ingestion.chunking import CHUNKERS, Chunk
from .ingestion.loaders import Document
from .observability import Tracer
from .providers.embeddings import build_embedder
from .providers.llm import build_llm
from .retrieval.hybrid import HybridRetriever, Scored


@dataclass
class QueryResult:
    question: str
    answer: str
    citations: list[str]
    contexts: list[Scored]
    faithfulness: float
    iterations: int
    blocked: bool = False
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def context_texts(self) -> list[str]:
        return [c.chunk.text for c in self.contexts]


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.custom_knowledge: dict[str, list[dict[str, Any]]] = {"handbook": [], "qa": []}
        self._custom_handbook_texts: dict[str, str] = {}
        self.custom_enabled = False
        self.retriever = HybridRetriever(
            embedder=build_embedder(self.settings.embed_provider, self.settings.embed_model),
            custom_enabled=self.custom_enabled,
        )
        self.llm = build_llm(self.settings.llm_provider, self.settings.llm_model)
        self.tool = RetrievalTool(self.retriever, k=self.settings.top_k,
                                  candidate_k=self.settings.candidate_k,
                                  rerank=self.settings.rerank)
        self.agent = AgenticRAG(
            llm=self.llm, tool=self.tool,
            model=("mock" if self.settings.llm_provider == "mock" else self.settings.llm_model),
            max_iterations=self.settings.max_iterations,
            faithfulness_threshold=self.settings.faithfulness_threshold)
        self._engine = self._select_engine()

    def _select_engine(self):
        """Use the LangGraph engine if requested and installed, else the built-in agent."""
        if self.settings.engine == "langgraph":
            try:
                from .graph import LangGraphRAG
                return LangGraphRAG(
                    llm=self.llm, tool=self.tool,
                    model=("mock" if self.settings.llm_provider == "mock" else self.settings.llm_model),
                    max_iterations=self.settings.max_iterations,
                    faithfulness_threshold=self.settings.faithfulness_threshold)
            except Exception:
                pass  # fall back gracefully
        return self.agent

    def enable_custom_knowledge(self, enabled: bool) -> bool:
        self.custom_enabled = bool(enabled)
        self.retriever.set_custom_enabled(self.custom_enabled)
        return self.custom_enabled

    def list_custom_knowledge(self) -> dict[str, Any]:
        return {
            "enabled": self.custom_enabled,
            "handbook": self.custom_knowledge["handbook"],
            "qa": self.custom_knowledge["qa"],
        }

    def remove_custom_knowledge(self) -> None:
        self.custom_knowledge = {"handbook": [], "qa": []}
        self._custom_handbook_texts = {}
        self.retriever.clear_custom()
        self.enable_custom_knowledge(False)

    def _reindex_custom_knowledge(self) -> None:
        self.retriever.clear_custom()
        for handbook in self.custom_knowledge["handbook"]:
            doc_id = handbook.get("id") or handbook.get("name") or "custom-doc"
            text = self._custom_handbook_texts.get(doc_id, handbook.get("text_preview", ""))
            if text:
                self.add_custom_document(text, doc_id.removeprefix("custom:"), strategy="section")
        for qa in self.custom_knowledge["qa"]:
            self.add_custom_qa(qa["question"], qa["answer"], qa["id"])

    # ingestion ---------------------------------------------------------------
    def add_document(self, text: str, doc_id: str = "doc", strategy: str = "section") -> int:
        chunker = CHUNKERS[strategy]
        chunks = chunker(text, doc_id=doc_id)
        self.retriever.index(chunks)
        return len(chunks)

    def add_documents(self, docs: list[Document], strategy: str = "section") -> int:
        return sum(self.add_document(d.text, d.doc_id, strategy) for d in docs)

    def add_custom_document(self, text: str, doc_id: str = "custom-doc", strategy: str = "section") -> int:
        if not text or not text.strip():
            raise ValueError("Custom handbook text cannot be empty.")
        normalized_id = doc_id if doc_id.startswith("custom:") else f"custom:{doc_id}"
        chunker = CHUNKERS[strategy]
        chunks = chunker(text, doc_id=normalized_id)
        for chunk in chunks:
            chunk.metadata["source"] = "custom"
            chunk.metadata["source_type"] = "handbook"
        self.retriever.index_custom(chunks)
        self._custom_handbook_texts[normalized_id] = text
        existing = next((item for item in self.custom_knowledge["handbook"] if item["id"] == normalized_id), None)
        payload = {
            "id": normalized_id,
            "name": doc_id,
            "source": "custom",
            "source_type": "handbook",
            "chunk_count": len(chunks),
            "text_preview": text[:180],
            "text": text,
        }
        if existing is not None:
            existing.clear()
            existing.update(payload)
        else:
            self.custom_knowledge["handbook"].append(payload)
        return len(chunks)

    def add_custom_qa(self, question: str, answer: str, qa_id: str | None = None) -> str:
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            raise ValueError("Custom Q&A requires both a question and an answer.")
        qa_key = qa_id or f"qa-{len(self.custom_knowledge['qa']) + 1}"
        chunk_id = f"custom:qa:{qa_key}"
        chunk = Chunk(
            id=chunk_id,
            text=f"Q: {question}\nA: {answer}",
            metadata={
                "source": "custom",
                "source_type": "qa",
                "question": question,
                "answer": answer,
                "qa_id": qa_key,
            },
        )
        self.retriever.index_custom([chunk])
        self.custom_knowledge["qa"].append({
            "id": qa_key,
            "question": question,
            "answer": answer,
            "source": "custom",
            "source_type": "qa",
        })
        return qa_key

    def update_custom_qa(self, qa_id: str, question: str, answer: str) -> dict[str, Any]:
        for item in self.custom_knowledge["qa"]:
            if item["id"] == qa_id:
                question = question.strip()
                answer = answer.strip()
                if not question or not answer:
                    raise ValueError("Custom Q&A requires both a question and an answer.")
                item["question"] = question
                item["answer"] = answer
                self._reindex_custom_knowledge()
                return item
        raise KeyError(f"QA item {qa_id!r} not found")

    def delete_custom_qa(self, qa_id: str) -> None:
        original = self.custom_knowledge["qa"]
        filtered = [item for item in original if item["id"] != qa_id]
        if len(filtered) == len(original):
            raise KeyError(f"QA item {qa_id!r} not found")
        self.custom_knowledge["qa"] = filtered
        self._reindex_custom_knowledge()

    # query -------------------------------------------------------------------
    def query(self, question: str) -> QueryResult:
        tracer = Tracer()
        try:
            question = check_input(question)
        except InputBlocked as exc:
            return QueryResult(question, f"Request blocked: {exc}", [], [], 0.0, 0,
                               blocked=True, trace=tracer.summary())
        with tracer.span("pipeline.query"):
            agent_answer = self._engine.run(question, tracer=tracer)
            chunks_by_id = {r.chunk.id: r for h in agent_answer.hops for r in h.contexts}
            contexts = [chunks_by_id[c] for c in agent_answer.citations if c in chunks_by_id]
            answer, _ = redact_pii(agent_answer.answer)
            faith = grounding_score(answer, [c.chunk.text for c in contexts])
        return QueryResult(question, answer, agent_answer.citations, contexts,
                           faith, agent_answer.iterations, trace=tracer.summary())
