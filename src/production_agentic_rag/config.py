"""Runtime configuration (env-overridable)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass
class Settings:
    # providers
    llm_provider: str = field(default_factory=lambda: _env("RAG_LLM_PROVIDER", "mock"))
    embed_provider: str = field(default_factory=lambda: _env("RAG_EMBED_PROVIDER", "hashing"))
    llm_model: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", "gpt-4o-mini"))
    embed_model: str = field(default_factory=lambda: _env("RAG_EMBED_MODEL", "text-embedding-3-small"))
    # retrieval
    top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "4")))
    candidate_k: int = field(default_factory=lambda: int(_env("RAG_CANDIDATE_K", "12")))
    chunk_size: int = field(default_factory=lambda: int(_env("RAG_CHUNK_SIZE", "120")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "24")))
    rerank: bool = field(default_factory=lambda: _env("RAG_RERANK", "1") == "1")
    # agent
    max_iterations: int = field(default_factory=lambda: int(_env("RAG_MAX_ITERS", "2")))
    engine: str = field(default_factory=lambda: _env("RAG_ENGINE", "agent"))  # agent | langgraph
    faithfulness_threshold: float = field(default_factory=lambda: float(_env("RAG_FAITHFULNESS", "0.6")))
    # chat orchestration
    chat_history_max_messages: int = field(default_factory=lambda: int(_env("CHAT_HISTORY_MAX_MESSAGES", "10")))
    chat_router_enabled: bool = field(default_factory=lambda: _env("CHAT_ROUTER_ENABLED", "1") == "1")
    chat_query_rewrite_enabled: bool = field(default_factory=lambda: _env("CHAT_QUERY_REWRITE_ENABLED", "1") == "1")
    chat_general_knowledge_enabled: bool = field(default_factory=lambda: _env("CHAT_GENERAL_KNOWLEDGE_ENABLED", "1") == "1")
    chat_summarization_enabled: bool = field(default_factory=lambda: _env("CHAT_SUMMARIZATION_ENABLED", "0") == "1")
    rag_debug: bool = field(default_factory=lambda: _env("RAG_DEBUG", "0") == "1")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()
