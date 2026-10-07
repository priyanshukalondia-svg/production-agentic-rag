"""production-agentic-rag — a provider-agnostic, production-grade agentic RAG system."""
from .chat import ChatAgent, ConversationRouter, QueryRewriter, RouterDecision, sanitize_history
from .config import Settings
from .pipeline import QueryResult, RAGPipeline

__all__ = [
    "RAGPipeline",
    "QueryResult",
    "Settings",
    "ChatAgent",
    "ConversationRouter",
    "QueryRewriter",
    "RouterDecision",
    "sanitize_history",
]
__version__ = "0.1.0"
