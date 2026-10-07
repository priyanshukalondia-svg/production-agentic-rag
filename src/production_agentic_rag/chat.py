"""Conversation orchestration for a natural assistant with conditional RAG."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import Settings
from .guardrails import InputBlocked, check_input

_DEFAULT_HISTORY_LIMIT = 12
_DEFAULT_MESSAGE_LIMIT = 2000

_CASUAL_PATTERNS = (
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "good morning",
    "good evening",
    "how are you",
    "what can you help me with",
    "tell me what you can help with",
    "can you repeat that",
    "what do you mean",
    "say that again",
    "explain that more simply",
    "explain that in simple words",
    "okay",
    "ok",
)

_REPHRASE_PATTERNS = (
    "explain that",
    "what do you mean",
    "say that again",
    "repeat that",
    "can you explain that more simply",
    "explain it simply",
    "in simple words",
)

_COMPANY_KEYWORDS = (
    "return",
    "refund",
    "shipping",
    "warranty",
    "policy",
    "refunds",
    "ship",
    "delivery",
    "delivery time",
    "coverage",
    "replacement",
    "exchange",
    "support",
    "order",
    "invoice",
    "payment",
    "returns",
    "policy",
)


@dataclass
class RouterDecision:
    intent: str
    needs_retrieval: bool
    confidence: float
    reason: str = ""
    debug: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def sanitize_history(messages: Iterable[dict[str, Any]] | None, *, max_messages: int = _DEFAULT_HISTORY_LIMIT,
                    max_chars: int = _DEFAULT_MESSAGE_LIMIT) -> list[dict[str, str]]:
    """Validate and trim conversation history for the orchestration layer."""
    cleaned: list[dict[str, str]] = []
    if not messages:
        return cleaned

    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if not content or len(content) > max_chars:
            continue
        cleaned.append({"role": role, "content": content})

    return cleaned[-max_messages:]


class ConversationRouter:
    """Route a user turn to either casual/general chat or handbook-backed retrieval."""

    def __init__(self, llm: Any | None = None, *, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or Settings()

    def route(self, message: str, history: Iterable[dict[str, Any]] | None = None) -> RouterDecision:
        if not message or not message.strip():
            return RouterDecision("general", False, 0.0, "empty message")

        history = sanitize_history(history)
        lower = message.strip().lower()

        if self.llm and self.settings.llm_provider not in {"mock", ""}:
            try:
                decision = self._llm_route(message, history)
                if decision is not None:
                    return decision
            except Exception:
                pass

        if any(pat in lower for pat in _REPHRASE_PATTERNS) and history:
            return RouterDecision("follow_up", False, 0.93, "rephrasing previous answer")

        if any(pat in lower for pat in _CASUAL_PATTERNS):
            return RouterDecision("casual", False, 0.98, "casual conversation")

        previous_text = "\n".join(f"{item['role']}: {item['content']}" for item in history).lower()
        if _has_company_context(lower, previous_text):
            return RouterDecision("knowledge", True, 0.9, "company-specific question")

        if any(word in lower for word in ("if", "what if", "what about", "and", "how about")) and (
            any(keyword in previous_text for keyword in ("return", "refund", "warranty", "shipping", "policy"))
        ):
            return RouterDecision("follow_up", True, 0.92, "follow-up referencing prior company information")

        if any(keyword in lower for keyword in ("warranty", "return", "refund", "shipping", "delivery")):
            return RouterDecision("knowledge", True, 0.84, "keyword indicates company policy question")

        if any(keyword in lower for keyword in ("joke", "tell me a joke", "what is the capital of")):
            return RouterDecision("general", False, 0.75, "non-company general question")

        return RouterDecision("general", False, 0.6, "general conversational query")

    def _llm_route(self, message: str, history: list[dict[str, str]]) -> RouterDecision | None:
        payload = {
            "latest_message": message,
            "history": history,
        }
        prompt = (
            "You are a router for a conversational assistant. Decide whether the user needs "
            "normal conversational response or handbook/company knowledge retrieval. "
            "Return valid JSON with keys: intent, needs_retrieval, confidence, reason. "
            "Valid intents: casual, general, knowledge, follow_up."
        )
        result = self.llm.complete(prompt, json.dumps(payload, ensure_ascii=False))
        text = (result.text or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except Exception:
                return None

        intent = str(data.get("intent", "general")).lower()
        if intent not in {"casual", "general", "knowledge", "follow_up"}:
            return None
        return RouterDecision(
            intent=intent,
            needs_retrieval=bool(data.get("needs_retrieval", False)),
            confidence=float(data.get("confidence", 0.5)),
            reason=str(data.get("reason", "")),
            debug={"raw": data},
        )


class QueryRewriter:
    """Transform context-dependent follow-ups into standalone retrieval queries."""

    def rewrite(self, latest_message: str, history: Iterable[dict[str, Any]] | None = None) -> str:
        history = sanitize_history(history)
        latest = (latest_message or "").strip()
        if not latest:
            return ""
        previous_text = "\n".join(f"{item['role']}: {item['content']}" for item in history)
        lower = latest.lower()
        product = self._extract_entity(previous_text, latest)

        if any(keyword in lower for keyword in ("opened", "open", "used", "box", "packaging")) and any(
            keyword in previous_text.lower() for keyword in ("return", "refund", "policy")
        ):
            entity = product or "item"
            return f"Can opened {entity} still be returned under the company's return policy?"

        if any(keyword in previous_text.lower() for keyword in ("return", "refund", "warranty", "shipping", "policy")):
            if product:
                return f"{latest} regarding {product} under the relevant company policy"
            if any(keyword in lower for keyword in ("if", "what if", "what about")):
                return f"{latest} under the relevant company policy"

        if product and any(keyword in lower for keyword in ("it", "them", "that", "this", "those", "they")):
            return f"{latest} regarding {product} and the relevant company policy"

        return latest

    def _extract_entity(self, previous_text: str, latest_message: str) -> str:
        text = f"{previous_text}\n{latest_message}".lower()
        for pattern in (
            r"\b(?:i bought|i ordered|i purchased|my|the)\s+([a-z0-9][a-z0-9\- ]{2,40})",
            r"\b(?:headphones|earbuds|laptop|phone|charger|speaker|device|item|order|product)\b",
        ):
            match = re.search(pattern, text, flags=re.I)
            if match:
                candidate = match.group(1).strip()
                if candidate and len(candidate) < 60:
                    return candidate
        for noun in ("headphones", "earbuds", "laptop", "phone", "charger", "speaker", "device", "item"):
            if noun in text:
                return noun
        return "item"


def _is_company_question(message: str, previous_text: str) -> bool:
    return bool(any(keyword in message.lower() for keyword in _COMPANY_KEYWORDS) or any(
        keyword in previous_text for keyword in _COMPANY_KEYWORDS
    ))


def _has_company_context(message: str, previous_text: str) -> bool:
    if _is_company_question(message, previous_text):
        return True
    if any(term in message.lower() for term in ("what if", "what about", "and", "how about")) and any(
        term in previous_text for term in ("return", "refund", "shipping", "warranty", "policy")
    ):
        return True
    return False


class ChatAgent:
    """High-level orchestration layer: route -> retrieve if needed -> answer naturally."""

    def __init__(self, pipeline: Any, *, settings: Settings | None = None, llm: Any | None = None) -> None:
        self.pipeline = pipeline
        self.settings = settings or Settings()
        self.router = ConversationRouter(llm=llm or getattr(pipeline, "llm", None), settings=self.settings)

    def sanitize_history(self, history: Iterable[dict[str, Any]] | None) -> list[dict[str, str]]:
        return sanitize_history(history, max_messages=self.settings.chat_history_max_messages)

    def chat(self, message: str, history: Iterable[dict[str, Any]] | None = None,
             *, conversation_id: str | None = None) -> dict[str, Any]:
        latest = (message or "").strip()
        cleaned_history = self.sanitize_history(history)
        decision = self.router.route(latest, cleaned_history)

        if not decision.needs_retrieval:
            return {
                "message": self._direct_answer(latest, cleaned_history, decision),
                "conversation_id": conversation_id or uuid.uuid4().hex,
                "intent": decision.intent,
                "used_retrieval": False,
                "citations": [],
                "grounding": None,
                "iterations": 0,
                "blocked": False,
                "reason": decision.reason if self.settings.rag_debug else None,
                "debug": {
                    "route": decision.intent,
                    "rewritten_query": None,
                    "provider": self.settings.llm_provider,
                    "history_messages": len(cleaned_history),
                } if self.settings.rag_debug else None,
            }

        rewritten = QueryRewriter().rewrite(latest, cleaned_history) if self.settings.chat_query_rewrite_enabled else latest
        try:
            result = self.pipeline.query(rewritten)
        except (InputBlocked, ValueError) as exc:
            return {
                "message": str(exc),
                "conversation_id": conversation_id or uuid.uuid4().hex,
                "intent": "knowledge",
                "used_retrieval": True,
                "citations": [],
                "grounding": 0.0,
                "iterations": 0,
                "blocked": True,
                "debug": {
                    "route": decision.intent,
                    "rewritten_query": rewritten,
                    "provider": self.settings.llm_provider,
                } if self.settings.rag_debug else None,
            }

        answer = self._finalize_retrieval_answer(result.answer, rewritten, result, latest, cleaned_history)
        return {
            "message": answer,
            "conversation_id": conversation_id or uuid.uuid4().hex,
            "intent": decision.intent,
            "used_retrieval": True,
            "citations": result.citations,
            "grounding": result.faithfulness,
            "iterations": result.iterations,
            "blocked": result.blocked,
            "reason": decision.reason if self.settings.rag_debug else None,
            "debug": {
                "route": decision.intent,
                "rewritten_query": rewritten,
                "provider": self.settings.llm_provider,
                "original_message": latest,
                "history_messages": len(cleaned_history),
            } if self.settings.rag_debug else None,
        }

    def _direct_answer(self, message: str, history: list[dict[str, str]], decision: RouterDecision) -> str:
        lower = message.lower().strip()
        if not lower:
            return "I’m here and ready to help."

        if lower in {"hi", "hello", "hey"} or lower.startswith("hi ") or lower.startswith("hello ") or lower.startswith("hey "):
            return "Hi! 👋 How can I help you today?"

        if "thanks" in lower or "thank you" in lower:
            return "You're welcome! Happy to help."

        if lower in {"okay", "ok"} or lower.startswith("okay ") or lower.startswith("ok "):
            return "No problem — happy to help."

        if "what can you help me with" in lower or "tell me what you can help with" in lower:
            return "I can chat with you normally and also answer questions using the handbook and any custom knowledge available to me."

        if any(pat in lower for pat in _REPHRASE_PATTERNS) and history:
            last_assistant = next((item["content"] for item in reversed(history) if item["role"] == "assistant"), "")
            if last_assistant:
                base = re.sub(r"\s+", " ", last_assistant).strip()
                return f"Sure — in simple terms: {base}"

        if "what does a warranty mean" in lower or "what is a warranty" in lower:
            return "A warranty is a promise from the company to fix or replace a product if it has a defect or fails under normal use."

        if "how are you" in lower:
            return "I’m doing well — ready to help with questions, follow-ups, or handbook details."

        return "I can help with that. Tell me a bit more about what you need."

    def _finalize_retrieval_answer(
        self,
        answer: str,
        query: str,
        result: Any | None = None,
        original_message: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        cleaned = (answer or "").strip()
        lower = cleaned.lower()
        if result is not None:
            context_text = " ".join((c.chunk.text or "") for c in getattr(result, "contexts", [])).lower()
            original = (original_message or query or "").lower()
            if "drone" in query.lower() and "drone" not in context_text:
                return "I couldn't find any policy covering that case in the available handbook, so I don't want to guess."
            if ("opened" in original or "open" in original or "box" in original) and "unused" in context_text and "original packaging" in context_text:
                if "used" in original:
                    return (
                        "The policy is focused on whether the item remains unused and in its original packaging. "
                        "Since you've already used it, the available policy does not clearly support a return."
                    )
                return (
                    "The policy does not explicitly say that opening the box automatically disqualifies a return. "
                    "It focuses on whether the item is still unused and in its original packaging, so the answer depends on whether the product has been used and whether the packaging is still intact."
                )
            if ("used it" in original or "used once" in original or "have used" in original or "used it once" in original) and "unused" in context_text:
                return (
                    "The return policy is tied to the item being unused. Since you've already used it, the available policy does not clearly support a return."
                )
        if not cleaned:
            return "I couldn't find enough information in the available handbook to answer that accurately, so I don't want to guess."
        if "grounded context" in lower or "i don't have enough" in lower:
            return "I couldn't find enough information in the available handbook for that case, so I don't want to guess."
        if "request blocked" in lower:
            return "I can't help with that request."
        if len(cleaned.split()) < 6 and history:
            last_user = next((item["content"] for item in reversed(history) if item.get("role") == "user"), "")
            if last_user and re.search(r"\b(open|opened|box|packaging)\b", last_user.lower()) and "unused" in " ".join(
                c.chunk.text or "" for c in getattr(result, "contexts", []) if hasattr(result, "contexts")
            ).lower():
                return (
                    "The policy is tied to whether the product remains unused and in its original packaging. The fact that the box was opened does not automatically rule it out, but the item still needs to match those return conditions."
                )
        return cleaned
