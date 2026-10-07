"""Agentic RAG loop: plan -> multi-hop retrieve -> synthesize -> self-correct."""
from __future__ import annotations
from dataclasses import dataclass, field

from ..guardrails import grounding_score
from ..observability import Tracer
from ..providers.llm import LLMProvider
from ..retrieval.hybrid import Scored
from .planner import plan
from .tools import RetrievalTool

_SYSTEM = (
    "You are a conversational assistant with access to authoritative company policy. "
    "Use the retrieved knowledge to answer the user's actual question, not to repeat or copy the source text verbatim. "
    "Interpret how the policy applies to the user's situation, using the conversation history when relevant. "
    "State clearly what the handbook explicitly says, what can reasonably be inferred, and what remains uncertain. "
    "Never invent company rules or pretend a policy exists when it is silent. "
    "Prefer a direct answer in natural language using 'you' and 'your item' when appropriate."
)


@dataclass
class Hop:
    sub_question: str
    contexts: list[Scored]


@dataclass
class AgentAnswer:
    question: str
    answer: str
    hops: list[Hop]
    faithfulness: float
    iterations: int
    citations: list[str] = field(default_factory=list)


def _build_prompt(question: str, contexts: list[str], *, history: list[dict[str, str]] | None = None) -> str:
    joined = "\n---\n".join(contexts)
    history_block = ""
    if history:
        lines = [f"{item['role'].title()}: {item['content']}" for item in history if item.get("content")]
        history_block = "\n".join(lines) + "\n"
    return (
        f"{history_block}Context:\n{joined}\n\nLatest user question: {question}\n\n"
        "Answer in a natural conversational way. Explain how the policy applies to the user's real situation, "
        "and distinguish between what the policy explicitly says and what is uncertain. Do not quote the source passage verbatim.\nAnswer:"
    )


@dataclass
class AgenticRAG:
    llm: LLMProvider
    tool: RetrievalTool
    model: str = "mock"
    max_iterations: int = 2
    faithfulness_threshold: float = 0.6

    def run(self, question: str, tracer: Tracer | None = None) -> AgentAnswer:
        tracer = tracer or Tracer()
        best: AgentAnswer | None = None
        query = question
        with tracer.span("agent.run", question=question):
            for iteration in range(1, self.max_iterations + 1):
                hops: list[Hop] = []
                with tracer.span("plan"):
                    sub_questions = plan(query)
                seen: dict[str, Scored] = {}
                for sub in sub_questions:
                    with tracer.span("retrieve", sub_question=sub):
                        results = self.tool(sub)
                    hops.append(Hop(sub, results))
                    for r in results:
                        seen.setdefault(r.chunk.id, r)
                contexts = [r.chunk.text for r in seen.values()]
                with tracer.span("synthesize"):
                    result = self.llm.complete(_SYSTEM, _build_prompt(question, contexts, history=[]))
                    tracer.record_usage(self.model, result.usage)
                faith = grounding_score(result.text, contexts)
                candidate = AgentAnswer(question, result.text, hops, faith, iteration,
                                        citations=list(seen.keys()))
                if best is None or faith > best.faithfulness:
                    best = candidate
                if faith >= self.faithfulness_threshold:
                    break
                query = self._reformulate(question, contexts)  # self-correction
        assert best is not None
        return best

    @staticmethod
    def _reformulate(question: str, contexts: list[str]) -> str:
        expansion = " ".join(contexts)[:240]
        return f"{question} {expansion}"
