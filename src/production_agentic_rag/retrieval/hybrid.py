"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion, then reranked."""
from __future__ import annotations
from dataclasses import dataclass

from ..ingestion.chunking import Chunk
from ..providers.embeddings import Embedder
from ..text import tokenize
from .bm25 import BM25
from .vectorstore import VectorStore


@dataclass
class Scored:
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, embedder: Embedder | None = None, rrf_k: int = 60,
                 *, custom_enabled: bool = False) -> None:
        self.bm25 = BM25()
        self.vectors = VectorStore(embedder)
        self._custom_bm25 = BM25()
        self._custom_vectors = VectorStore(embedder)
        self.rrf_k = rrf_k
        self.custom_enabled = custom_enabled
        self._chunks: dict[str, Chunk] = {}
        self._custom_chunks: dict[str, Chunk] = {}

    def set_custom_enabled(self, enabled: bool) -> None:
        self.custom_enabled = bool(enabled)

    def index(self, chunks: list[Chunk]) -> None:
        self._index_source(chunks, self.bm25, self.vectors)

    def index_custom(self, chunks: list[Chunk]) -> None:
        self._index_source(chunks, self._custom_bm25, self._custom_vectors)

    def clear_custom(self) -> None:
        self._custom_bm25 = BM25()
        self._custom_vectors = VectorStore(self.vectors.embedder)
        self._custom_chunks = {}

    def _index_source(self, chunks: list[Chunk], bm25: BM25, vectors: VectorStore) -> None:
        for c in chunks:
            bm25.index(c.id, c.text)
            vectors.add(c)
            self._chunks[c.id] = c
            if c.id.startswith("custom:") or c.metadata.get("source") == "custom":
                self._custom_chunks[c.id] = c

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def custom_size(self) -> int:
        return len(self._custom_chunks)

    def _rrf(self, rankings: list[list[str]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return fused

    def _retrieve_source(self, bm25: BM25, vectors: VectorStore, query: str, k: int = 4,
                         *, candidate_k: int = 12, rerank: bool = True) -> list[Scored]:
        lexical = [cid for cid, _ in bm25.search(query, candidate_k)]
        dense = [cid for cid, _ in vectors.search(query, candidate_k)]
        fused = self._rrf([lexical, dense])
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        results = [Scored(self._chunks.get(cid, vectors.chunk(cid)), score)
                   for cid, score in ranked if cid in self._chunks or cid in vectors._chunks]
        if rerank:
            results = self._rerank(query, results)
        return results[:k]

    def retrieve(self, query: str, k: int = 4, *, candidate_k: int = 12,
                 rerank: bool = True) -> list[Scored]:
        default_results = self._retrieve_source(self.bm25, self.vectors, query,
                                               k=candidate_k, candidate_k=candidate_k,
                                               rerank=rerank)
        if not self.custom_enabled or not self._custom_chunks:
            return default_results[:k]

        custom_results = self._retrieve_source(self._custom_bm25, self._custom_vectors, query,
                                              k=candidate_k, candidate_k=candidate_k,
                                              rerank=rerank)
        merged: dict[str, Scored] = {}
        for result in custom_results:
            merged[result.chunk.id] = Scored(result.chunk, result.score + 1.5)
        for result in default_results:
            current = merged.get(result.chunk.id)
            if current is None:
                merged[result.chunk.id] = Scored(result.chunk, result.score)
            else:
                current.score = max(current.score, result.score)

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        if rerank:
            ranked = self._rerank(query, ranked)
        return ranked[:k]

    def _rerank(self, query: str, results: list[Scored]) -> list[Scored]:
        q = set(tokenize(query))
        for r in results:
            overlap = len(q & set(tokenize(r.chunk.text)))
            phrase = 0.5 if query.lower() in r.chunk.text.lower() else 0.0
            r.score = r.score + 0.2 * overlap + phrase
        results.sort(key=lambda s: s.score, reverse=True)
        return results
