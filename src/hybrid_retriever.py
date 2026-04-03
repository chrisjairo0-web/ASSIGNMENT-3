from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .doc_ingestion import build_or_load_faiss


class HybridRetriever:
    """Hybrid BM25 + semantic retrieval with RRF fusion and optional reranking."""

    def __init__(
        self,
        chunks: List[Dict],
        source_id: str,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_dir: str = ".cache",
    ) -> None:
        self.chunks = chunks
        self.source_id = source_id
        self.texts = [c["text"] for c in chunks]
        self.metadata = [c["metadata"] for c in chunks]

        self.bm25_tokens = [self._tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.bm25_tokens)

        self.index, self.embeddings, self.embedding_model = build_or_load_faiss(
            chunks=self.chunks,
            source_id=self.source_id,
            embedding_model_name=embedding_model_name,
            cache_dir=cache_dir,
        )

        self.reranker = None
        try:
            self.reranker = CrossEncoder(reranker_model_name)
        except Exception:
            self.reranker = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.lower() for token in text.split() if token.strip()]

    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        query_emb = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        scores, indices = self.index.search(query_emb, top_k)
        result: List[Tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                result.append((int(idx), float(score)))
        return result

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        ranked_idx = np.argsort(bm25_scores)[::-1][:top_k]
        return [(int(idx), float(bm25_scores[idx])) for idx in ranked_idx]

    @staticmethod
    def _rrf_rank(
        semantic_hits: List[Tuple[int, float]],
        lexical_hits: List[Tuple[int, float]],
        k: int = 60,
    ) -> Dict[int, float]:
        fused: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(semantic_hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)

        for rank, (idx, _) in enumerate(lexical_hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)

        return fused

    def _rerank(self, query: str, candidate_indices: List[int]) -> List[Tuple[int, float]]:
        if not self.reranker:
            return [(idx, 0.0) for idx in candidate_indices]

        pairs = [(query, self.texts[idx]) for idx in candidate_indices]
        scores = self.reranker.predict(pairs)
        return list(zip(candidate_indices, [float(s) for s in scores]))

    def retrieve(
        self,
        query: str,
        top_k_semantic: int = 8,
        top_k_bm25: int = 8,
        top_n: int = 5,
    ) -> List[Dict]:
        semantic_hits = self._semantic_search(query, top_k=top_k_semantic)
        lexical_hits = self._bm25_search(query, top_k=top_k_bm25)

        fused = self._rrf_rank(semantic_hits, lexical_hits)
        fused_sorted = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        candidate_indices = [idx for idx, _ in fused_sorted[: max(top_n * 2, top_n)]]

        reranked = self._rerank(query, candidate_indices)
        if self.reranker:
            reranked_sorted = sorted(reranked, key=lambda item: item[1], reverse=True)
        else:
            reranked_sorted = [(idx, fused[idx]) for idx in candidate_indices]
            reranked_sorted.sort(key=lambda item: item[1], reverse=True)

        semantic_lookup = {idx: score for idx, score in semantic_hits}
        lexical_lookup = {idx: score for idx, score in lexical_hits}

        output: List[Dict] = []
        for rank, (idx, rerank_score) in enumerate(reranked_sorted[:top_n], start=1):
            meta = self.metadata[idx]
            fused_score = fused.get(idx, 0.0)
            confidence = float(max(0.0, min(1.0, (rerank_score + 10.0) / 20.0))) if self.reranker else float(
                max(0.0, min(1.0, fused_score * 30.0))
            )

            output.append(
                {
                    "rank": rank,
                    "score": round(rerank_score if self.reranker else fused_score, 4),
                    "confidence": round(confidence, 4),
                    "fused_score": round(fused_score, 4),
                    "semantic_score": round(semantic_lookup.get(idx, 0.0), 4),
                    "bm25_score": round(lexical_lookup.get(idx, 0.0), 4),
                    "text": self.texts[idx],
                    "snippet": self.texts[idx][:320].replace("\n", " "),
                    "metadata": meta,
                    "page_number": meta.get("page_start"),
                    "section": meta.get("section") or meta.get("section_title") or "N/A",
                }
            )

        return output
