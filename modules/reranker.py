"""Cross-encoder based re-ranking for the custom RAG stack."""

from collections import defaultdict
from typing import List, Sequence

import torch
from sentence_transformers import CrossEncoder

from modules.models import RetrievedDocument


class CrossEncoderReranker:
    """A lightweight cross-encoder reranker implemented with PyTorch-backed transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        try:
            self.model = CrossEncoder(model_name, device=self.device)
        except Exception:
            self.model = None

    def rerank(self, docs: Sequence[RetrievedDocument], query: str, k: int = 5) -> List[RetrievedDocument]:
        """Re-rank documents using cross-encoder scores when available, else fallback to score fusion."""
        if not docs:
            return []

        if self.model is not None:
            pairs = [(query, doc.text) for doc in docs]
            scores = self.model.predict(pairs, convert_to_numpy=True)
            ranked = sorted(zip(scores, docs), key=lambda tup: tup[0], reverse=True)
            reranked = []
            for index, (_, doc) in enumerate(ranked[:k], start=1):
                doc.rank = index
                reranked.append(doc)
            return reranked

        return self._fallback_rerank(docs, k)

    def _fallback_rerank(self, docs: Sequence[RetrievedDocument], k: int) -> List[RetrievedDocument]:
        """Fallback to hybrid-score ordering with a diversity bonus."""
        source_counts = defaultdict(int)
        scored_docs = []
        for doc in docs:
            diversity_bonus = 1.0 / (1 + source_counts[doc.source])
            final_score = (0.7 * doc.combined_score) + (0.3 * diversity_bonus * doc.combined_score)
            scored_docs.append((final_score, doc))
            source_counts[doc.source] += 1

        scored_docs.sort(reverse=True, key=lambda item: item[0])
        reranked = []
        for index, (_, doc) in enumerate(scored_docs[:k], start=1):
            doc.rank = index
            reranked.append(doc)
        return reranked


class ReRanker(CrossEncoderReranker):
    """Backward-compatible alias retained for the existing retrieval engine."""

    def rerank(self, docs: Sequence[RetrievedDocument], k: int = 5, query: str | None = None) -> List[RetrievedDocument]:
        if query is None:
            return super()._fallback_rerank(docs, k)
        return super().rerank(docs, query, k=k)

