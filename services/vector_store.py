from typing import List, Dict, Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import EMBEDDING_DIM, TOP_K_RETRIEVAL, RRF_K
from services.embeddings import EmbeddingService


class HybridVectorStore:
    """FAISS (dense) + BM25 (sparse) with Reciprocal Rank Fusion."""

    def __init__(self):
        self._emb = EmbeddingService.get_instance()
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None


    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        vecs = self._emb.encode(chunks)
        self._index.add(vecs)
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        self._rebuild_bm25()

    def hybrid_search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        k = min(k, len(self.chunks))

        # --- Dense search ---
        q_vec = self._emb.encode_single(query).reshape(1, -1)
        _, faiss_idxs = self._index.search(q_vec, k)
        faiss_ranks: Dict[int, int] = {
            int(idx): rank
            for rank, idx in enumerate(faiss_idxs[0])
            if idx >= 0
        }

        # --- Sparse search (BM25) ---
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:k]
        bm25_ranks: Dict[int, int] = {int(idx): rank for rank, idx in enumerate(bm25_top)}

        # --- RRF fusion ---
        all_idxs = set(faiss_ranks) | set(bm25_ranks)
        rrf: Dict[int, float] = {}
        for idx in all_idxs:
            score = 0.0
            if idx in faiss_ranks:
                score += 1.0 / (RRF_K + faiss_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (RRF_K + bm25_ranks[idx])
            rrf[idx] = score

        sorted_idxs = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:k]
        return [
            {
                "chunk": self.chunks[i],
                "metadata": self.metadata[i],
                "score": rrf[i],
                "faiss_rank": faiss_ranks.get(i),
                "bm25_rank": bm25_ranks.get(i),
            }
            for i in sorted_idxs
        ]

    def reset(self) -> None:
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.chunks = []
        self.metadata = []
        self._bm25 = None

    def _rebuild_bm25(self) -> None:
        tokenized = [c.lower().split() for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)