"""A from-scratch TF-IDF + cosine-similarity vector retriever.

No external ML/vector-search libraries (no faiss, no sentence-transformers,
no torch) are used -- just numpy. This keeps the project fully portable and
installable anywhere Python + numpy run, while still demonstrating the core
mechanics of a retrieval index: build a term-document matrix, weight it by
inverse document frequency, L2-normalize rows, and rank by dot product
(== cosine similarity for unit vectors).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

# A small, generic stopword list. Kept intentionally short so domain terms
# (e.g. product names) are never accidentally filtered out.
STOPWORDS = frozenset(
    """
    a an the and or but if while of to in on at for with by from as is are
    was were be been being this that these those it its it's into about
    than then so such not no nor can could will would shall should may
    might must do does did doing have has had having i you he she we they
    them him her his hers our ours your yours their theirs what which who
    whom out up down over under again further here there all any both each
    few more most other some own same too very just
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase, extract alphanumeric tokens, and drop stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class TfidfRetriever:
    """In-memory TF-IDF index over a list of Chunks."""

    def __init__(self) -> None:
        self.vocab: dict = {}  # token -> column index
        self.idf: Optional[np.ndarray] = None  # shape (V,)
        self.matrix: Optional[np.ndarray] = None  # shape (N, V), L2-normalized
        self.chunks: List[Chunk] = []

    @property
    def is_fitted(self) -> bool:
        return self.matrix is not None and len(self.chunks) > 0

    def fit(self, chunks: List[Chunk]) -> "TfidfRetriever":
        """Build the TF-IDF matrix from a list of Chunks."""
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")

        self.chunks = list(chunks)
        tokenized = [tokenize(c.text) for c in self.chunks]

        # Build vocabulary sorted alphabetically for determinism.
        vocab_set = set()
        for tokens in tokenized:
            vocab_set.update(tokens)
        self.vocab = {term: idx for idx, term in enumerate(sorted(vocab_set))}
        vocab_size = len(self.vocab)

        n_docs = len(self.chunks)
        counts = np.zeros((n_docs, vocab_size), dtype=np.float64)
        for row, tokens in enumerate(tokenized):
            for tok in tokens:
                counts[row, self.vocab[tok]] += 1.0

        # Smoothed IDF, same formula scikit-learn uses by default:
        # idf(t) = ln((1 + n) / (1 + df(t))) + 1
        doc_freq = (counts > 0).sum(axis=0)
        self.idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0

        tfidf = counts * self.idf  # broadcast over columns
        self.matrix = _l2_normalize_rows(tfidf)
        return self

    def _vectorize_query(self, text: str) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("Retriever has not been fitted yet.")
        vec = np.zeros(len(self.vocab), dtype=np.float64)
        for tok in tokenize(text):
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        vec = vec * self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def query(self, text: str, top_k: int = 4) -> List[RetrievalResult]:
        """Return the top_k chunks most similar to `text`, best first.

        Chunks with zero similarity (no vocabulary overlap with the query)
        are excluded even if that means returning fewer than top_k results.
        """
        if not self.is_fitted:
            raise RuntimeError("Retriever has not been fitted yet.")
        query_vec = self._vectorize_query(text)
        if not np.any(query_vec):
            return []

        scores = self.matrix @ query_vec  # cosine similarity, both sides unit-norm
        top_k = min(top_k, len(self.chunks))
        # argpartition for efficiency, then sort just the top slice.
        candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]

        results = []
        for idx in candidate_idx:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(RetrievalResult(chunk=self.chunks[idx], score=score))
        return results

    # -- persistence -------------------------------------------------

    def save(self, dir_path: str) -> None:
        """Persist the index to `dir_path` so it can be reloaded without
        re-reading and re-chunking the source documents."""
        os.makedirs(dir_path, exist_ok=True)
        np.save(os.path.join(dir_path, "matrix.npy"), self.matrix)
        np.save(os.path.join(dir_path, "idf.npy"), self.idf)
        with open(os.path.join(dir_path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)
        with open(os.path.join(dir_path, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": c.id,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "metadata": c.metadata,
                    }
                    for c in self.chunks
                ],
                f,
            )

    @classmethod
    def load(cls, dir_path: str) -> "TfidfRetriever":
        retriever = cls()
        retriever.matrix = np.load(os.path.join(dir_path, "matrix.npy"))
        retriever.idf = np.load(os.path.join(dir_path, "idf.npy"))
        with open(os.path.join(dir_path, "vocab.json"), "r", encoding="utf-8") as f:
            retriever.vocab = json.load(f)
        with open(os.path.join(dir_path, "chunks.json"), "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        retriever.chunks = [
            Chunk(
                id=rc["id"],
                source=rc["source"],
                chunk_index=rc["chunk_index"],
                text=rc["text"],
                metadata=rc.get("metadata", {}),
            )
            for rc in raw_chunks
        ]
        return retriever


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for all-zero rows
    return matrix / norms
