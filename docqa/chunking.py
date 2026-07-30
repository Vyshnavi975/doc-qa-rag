"""Splitting document text into overlapping, retrievable chunks.

Chunking is word-based (not character-based) so chunk boundaries respect word
integrity, and overlapping windows help avoid cutting a relevant sentence in
half between two chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .loaders import Document

DEFAULT_CHUNK_SIZE = 150  # words per chunk
DEFAULT_CHUNK_OVERLAP = 30  # words shared between consecutive chunks

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    """A chunk of text extracted from one source document."""

    id: str
    source: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split `text` into a list of overlapping word-window chunks.

    `chunk_size` is the target number of words per chunk and `overlap` is how
    many trailing words of one chunk are repeated at the start of the next,
    which helps retrieval when a relevant idea straddles a chunk boundary.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    words = normalized.split(" ")
    if len(words) <= chunk_size:
        return [normalized]

    stride = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += stride
    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk a list of loaded Documents into a flat list of Chunk objects."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    id=f"{doc.source}::chunk{i}",
                    source=doc.source,
                    chunk_index=i,
                    text=piece,
                    metadata={"path": doc.path},
                )
            )
    return all_chunks
