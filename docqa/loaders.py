"""Document loading: read .txt/.md/.pdf files from a folder into plain text."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


@dataclass
class Document:
    """A single loaded source document."""

    source: str  # filename, used for citations
    path: str  # full path on disk
    text: str  # extracted plain text


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_pdf_file(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only if pypdf missing
        raise RuntimeError(
            "pypdf is required to read PDF files. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            # Some PDF pages fail to extract cleanly; skip rather than crash the run.
            pages_text.append("")
    return "\n".join(pages_text)


def load_document(path: str) -> Document:
    """Load a single supported file into a Document."""
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTENSIONS:
        text = _read_text_file(path)
    elif ext in PDF_EXTENSIONS:
        text = _read_pdf_file(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return Document(source=os.path.basename(path), path=path, text=text)


def load_documents(folder: str) -> List[Document]:
    """Load every supported file directly inside `folder` (non-recursive).

    Raises FileNotFoundError if the folder does not exist, and ValueError if
    no supported documents are found inside it.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Documents folder not found: {folder}")

    documents: List[Document] = []
    for name in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, name)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        doc = load_document(full_path)
        if doc.text.strip():
            documents.append(doc)

    if not documents:
        raise ValueError(
            f"No supported documents (.txt, .md, .pdf) with extractable text "
            f"were found in: {folder}"
        )
    return documents
