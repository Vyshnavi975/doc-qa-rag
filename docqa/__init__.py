"""docqa: a lightweight, dependency-minimal Retrieval-Augmented Generation (RAG)
toolkit for answering questions over a folder of local text/PDF documents.

Modules:
    loaders   - read .txt/.md/.pdf files from a folder into raw document text
    chunking  - split document text into overlapping, retrievable chunks
    retriever - from-scratch TF-IDF + cosine-similarity vector index
    llm       - optional answer synthesis via Anthropic/OpenAI, with a
                no-API-key "demo mode" extractive fallback
    cli       - command line entry point tying the above together
"""

__version__ = "1.0.0"
