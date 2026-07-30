"""Command-line interface for docqa.

Usage:
    # Build a persisted index once, then ask many questions against it:
    python -m docqa.cli build --docs sample_docs --index .docqa_index
    python -m docqa.cli ask --index .docqa_index --question "What products does the company sell?"

    # Or skip persistence and just ask directly against a folder each time
    # (the index is built in memory and discarded afterwards):
    python -m docqa.cli ask --docs sample_docs --question "Who founded the company?"
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import chunking, llm
from .loaders import load_documents
from .retriever import RetrievalResult, TfidfRetriever


def _build_retriever(
    docs_folder: str, chunk_size: int, overlap: int, quiet: bool = False
) -> TfidfRetriever:
    documents = load_documents(docs_folder)
    chunks = chunking.chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not quiet:
        print(
            f"Loaded {len(documents)} document(s), split into {len(chunks)} chunk(s) "
            f"(chunk_size={chunk_size} words, overlap={overlap} words).",
            file=sys.stderr,
        )
    retriever = TfidfRetriever()
    retriever.fit(chunks)
    return retriever


def cmd_build(args: argparse.Namespace) -> int:
    retriever = _build_retriever(args.docs, args.chunk_size, args.overlap)
    retriever.save(args.index)
    print(f"Index built from '{args.docs}' and saved to '{args.index}'.")
    print(f"  documents indexed as {len(retriever.chunks)} chunk(s)")
    print(f"  vocabulary size: {len(retriever.vocab)} terms")
    return 0


def _get_retriever_for_ask(args: argparse.Namespace) -> TfidfRetriever:
    if args.index:
        try:
            return TfidfRetriever.load(args.index)
        except FileNotFoundError:
            if not args.docs:
                raise
            print(
                f"No index found at '{args.index}'; building one from "
                f"'{args.docs}' instead (not saved -- run 'build' to persist it).",
                file=sys.stderr,
            )
    if not args.docs:
        raise SystemExit(
            "error: --docs is required when --index does not point to an existing index"
        )
    return _build_retriever(args.docs, args.chunk_size, args.overlap)


def format_answer_block(
    question: str, answer: str, results: List[RetrievalResult], mode: str
) -> str:
    lines = [
        "=" * 72,
        f"Q: {question}",
        "-" * 72,
        answer,
    ]
    if mode == "generated" and results:
        lines.append("")
        lines.append("Sources consulted:")
        seen = []
        for r in results:
            if r.chunk.source not in seen:
                seen.append(r.chunk.source)
        for s in seen:
            lines.append(f"  - {s}")
    lines.append("=" * 72)
    return "\n".join(lines)


def cmd_ask(args: argparse.Namespace) -> int:
    retriever = _get_retriever_for_ask(args)
    results = retriever.query(args.question, top_k=args.top_k)

    provider = None if args.no_llm else llm.detect_provider()

    if provider is not None:
        try:
            answer = llm.generate_answer(args.question, results, provider=provider)
            mode = "generated"
        except llm.LLMError as exc:
            print(f"warning: LLM generation failed ({exc}); falling back to demo mode.",
                  file=sys.stderr)
            answer = llm.demo_answer(args.question, results)
            mode = "demo"
    else:
        answer = llm.demo_answer(args.question, results)
        mode = "demo"

    print(format_answer_block(args.question, answer, results, mode))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docqa",
        description="Answer questions over a folder of local text/PDF documents using "
        "retrieval-augmented generation (RAG). Uses an LLM to synthesize an answer if "
        "ANTHROPIC_API_KEY or OPENAI_API_KEY is set; otherwise falls back to demo mode "
        "and returns the most relevant retrieved passages directly.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser(
        "build", help="Build and persist a vector index from a folder of documents."
    )
    build_p.add_argument("--docs", required=True, help="Folder of .txt/.md/.pdf documents.")
    build_p.add_argument("--index", required=True, help="Output directory for the saved index.")
    build_p.add_argument("--chunk-size", type=int, default=chunking.DEFAULT_CHUNK_SIZE,
                          help=f"Words per chunk (default: {chunking.DEFAULT_CHUNK_SIZE}).")
    build_p.add_argument("--overlap", type=int, default=chunking.DEFAULT_CHUNK_OVERLAP,
                          help=f"Overlap words between chunks (default: {chunking.DEFAULT_CHUNK_OVERLAP}).")
    build_p.set_defaults(func=cmd_build)

    ask_p = subparsers.add_parser(
        "ask", help="Ask a question, using a saved index and/or a documents folder."
    )
    ask_p.add_argument("--question", "-q", required=True, help="The natural-language question.")
    ask_p.add_argument("--index", default=None, help="Path to a previously built index directory.")
    ask_p.add_argument("--docs", default=None,
                        help="Folder of documents to build an in-memory index from if "
                        "--index is not given, or is given but does not exist yet.")
    ask_p.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4).")
    ask_p.add_argument("--chunk-size", type=int, default=chunking.DEFAULT_CHUNK_SIZE,
                        help=f"Words per chunk when building in-memory (default: {chunking.DEFAULT_CHUNK_SIZE}).")
    ask_p.add_argument("--overlap", type=int, default=chunking.DEFAULT_CHUNK_OVERLAP,
                        help=f"Overlap words between chunks when building in-memory "
                        f"(default: {chunking.DEFAULT_CHUNK_OVERLAP}).")
    ask_p.add_argument("--no-llm", action="store_true",
                        help="Force demo mode (extractive passages only), even if an API key is set.")
    ask_p.set_defaults(func=cmd_ask)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
