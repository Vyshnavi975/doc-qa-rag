"""Optional answer synthesis via a hosted LLM, with a no-API-key demo mode.

If ANTHROPIC_API_KEY or OPENAI_API_KEY is set in the environment, the
retrieved chunks are sent to that provider's chat API and the model is asked
to answer the user's question using only the provided context. If neither
key is set (or the API call fails), `demo_answer` provides an extractive
fallback: the most relevant retrieved passages, verbatim, clearly labeled as
demo mode so it's never mistaken for a generated answer.

Only the standard `requests` library is used here -- no provider SDKs -- to
keep the dependency footprint small.
"""

from __future__ import annotations

import os
from typing import List, Optional

import requests

from .retriever import RetrievalResult

ANTHROPIC_MODEL = os.environ.get("DOCQA_ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
OPENAI_MODEL = os.environ.get("DOCQA_OPENAI_MODEL", "gpt-4o-mini")

REQUEST_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's "
    "question using ONLY the information in the provided context passages. "
    "Each passage is labeled with its source document. Cite the source "
    "document name(s) you used in parentheses at the end of relevant "
    "sentences. If the context does not contain enough information to "
    "answer the question, say so plainly instead of guessing."
)


class LLMError(RuntimeError):
    """Raised when a configured LLM provider fails to produce an answer."""


def detect_provider() -> Optional[str]:
    """Return 'anthropic', 'openai', or None based on which API key (if any)
    is set in the environment. Anthropic is preferred if both are set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _format_context(results: List[RetrievalResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[Passage {i} - source: {r.chunk.source}]\n{r.chunk.text}")
    return "\n\n".join(blocks)


def _build_user_message(question: str, results: List[RetrievalResult]) -> str:
    context = _format_context(results)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context passages above."
    )


def _call_anthropic(question: str, results: List[RetrievalResult]) -> str:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 500,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": _build_user_message(question, results)}
                ],
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        parts = data.get("content", [])
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not text:
            raise LLMError(f"Anthropic API returned no text content: {data}")
        return text.strip()
    except requests.RequestException as exc:
        raise LLMError(f"Anthropic API request failed: {exc}") from exc


def _call_openai(question: str, results: List[RetrievalResult]) -> str:
    api_key = os.environ["OPENAI_API_KEY"]
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_message(question, results)},
                ],
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise LLMError(f"OpenAI API returned no choices: {data}")
        text = choices[0]["message"]["content"]
        return text.strip()
    except requests.RequestException as exc:
        raise LLMError(f"OpenAI API request failed: {exc}") from exc


def generate_answer(
    question: str, results: List[RetrievalResult], provider: Optional[str] = None
) -> str:
    """Generate a grounded answer using the given (or auto-detected) provider.

    Raises LLMError if no provider is configured or the API call fails; the
    caller (cli.py) is responsible for falling back to demo_answer.
    """
    provider = provider or detect_provider()
    if provider is None:
        raise LLMError("No LLM provider configured (no API key found).")
    if not results:
        raise LLMError("No retrieved passages to ground an answer in.")

    if provider == "anthropic":
        return _call_anthropic(question, results)
    if provider == "openai":
        return _call_openai(question, results)
    raise LLMError(f"Unknown provider: {provider}")


def demo_answer(question: str, results: List[RetrievalResult]) -> str:
    """Extractive fallback used when no API key is configured: return the
    top retrieved passages verbatim, clearly labeled as demo mode."""
    if not results:
        return (
            "[DEMO MODE - no API key found, showing retrieved passages]\n\n"
            "No relevant passages were found in the indexed documents for "
            f"the question: {question!r}"
        )

    lines = [
        "[DEMO MODE - no ANTHROPIC_API_KEY / OPENAI_API_KEY found in the "
        "environment, so no answer is being generated by an LLM. Showing the "
        "most relevant passages retrieved from your documents instead.]",
        "",
    ]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. (source: {r.chunk.source}, relevance: {r.score:.3f})")
        lines.append(f"   {r.chunk.text}")
        lines.append("")
    return "\n".join(lines).rstrip()
