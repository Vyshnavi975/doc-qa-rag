# Document Q&A (RAG)

A command-line Retrieval-Augmented Generation (RAG) tool. Point it at a folder
of `.txt` / `.md` / `.pdf` documents, and it builds a local vector index over
the chunked document text, then answers natural-language questions by
retrieving the most relevant chunks and (optionally) using an LLM to
synthesize a grounded answer from them.

No vector database, no `faiss`, no `torch`, no server to run. The retriever
is a from-scratch TF-IDF + cosine-similarity implementation built on plain
`numpy`, so the whole project installs and runs anywhere Python does.

## Features

- **Local, from-scratch vector retrieval.** TF-IDF term weighting with
  smoothed IDF, L2-normalized vectors, and cosine similarity via a single
  `numpy` matrix-vector product — no external vector-search library.
- **Works out of the box, no API key required.** If no LLM API key is
  configured, the tool runs in **demo mode**: it returns the top retrieved
  passages verbatim, clearly labeled, so it's still a fully functional
  extractive Q&A tool with zero setup.
- **LLM-synthesized answers when you want them.** Set `ANTHROPIC_API_KEY` or
  `OPENAI_API_KEY` and the tool sends the retrieved chunks to Claude or GPT
  to generate a natural-language answer grounded in (and citing) your
  documents.
- **Handles `.txt`, `.md`, and `.pdf`** input via `pypdf` for PDF text
  extraction.
- **Persisted or ephemeral indexes.** Build an index once and reuse it
  (`docqa build`), or build one in memory for a single question
  (`docqa ask --docs ...`).
- **Word-window chunking with overlap** so relevant sentences aren't split
  across chunk boundaries.
- **Clean, tested module structure** — chunking, retrieval, and LLM calling
  are independent, unit-testable modules.

## How it works (architecture)

```
 sample_docs/*.txt,*.pdf
          |
          v
   docqa/loaders.py        -- read .txt/.md/.pdf into plain text per document
          |
          v
   docqa/chunking.py       -- split each document into overlapping ~150-word chunks
          |
          v
   docqa/retriever.py      -- TfidfRetriever: build vocabulary, term-frequency
          |                    matrix, apply smoothed IDF, L2-normalize rows
          |                    (numpy only)
          v
   retriever.query(question, top_k)
          |                -- tokenize the question with the same vocabulary,
          |                   vectorize, cosine-similarity rank all chunks,
          |                   return the top_k most relevant
          v
   docqa/llm.py
      |         \
      |          \-- no API key --> demo_answer(): return retrieved passages
      |                              verbatim, labeled "[DEMO MODE]"
      |
      \-- ANTHROPIC_API_KEY or OPENAI_API_KEY set
              --> generate_answer(): send retrieved chunks + question to
                  Claude/GPT via a plain HTTPS request, ask it to answer
                  using ONLY that context and cite sources
          |
          v
   docqa/cli.py             -- formats and prints the final answer + sources
```

Retrieval math, concretely: each chunk becomes a row vector of
`term_frequency * inverse_document_frequency` weights over the corpus
vocabulary, L2-normalized to unit length. A question is embedded into the
same vector space (using the corpus's existing vocabulary and IDF weights),
also L2-normalized. Because both sides are unit vectors, their dot product
*is* the cosine similarity, so ranking is a single `matrix @ query_vector`
call — no iterative search, no approximate-nearest-neighbor index needed at
this scale.

## Project layout

```
doc-qa-rag/
├── docqa/
│   ├── __init__.py
│   ├── loaders.py      # read .txt / .md / .pdf into Document objects
│   ├── chunking.py      # split Documents into overlapping Chunk objects
│   ├── retriever.py     # from-scratch TF-IDF retriever (numpy)
│   ├── llm.py            # optional Anthropic/OpenAI answer synthesis + demo mode
│   └── cli.py            # `docqa build` / `docqa ask` command-line interface
├── sample_docs/          # 3 short original documents about a fictional company
│   ├── company_overview.txt
│   ├── product_lineup.txt
│   └── faq_support.txt
├── tests/
│   ├── test_chunking.py
│   └── test_retriever.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

Requires Python 3.9+.

```bash
cd doc-qa-rag
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### (Optional) enable LLM-generated answers

By default the tool runs entirely offline in demo mode. To have it generate
natural-language answers instead of returning raw passages, export one API
key before running it:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

If `ANTHROPIC_API_KEY` is set it is used (via `claude-3-5-haiku-20241022` by
default); otherwise `OPENAI_API_KEY` is used (via `gpt-4o-mini` by default).
Override the model with `DOCQA_ANTHROPIC_MODEL` / `DOCQA_OPENAI_MODEL`. If
the API call fails for any reason (bad key, network error, rate limit), the
tool automatically falls back to demo mode rather than crashing.

## Usage

### Quick start — ask a question directly against a folder

No index-building step required; the index is built in memory each run:

```bash
python -m docqa.cli ask --docs sample_docs --question "What products does Solstice Robotics sell?"
```

### Build a persisted index, then ask multiple questions against it

```bash
python -m docqa.cli build --docs sample_docs --index .docqa_index
python -m docqa.cli ask --index .docqa_index --question "Who founded the company?"
python -m docqa.cli ask --index .docqa_index --question "What is the return policy?"
```

### CLI options

```
docqa build --docs DOCS_FOLDER --index INDEX_DIR [--chunk-size N] [--overlap N]

docqa ask --question "..." [--index INDEX_DIR] [--docs DOCS_FOLDER]
           [--top-k N] [--no-llm] [--chunk-size N] [--overlap N]
```

- `--no-llm` forces demo mode even if an API key is configured — useful for
  reproducible, offline output.
- `--top-k` controls how many chunks are retrieved (default 4).
- If `--index` points to an index that doesn't exist yet but `--docs` is also
  given, `ask` builds an in-memory index from `--docs` automatically (and
  tells you it isn't being saved).

## Sample Q&A output (against `sample_docs/`, demo mode, no API key)

The `sample_docs/` folder contains three short original documents describing
a fictional greenhouse-robotics company, **Solstice Robotics**. Running the
tool against them with no API key set produces:

```
$ python -m docqa.cli ask --docs sample_docs --question "What is Scout Mini's battery life?" --top-k 1

Loaded 3 document(s), split into 9 chunk(s) (chunk_size=150 words, overlap=30 words).
========================================================================
Q: What is Scout Mini's battery life?
------------------------------------------------------------------------
[DEMO MODE - no ANTHROPIC_API_KEY / OPENAI_API_KEY found in the environment, so no answer is being generated by an LLM. Showing the most relevant passages retrieved from your documents instead.]

1. (source: product_lineup.txt, relevance: 0.321)
   Solstice Robotics: Product Lineup Solstice Robotics sells three products, all built around the Scout robot platform and the Solstice Hub software dashboard. Scout Mini is the entry-level robot, designed for hobbyist growers and small greenhouses under 2,000 square feet. Scout Mini has a battery life of about 6 hours per charge, moves along a single greenhouse bench rail, and reports soil moisture and temperature readings every 15 minutes. Scout Mini costs $899 and does not include a camera. It is the best-selling product by unit volume, accounting for roughly 60 percent of all Scout robots shipped. Scout Pro is the mid-tier robot aimed at commercial greenhouse operators. It adds a high-resolution camera for leaf and pest inspection, can switch between multiple bench rails using a docking rail-changer accessory, and has a battery life of about 10 hours. Scout Pro costs $2,400 and includes one year of Solstice Hub Premium at
========================================================================
```

```
$ python -m docqa.cli ask --docs sample_docs --question "How do I contact customer support?" --top-k 1

Loaded 3 document(s), split into 9 chunk(s) (chunk_size=150 words, overlap=30 words).
========================================================================
Q: How do I contact customer support?
------------------------------------------------------------------------
[DEMO MODE - no ANTHROPIC_API_KEY / OPENAI_API_KEY found in the environment, so no answer is being generated by an LLM. Showing the most relevant passages retrieved from your documents instead.]

1. (source: faq_support.txt, relevance: 0.276)
   Solstice Robotics: Support FAQ Q: How do I contact customer support? A: Email support@solsticerobotics.example or call the support line at (503) 555-0148, Monday through Friday, 8am to 6pm Pacific Time. Canadian customers can also reach the Guelph, Ontario support office directly at (519) 555-0122 during the same hours in Eastern Time. Q: My Scout robot won't leave its charging dock. What should I check first? A: First confirm the bench rail is clear of obstructions and that the rail-end sensor is not covered in dust, since a blocked sensor can make the robot think the rail is occupied. Second, check that the robot's battery is above 15 percent charge; Scout robots refuse to start a run below that threshold to make sure they can return to the dock. If both of those look fine, restart the robot by holding the power button for 10 seconds, then try again. If it
========================================================================
```

```
$ python -m docqa.cli ask --docs sample_docs --question "What is the return policy for Scout Enterprise?" --top-k 1

Loaded 3 document(s), split into 9 chunk(s) (chunk_size=150 words, overlap=30 words).
========================================================================
Q: What is the return policy for Scout Enterprise?
------------------------------------------------------------------------
[DEMO MODE - no ANTHROPIC_API_KEY / OPENAI_API_KEY found in the environment, so no answer is being generated by an LLM. Showing the most relevant passages retrieved from your documents instead.]

1. (source: faq_support.txt, relevance: 0.238)
   from the mobile app under Settings > Robot > Transfer Robot. Note that transferring a robot does not transfer its historical sensor data, which stays associated with the original Hub account. Q: What is the return policy? A: Scout Mini and Scout Pro purchases can be returned within 30 days of delivery for a full refund, provided the robot is in working condition. Scout Enterprise fleet orders are custom-configured and are not eligible for return once installation has begun, though they remain covered by the standard hardware warranty. Q: Does Solstice Hub work without an internet connection? A: Scout robots can operate and store data locally for up to 14 days without an internet connection. Once the greenhouse's network connection is restored, the robot automatically uploads any queued data to Solstice Hub.
========================================================================
```

Every retrieved answer above is a real chunk pulled from `sample_docs/`
purely by TF-IDF cosine similarity — nothing was hand-picked or hardcoded.
Try your own questions, e.g. `"Who is the CEO?"`, `"How much does the
extended warranty cost?"`, or `"Does the robot work without internet?"`.

With `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` set, the same commands instead
return a synthesized natural-language answer (e.g. *"Scout Mini has a
battery life of about 6 hours per charge (product_lineup.txt)."*) with a
"Sources consulted" list, rather than raw passages.

## Running the tests

```bash
pip install pytest    # or just use the standard-library unittest runner below
python -m pytest tests/ -v
# or, with no extra dependency:
python -m unittest discover -s tests -v
```

Tests cover word-window chunking (boundary sizes, overlap correctness, no
words lost, empty-input handling) and the TF-IDF retriever (tokenization,
ranking relevant chunks above irrelevant ones, handling queries with no
vocabulary overlap, and index save/load round-tripping) — none require
network access or an API key.

## Design notes / limitations

- TF-IDF is a lexical (keyword-overlap) retriever, not a semantic embedding
  model — it won't match a query and passage that use entirely different
  wording for the same concept. This is a deliberate tradeoff for zero
  dependencies and full portability; swapping in a real embedding model
  (e.g. via a local sentence-transformers model or an embeddings API) would
  be a natural extension and would only require changing `retriever.py`.
- Chunking is word-count-based, not sentence- or paragraph-aware. It's
  simple and fast, and the overlap window mitigates most boundary-cutting
  issues, but a production system might chunk on paragraph/sentence
  boundaries instead.
- The saved index format is not versioned; if you change `--chunk-size` or
  `--overlap`, rebuild the index rather than reusing an old one.
