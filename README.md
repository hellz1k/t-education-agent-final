# T-Education AI Agent (`t-education-agent`)

Production-ready RAG AI assistant for T-Education programs (`НЭТИ-НЕКСТ 2.0`, track **ML Product / Partner-Agent**).

## Project Structure (Stages 1–3: Ingestion, Hybrid Retrieval & Stateful Agent)

```text
t-education-agent/
├── configs/settings.yaml        # Centralized non-secret configuration
├── data/
│   ├── raw/                     # Source Markdown / DOCX / HTML knowledge base files
│   ├── processed/               # Cleaned documents, contextualized chunks, registry, KB stats & report
│   └── index/                   # Cached dense embeddings, model cache, and index manifest
├── evals/
│   ├── retrieval_golden.json    # 20-query golden retrieval dataset (T1-T4, filters, negative skills)
│   ├── retrieval_holdout.json   # 10-query blind holdout dataset (unseen curriculum & multi-program queries)
│   └── retrieval_benchmark.py   # BM25 vs Dense vs Hybrid RRF evaluation harness
├── scripts/
│   ├── ingest.py                # Reproducible CLI runner for KB ingestion
│   ├── build_index.py           # Build/validate cached dense + BM25 index
│   ├── eval_retrieval.py        # Run retrieval quality & latency benchmark (--holdout, --verbose)
│   └── chat_cli.py              # Terminal chat runner for manual testing with Groq
├── src/t_edu_agent/
│   ├── config.py                # Pydantic v2 configuration loader
│   ├── schemas.py               # Strict data contracts (Document, Chunk, Registry, KBStats)
│   ├── ingestion/
│   │   ├── loader.py            # Loaders + noise cleaning (.md, .txt, .docx, .html/.htm)
│   │   ├── parser.py            # Section parser & deterministic metadata extractor
│   │   ├── chunker.py           # Contextualized section-aware chunker (<= 512 tokens)
│   │   ├── pipeline.py          # Deduplication, URL version resolution, report generation
│   │   └── cli.py               # CLI entry point
│   ├── retrieval/
│   │   ├── models.py            # SearchFilters, RetrievedChunk, ProgramHit, RetrievalResult
│   │   ├── text_processing.py   # Russian/English tokenizer, stemmer, alias expansion, hint inference
│   │   ├── bm25.py              # Zero-dependency Okapi BM25 scorer
│   │   ├── embeddings.py        # FastEmbed ONNX / SentenceTransformers / TF-IDF + SHA-256 disk cache
│   │   └── hybrid.py            # Hybrid RRF + catalog metadata boosts + diversity cap
│   └── agent/
│       ├── state.py             # SessionStore, UserProfile, RetrievalPlan, SourceRef, AgentResponse
│       ├── llm_client.py        # Groq SDK wrapper with retries, backoff, and fallback model
│       ├── prompts.py           # Grounded Russian prompts and XML-isolated source formatting
│       └── orchestrator.py      # TEducationAgent: query rewriting, multi-turn retrieval, citations
└── tests/
    ├── test_ingestion.py        # Ingestion unit & integration tests
    ├── test_retrieval.py        # Retrieval & benchmark tests
    └── test_agent.py            # Multi-turn agent, comparison, isolation & Groq fallback tests
```

## Quick Start (Stages 1–3)

1. Run repeatable ingestion and regenerate `data/processed/*`:

```powershell
python scripts/ingest.py
```

Add a new or updated document via the standard CLI (e.g., new `.md` / `.docx` / `.html` or synthetic test file) and automatically rebuild the processed KB:

```powershell
python scripts/ingest.py --add-file path/to/document.md
```

2. Build/refresh the dense embedding cache (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` via `fastembed` ONNX) and run retrieval benchmarks:

```powershell
python scripts/build_index.py
python scripts/eval_retrieval.py
python scripts/eval_retrieval.py --holdout --verbose
```

3. Test the stateful agent interactively in terminal (requires `GROQ_API_KEY` in `.env`):

```powershell
python scripts/chat_cli.py
```

4. Run all tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Supported Formats & Known Limitations

- **Supported formats:** `.md`, `.txt`, `.docx` (headings, paragraphs, pipe tables), `.html`/`.htm` (navigation, scripts, styles, footers stripped).
- **Not supported in Stage 1:** Binary `.pdf` files (including scanned PDFs without OCR) are skipped and logged as explicit errors in `data/processed/kb_stats.json`.
- **Snapshot freshness:** Answers reflect the saved local snapshot (`verified_at` dates in `data/processed/source_registry.json`), not live scraping at query time.
- **Missing fields:** Unspecified values in official pages (e.g., passing score, seat count, JavaScript program cost/year, office street addresses) are preserved as unknown rather than guessed.
