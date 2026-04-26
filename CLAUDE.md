# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
uv sync                         # install all deps into .venv
docker compose up -d            # start local Qdrant on :6333

# Run a phase
uv run gme <phase>              # see phases below
uv run gme status               # check progress from state.db

# Test
uv run pytest tests/ -v         # all tests
uv run pytest tests/test_state.py -v  # single file
uv run pytest -k test_insert    # single test by name

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Shortcuts
make pilot   # ingest --limit 500, then all phases end-to-end
make full    # same without --limit
make verify
```

## Pipeline Architecture

The pipeline is split into 8 sequential, idempotent phases, each a standalone `gme <cmd>`. All state is persisted in `data/state.db` (SQLite WAL), so any phase can be interrupted and re-run safely.

```
ingest → download-images → build-shards → submit → collect → qdrant-init → qdrant-upsert → verify
```

**State machine per record** (column `embed_status`):
`pending` → `in_batch` (build-shards assigns) → `ok` / `failed` (collect sets)

**Token budget enforcement** (Tier 1 @ 90%):
- `MAX_ENQUEUED_TOKENS=432_000` — sum of `est_tokens` across all in-flight Gemini batch jobs
- `MAX_CONCURRENT_JOBS=9` — concurrent batch job cap
- Both enforced in `batch_submit.py`'s submit loop before calling `client.batches.create_embeddings()`

## Key Design Decisions

**SQLite as state store** (`state.py`): `records` table tracks per-row progress through all phases. `batches` table tracks Gemini batch job lifecycle. Always use `get_conn()` context manager — it sets WAL mode, `busy_timeout=5000`, and auto-commits/rolls back.

**Deterministic point IDs**: `uuid5(NAMESPACE_URL, article_id)` in `dataset.py`. This makes Qdrant upserts idempotent — re-running `qdrant-upsert` is always safe.

**Batch result parsing** (`batch_collect.py`): Each result JSONL line has shape `{"key": "<article_id>", "response": {"embeddings": [{"values": [...]}]}}` on success, or `{"key": "...", "error": {...}}` on per-record failure. Succeeded batches can contain per-record errors — check both.

**Image pipeline** (`images.py`): Images are downloaded async (httpx, HTTP/2, semaphore=32), normalized to JPEG at ≤512px longest side via PIL, and stored at `data/images/<sha256>.jpg`. The SHA256 filename makes the cache idempotent.

**Shard JSONL format** (`batch_builder.py`): Each line is `{"key": "<article_id>", "request": {"contents": [...text + inline_data base64...], "config": {"output_dimensionality": 1536}}}`. Images are embedded inline as base64 JPEG — shards can be large files.

**Qdrant collection**: 1536-dim COSINE, on-disk vectors + HNSW, binary quantization (`always_ram=True` keeps quantized index hot), on-disk payload. `qdrant_setup.py` creates KEYWORD payload indexes on 9 low-cardinality fields.

## Configuration

All tuning in `.env` (see `.env.example`). Key knobs:
- `RECORDS_PER_SHARD` (default 100) — calibrated: 512px JPEG costs 259 image tokens + ~70 text (text_to_embed is ~50–70 tokens) ≈ 330 tokens/record. At 9 concurrent: 100 × 330 × 9 = 297K enqueued < 432K cap
- `EMBEDDING_DIM` — Matryoshka-truncated; 1536 is the default (auto-normalized by Gemini)

## Gemini SDK Usage

Uses `google-genai` (not the deprecated `google-generativeai`):
```python
from google import genai
client = genai.Client(api_key=settings.gemini_api_key)
# Batch embeddings:
client.batches.create_embeddings(model="gemini-embedding-2", src=..., config=...)
# File upload for batch input:
client.files.upload(file=path, config={"mime_type": "application/jsonl"})
# File download for batch output:
client.files.download(name=file_name)
# Single embedding (used in verify):
client.models.embed_content(model="gemini-embedding-2", contents=..., config=EmbedContentConfig(...))
```
