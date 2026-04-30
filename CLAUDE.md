# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
uv sync                         # install all deps into .venv
docker compose up -d            # start local Qdrant on :6333

# Dataset config (for new datasets; H&M default ships as dataset.yaml)
uv run gme inspect --dataset <hf/dataset> --generate-config  # scaffold dataset.yaml
uv run gme inspect --dataset <hf/dataset>                    # print schema only

# Run a phase
uv run gme <phase>              # see phases below
uv run gme status               # check progress from state.db
uv run gme cleanup --dry-run    # preview reclaimable space (shards + results)
uv run gme cleanup              # delete fully-processed intermediates

# Lint
uv run ruff check src/
uv run ruff format src/

# Shortcuts
make pilot    # ingest --limit 500, then all phases end-to-end
make full     # same without --limit
make verify
make cleanup  # delete eligible intermediates (shards + results)
```

## Pipeline Architecture

8 sequential, idempotent phases — each a standalone `gme <cmd>`. All state is persisted in `data/state.db` (SQLite WAL) so any phase can be interrupted and re-run safely.

```
ingest → download-images → build-shards → submit → collect → qdrant-init → qdrant-upsert → verify
```

**Two independent state columns per record:**
- `embed_status`: `pending` → `in_batch` (build-shards) → `ok` / `failed` (collect)
- `upsert_status`: `pending` → `ok` (qdrant-upsert)

Failed/expired batches reset their records to `embed_status='pending'` and `shard_id=NULL` so they are re-sharded on the next `build-shards` run.

**Token budget enforcement** (Tier 1 @ 90%):
- `MAX_ENQUEUED_TOKENS=432_000` — sum of `est_tokens` across all in-flight Gemini batch jobs
- `MAX_CONCURRENT_JOBS=9` — concurrent batch job cap
- Both enforced in `batch_submit.py`'s submit loop before calling `client.batches.create_embeddings()`

## Data Directory Layout

```
data/
├── state.db              — SQLite state machine (never delete)
├── vectors.parquet       — collected 1536-dim embeddings (never delete)
├── images/               — SHA256-named JPEGs ≤512px (keep; prevents re-download)
├── batches/in/           — shard_XXXXX.jsonl (text + base64 image, ~870 KB each)
└── batches/out/          — <batch_id>.jsonl result files from Gemini
```

`gme cleanup` deletes `batches/in/` and `batches/out/` files once state.db confirms all their records are fully embedded and upserted.

## Key Design Decisions

**Generic dataset adapter** (`adapters.py`): `DatasetConfig` (loaded from `dataset.yaml`) drives the whole pipeline — which HF dataset, which columns are the ID / text / image, what goes in the Qdrant payload, and which payload fields get KEYWORD indexes. `HuggingFaceAdapter` normalises any HF dataset row into a `NormalizedRecord` and handles both URL-based and PIL-based image columns. `run_inspect` powers `gme inspect`.

**SQLite as state store** (`state.py`): `records` table tracks per-row progress; `batches` table tracks Gemini batch job lifecycle. Always use `get_conn()` — it sets WAL mode, `busy_timeout=5000`, and auto-commits/rolls back.

**Deterministic point IDs**: `uuid5(NAMESPACE_URL, record_id)` in `dataset.py`. Makes Qdrant upserts idempotent — re-running `qdrant-upsert` is always safe.

**Batch result parsing** (`batch_collect.py`): Each result JSONL line is `{"key": "<article_id>", "response": {"embedding": {"values": [...]}}}` on success, or `{"key": "...", "error": {...}}` on per-record failure. Succeeded batches can still contain per-record errors — check both levels.

**Image pipeline** (`images.py`): Downloaded async (httpx, HTTP/2, semaphore=32), normalized to JPEG at ≤512px longest side via PIL, stored at `data/images/<sha256>.jpg`. SHA256 filename makes the cache idempotent.

**Shard JSONL format** (`batch_builder.py`): Each line is `{"key": "<article_id>", "request": {"contents": [...text + inline_data base64...], "config": {"output_dimensionality": 1536}}}`. Images are embedded inline as base64 — shards can be large.

**Qdrant payload exclusion**: Only `id_column` is auto-excluded from the payload (it becomes the point UUID). `image_column` and text-template columns are kept as useful search-result metadata. Extra columns can be excluded via `payload.exclude` in `dataset.yaml`.

**Qdrant collection**: 1536-dim COSINE, on-disk vectors + HNSW, binary quantization (`always_ram=True` keeps quantized index hot), on-disk payload. `qdrant_setup.py` reads KEYWORD payload index definitions from `DatasetConfig.payload.indexes`.

## Configuration

All tuning in `.env` (see `.env.example`). Key knobs:
- `RECORDS_PER_SHARD` (default 100) — calibrated: 512px JPEG ≈ 259 image tokens + ~70 text ≈ 330 tokens/record. At 9 concurrent jobs: 100 × 330 × 9 = 297K enqueued < 432K cap
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
