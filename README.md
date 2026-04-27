# Gemini Multimodal Embeddings Pipeline
ETL pipeline that generates multimodal (text + image) embeddings for H&M fashion products using the **Gemini Embedding 2** model via the **Gemini Batch API**, then stores them in **Qdrant** for vector search.


<img width="1693" height="929" alt="high-level-architecture" src="https://github.com/user-attachments/assets/8a2884ca-c647-4a9f-a949-828abfc4ebf0" />



## Overview

```
HuggingFace Dataset          Gemini Batch API             Qdrant (local)
  Qdrant/hm_ecommerce    →   gemini-embedding-2    →    hm_products collection
  ~105,000 products          text + image (base64)       1536-dim COSINE vectors
                             1536-dim Matryoshka          + metadata payload
```

The pipeline is built around the Gemini Batch API (50% cost discount vs. synchronous calls) and enforces Tier 1 rate limits at 90%:

| Limit | Tier 1 | This pipeline |
|---|---|---|
| Batch enqueued tokens (Embedding) | 500,000 | ≤ 450,000 |
| Concurrent batch jobs | 100 | ≤ 9 (with 40 records/shard ≈ 48K tokens/shard) |

### Architecture

```mermaid
flowchart TD
    %% ── External services ──────────────────────────────────────────
    HF{{HuggingFace\nQdrant/hm_ecommerce_products\n~105K products}}
    GEMINI{{Gemini Batch API\ngemini-embedding-2\n1536-dim Matryoshka}}
    QDRANT_SVC{{Qdrant\nlocalhost:6333\nhm_products}}

    %% ── State store ────────────────────────────────────────────────
    DB[(SQLite state.db\nrecords · batches\nWAL mode)]

    %% ── Data artifacts ─────────────────────────────────────────────
    IMAGES[/data/images/\nsha256.jpg\ncached JPEGs ≤512px/]
    SHARDS[/data/batches/in/\nshard_*.jsonl\ntext + base64 image/]
    RESULTS[/data/batches/out/\nbatch_id.jsonl\nembedding results/]
    PARQUET[/data/vectors.parquet\narticle_id · point_uuid\nvector 1536-dim/]

    %% ════════════════════════════════════════════════════════════════
    subgraph INGEST["Ingestion — Phases 1 & 2"]
        direction TB
        P1["Phase 1 · dataset.py\ngme ingest\nLoad HF parquet → SQLite"]
        P2["Phase 2 · images.py\ngme download-images\nasync HTTP/2 · semaphore=32 · 5× retry"]
    end

    subgraph EMBED["Batch Embedding — Phases 3 · 4 · 5"]
        direction TB
        P3["Phase 3 · batch_builder.py\ngme build-shards\nPartition embeddable records into JSONL shards"]
        P4["Phase 4 · batch_submit.py\ngme submit\nUpload · poll · token budget\n≤9 concurrent · ≤432K tokens"]
        P5["Phase 5 · batch_collect.py\ngme collect\nDownload results → vectors.parquet"]
    end

    subgraph STORE["Vector Storage — Phases 6 & 7"]
        direction TB
        P6["Phase 6 · qdrant_setup.py\ngme qdrant-init\nCreate collection · HNSW · binary quant\nPayload keyword indexes"]
        P7["Phase 7 · qdrant_upsert.py\ngme qdrant-upsert\nBatch upsert · parallel=4\nDeterministic UUID5 IDs"]
    end

    subgraph VERIFY["Verification — Phase 8"]
        direction TB
        P8["Phase 8 · verify.py\ngme verify\nCount match · spot-check · self-search · cross-modal"]
    end

    %% ── Phase sequencing ───────────────────────────────────────────
    HF -->|"parquet stream"| P1
    P1 -->|"105K records\nembed_status=pending"| P2
    P2 -->|"image_status=ok"| P3
    P3 --> P4
    P4 -->|"SUCCEEDED batches"| P5
    P5 --> P6
    P6 --> P7
    P7 --> P8

    %% ── State DB interactions ───────────────────────────────────────
    P1 -->|"INSERT records"| DB
    P2 -->|"SET image_status\nok / failed_404 / failed_other"| DB
    DB -->|"image_status=ok\nembed_status=pending"| P3
    P3 -->|"INSERT batches PENDING\nSET embed_status=in_batch"| DB
    DB -->|"PENDING shards\ntoken budget check"| P4
    P4 -->|"SET batches SUBMITTED\n→ RUNNING → SUCCEEDED\nFAILED resets to pending"| DB
    DB -->|"SUCCEEDED batches\nresult_file paths"| P5
    P5 -->|"SET embed_status=ok\nper record"| DB
    DB -->|"embed_status=ok\nupsert_status=pending"| P7
    P7 -->|"SET upsert_status=ok"| DB
    DB -->|"counts · upserted IDs"| P8

    %% ── Artifact flows ─────────────────────────────────────────────
    P2 -->|"write JPEG"| IMAGES
    IMAGES -->|"base64 encode"| P3
    P3 -->|"write"| SHARDS
    SHARDS -->|"upload file"| GEMINI
    GEMINI -->|"result file"| RESULTS
    RESULTS -->|"parse embeddings"| P5
    P5 -->|"write"| PARQUET
    PARQUET -->|"read vectors"| P7
    PARQUET -->|"self-search sample"| P8

    %% ── External service interactions ──────────────────────────────
    P4 <-->|"create / poll jobs"| GEMINI
    P6 -->|"create collection\nset HNSW + binary quant\ncreate payload indexes"| QDRANT_SVC
    P7 -->|"batch upsert\nvectors + payloads"| QDRANT_SVC
    P8 -->|"count · retrieve · query_points"| QDRANT_SVC
    P8 -->|"embed_content\ncross-modal queries"| GEMINI

    %% ── Styling ─────────────────────────────────────────────────────
    classDef phase     fill:#1e40af,stroke:#93c5fd,color:#fff
    classDef external  fill:#92400e,stroke:#fcd34d,color:#fff
    classDef artifact  fill:#14532d,stroke:#86efac,color:#fff
    classDef statedb   fill:#4c1d95,stroke:#c4b5fd,color:#fff

    class P1,P2,P3,P4,P5,P6,P7,P8 phase
    class HF,GEMINI,QDRANT_SVC external
    class IMAGES,SHARDS,RESULTS,PARQUET artifact
    class DB statedb
```

## Dataset

[Qdrant/hm_ecommerce_products](https://huggingface.co/datasets/Qdrant/hm_ecommerce_products) — 105,000 H&M product rows with:
- `text_to_embed` — concatenated product attributes (name, type, color, description)
- `image_url` — S3-hosted product image

Precomputed embedding columns (`dense_embedding`, `sparse_indices`, `sparse_values`) are dropped. All other columns are stored as Qdrant payload.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for environment management
- Docker (for local Qdrant)
- A [Gemini API key](https://aistudio.google.com/app/apikey) with Tier 1 billing

## Setup

```bash
# 1. Clone and enter repo
git clone <repo-url>
cd gemini-multimodal-embeddings

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY

# 4. Start local Qdrant
docker compose up -d
```

## Running the Pipeline

Each phase is independently runnable and fully **idempotent/resumable** — interrupted runs pick up where they left off.

### Pilot run (recommended first)

Test end-to-end with 500 records before committing to the full 105K:

```bash
make pilot
# or step by step:
uv run gme ingest --limit 500
uv run gme download-images
uv run gme build-shards
uv run gme submit        # blocks until all batches complete
uv run gme collect
uv run gme qdrant-init
uv run gme qdrant-upsert
uv run gme verify
```

Inspect the result files under `data/batches/out/` after the first few shards and tune `RECORDS_PER_SHARD` in `.env` based on actual token usage.

### Full run

```bash
make full
# or:
uv run gme ingest        # no --limit flag
uv run gme download-images
uv run gme build-shards
uv run gme submit
uv run gme collect
uv run gme qdrant-init
uv run gme qdrant-upsert
uv run gme verify
```

### Check progress anytime

```bash
uv run gme status
```

## Pipeline Phases

| Command | Phase | Description |
|---|---|---|
| `gme ingest` | 1 | Load HF parquet → SQLite state DB |
| `gme download-images` | 2 | Download + cache images as JPEG ≤1024px |
| `gme build-shards` | 3 | Partition records into JSONL batch files |
| `gme submit` | 4 | Upload shards to Gemini, poll to completion |
| `gme collect` | 5 | Download results → `vectors.parquet` |
| `gme qdrant-init` | 6 | Create Qdrant collection + payload indexes |
| `gme qdrant-upsert` | 7 | Upsert vectors + payloads into Qdrant |
| `gme verify` | 8 | Count match, spot-check, self-search sanity |

## Configuration

All tunable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Required |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `hm_products` | Collection name |
| `EMBEDDING_DIM` | `1536` | Matryoshka output dim (128–3072) |
| `RECORDS_PER_SHARD` | `100` | Records per batch job (~330 tokens/record: 259 image + ~70 text) |
| `MAX_CONCURRENT_JOBS` | `9` | Concurrent Gemini batch jobs |
| `MAX_ENQUEUED_TOKENS` | `432000` | Token cap across in-flight jobs (90% of 500K) |
| `IMAGE_DOWNLOAD_CONCURRENCY` | `32` | Parallel image downloads |
| `IMAGE_MAX_SIDE_PX` | `512` | Image resize cap before embedding (512px reduces Gemini image token cost) |

## Qdrant Collection Schema

- **Vectors**: 1536-dim, COSINE distance, on-disk HNSW (`m=16`, `ef_construct=128`)
- **Quantization**: Binary quantization (`always_ram=True`) — reduces index memory ~32× vs. float32
- **Payload**: all dataset metadata columns except precomputed embeddings
- **Payload indexes (KEYWORD)**: `product_type_name`, `product_group_name`, `colour_group_name`, `perceived_colour_master_name`, `index_group_name`, `garment_group_name`, `department_name`, `section_name`, `article_id`

## Project Structure

```
src/gme/
  config.py          # pydantic-settings (env-driven)
  state.py           # SQLite WAL state store
  dataset.py         # phase 1: ingest
  images.py          # phase 2: image download
  batch_builder.py   # phase 3: JSONL shard builder
  batch_submit.py    # phase 4: Gemini Batch submit + poll daemon
  batch_collect.py   # phase 5: result collection → parquet
  qdrant_setup.py    # phase 6: collection + index creation
  qdrant_upsert.py   # phase 7: vector upsert
  verify.py          # phase 8: end-to-end verification
  cli.py             # Typer CLI entry point
data/
  state.db           # SQLite pipeline state
  images/            # cached JPEGs (sha256-named)
  batches/in/        # JSONL request shards
  batches/out/       # downloaded result files
  vectors.parquet    # collected id→vector table
```

## Error Handling

- **404 images**: permanently skipped, counted in verify report
- **Transient download failures**: retried 5× with exponential backoff
- **Failed/expired batch jobs**: member records reset to `pending`, re-sharded on next `build-shards` run (max 3 attempts per record)
- **Per-record failures within a succeeded batch**: individually marked failed; siblings unaffected
- **Qdrant upsert**: retried 3× via client; deterministic UUID5 IDs make re-runs safe

## Development

```bash
make lint    # ruff check + format check
```

## Big Thanks To
❤️ Claude Code, Gemini CLI, Antigravity and GPT-Image 2.0 
