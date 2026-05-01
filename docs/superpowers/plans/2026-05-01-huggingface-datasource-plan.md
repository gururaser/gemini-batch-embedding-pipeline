# Implementation Plan: Generic HuggingFace Dataset Source

## Overview

Make the pipeline dataset-agnostic by introducing a `HuggingFaceAdapter` abstraction and a `dataset.yaml` config file. Users configure which HF dataset to use, which columns map to ID/text/image, which modality to embed, and which payload fields to index in Qdrant — all without touching source code. The 8-phase pipeline is preserved; only 6 existing files need changes, plus 2 new files.

## Architecture Decisions

- `DatasetConfig` and `HuggingFaceAdapter` live together in `src/gme/adapters.py` — they are dataset-specific and tightly coupled to each other, but loosely coupled to the rest of the pipeline
- Image type (URL vs PIL) is detected once at adapter init via `ds.features`, not per-record — mixed-type columns are not supported
- `pyyaml` is added as a dependency for `dataset.yaml` parsing
- `payload.include` defaults to "all non-embedding columns" when omitted — explicit allowlist when present
- `gme inspect --generate-config` refuses to overwrite an existing `dataset.yaml` and prints a diff instead

## Dependency Graph

```
dataset.yaml (new)
    │
    ▼
DatasetConfig (adapters.py)         ← must exist before anything else
    │
    ├──▶ HuggingFaceAdapter (adapters.py)
    │           │
    │           └──▶ dataset.py (thin wrapper) ─── removes fields from config.py
    │
    ├──▶ state.py (modality param on get_embeddable_records)
    │           │
    │           └──▶ batch_builder.py (modality-aware request building)
    │
    ├──▶ qdrant_setup.py (configurable payload indexes)
    │
    └──▶ cli.py (new ingest flags + inspect command)
```

---

## Phase 1: Foundation — DatasetConfig and dataset.yaml

### Task 1: Add `DatasetConfig`, `dataset.yaml`, and `pyyaml` dependency

**Description:** Create the configuration foundation. `DatasetConfig` is a dataclass loaded from `dataset.yaml` via PyYAML. `dataset.yaml` at the project root encodes the existing HM dataset mapping so nothing breaks yet. This task creates no behaviour change — it only introduces the new config structure.

**Acceptance criteria:**
- [ ] `src/gme/adapters.py` exists and defines `DatasetConfig` as a dataclass with fields: `dataset`, `split`, `id_column`, `text_template`, `image_column`, `modality`, `payload` (include list + indexes dict)
- [ ] `DatasetConfig.from_yaml(path)` loads a `dataset.yaml` file; missing optional fields use defaults (`split="train"`, `modality="multimodal"`, `id_column=None`, `image_column=None`, `payload.include=None`, `payload.indexes={}`)
- [ ] `dataset.yaml` exists at project root and correctly represents the existing HM ecommerce dataset mapping
- [ ] `pyyaml` is added to `pyproject.toml` dependencies
- [ ] `uv sync` succeeds

**Verification:**
- [ ] `python -c "from gme.adapters import DatasetConfig; c = DatasetConfig.from_yaml('dataset.yaml'); print(c.dataset)"` prints `Qdrant/hm_ecommerce_products`
- [ ] `uv run ruff check src/` passes

**Dependencies:** None

**Files touched:**
- `src/gme/adapters.py` (new)
- `dataset.yaml` (new)
- `pyproject.toml`

**Estimated scope:** Small

---

### Task 2: Implement `HuggingFaceAdapter` (URL path) and rewire ingest

**Description:** Implement `NormalizedRecord` and `HuggingFaceAdapter` for URL-type image columns. Add all startup validation (schema checks, modality/image_column consistency, indexes/include consistency). Rewrite `dataset.py` as a thin wrapper that instantiates the adapter and iterates records into the existing DB insert path. Remove `hf_dataset`, `dataset_split`, and `exclude_columns` from `config.py`.

**Acceptance criteria:**
- [ ] `NormalizedRecord` dataclass is defined in `adapters.py` with fields: `record_id`, `text`, `image_url`, `image_path`, `payload`
- [ ] `HuggingFaceAdapter.__init__` validates the dataset schema on construction and raises `ValueError` with the exact error messages from the spec for all 7 error conditions
- [ ] `HuggingFaceAdapter.iter_records(limit)` yields `NormalizedRecord` for each row; URL-type columns set `image_url`, leave `image_path=None`
- [ ] Row-index fallback prints a yellow Rich warning once before iteration begins
- [ ] `payload` on each record respects `payload.include` (allowlist when set; all non-embedding columns when omitted); non-primitive values are coerced to `str`
- [ ] `dataset.py`'s `run_ingest` instantiates `HuggingFaceAdapter` from `DatasetConfig` loaded from `dataset.yaml`; DB insert logic is unchanged
- [ ] `config.py` no longer contains `hf_dataset`, `dataset_split`, or `exclude_columns`
- [ ] `gme ingest --limit 5` completes successfully against the HM dataset via `dataset.yaml`

**Verification:**
- [ ] `uv run gme ingest --limit 5` runs to completion, inserts 5 records into `data/state.db`
- [ ] `uv run gme status` shows 5 total records
- [ ] `uv run ruff check src/` passes

**Dependencies:** Task 1

**Files touched:**
- `src/gme/adapters.py`
- `src/gme/dataset.py`
- `src/gme/config.py`

**Estimated scope:** Medium

---

### Checkpoint: Phase 1

- [ ] `uv run gme ingest --limit 5` succeeds
- [ ] `uv run gme status` shows correct counts
- [ ] `uv run ruff check src/` passes
- [ ] No hardcoded column names remain in `config.py`, `dataset.py`

---

## Phase 2: PIL Image Support

### Task 3: Add PIL image path to `HuggingFaceAdapter`

**Description:** Extend `HuggingFaceAdapter` to handle `datasets.Image` feature-typed columns. At init, check `ds.features[image_column]` — if it is a `datasets.Image` instance, set `self._image_type = "pil"`. During `iter_records`, PIL images are decoded by the `datasets` library, normalised to JPEG ≤512px via the existing `_normalize_image` logic from `images.py`, written to `data/images/<sha256>.jpg`, and the resulting path is set on `NormalizedRecord.image_path`. `image_url` is left as an empty string. The ingest loop in `dataset.py` then sets `image_status='ok'` for these records immediately (rather than `'pending'`), bypassing the download phase.

**Acceptance criteria:**
- [ ] `HuggingFaceAdapter` detects `datasets.Image` feature type at init and sets `_image_type = "pil"` (vs `"url"` for string columns)
- [ ] For PIL-type records: image is normalised and written to `data/images/<sha256>.jpg`; `NormalizedRecord.image_path` is set; `NormalizedRecord.image_url` is empty string
- [ ] PIL image decode failure per-record sets `image_status='failed'` and continues (does not abort the batch)
- [ ] `dataset.py`'s ingest loop sets `image_status='ok'` for PIL records at insert time
- [ ] After running `gme ingest` on a PIL-type dataset, `gme download-images` prints "0 images pending" and exits cleanly

**Verification:**
- [ ] Test manually with a small PIL-type HF dataset (e.g., `lambdalabs/pokemon-blip-captions` which has a native `Image` column): `uv run gme ingest --limit 5` completes with all 5 records showing `image_status='ok'`
- [ ] `uv run gme download-images` immediately exits with "0 images pending"
- [ ] `uv run ruff check src/` passes

**Dependencies:** Task 2

**Files touched:**
- `src/gme/adapters.py`
- `src/gme/dataset.py`

**Estimated scope:** Small

---

### Checkpoint: Phase 2

- [ ] PIL-type dataset ingest works end-to-end
- [ ] Download-images phase is correctly a no-op for PIL datasets
- [ ] URL-type dataset (HM ecommerce) still works unchanged

---

## Phase 3: Modality-Aware Sharding

### Task 4: Update `state.py` and `batch_builder.py` for modality

**Description:** `get_embeddable_records` in `state.py` gains a `modality` parameter that changes the SQL query — text-only modality does not require `image_status='ok'`. `batch_builder.py` reads `modality` from `DatasetConfig` (loaded from `dataset.yaml`) and builds the correct Gemini request parts: text-only omits the inline_data part; image-only omits the text part; multimodal keeps both (current behaviour).

**Acceptance criteria:**
- [ ] `get_embeddable_records(conn, modality)` in `state.py` uses the correct SQL for each modality:
  - `"text"`: `WHERE embed_status='pending' AND embed_attempts < 3`
  - `"image"` or `"multimodal"`: `WHERE embed_status='pending' AND image_status='ok' AND embed_attempts < 3`
- [ ] `run_build_shards` in `batch_builder.py` loads `DatasetConfig` from `dataset.yaml` and passes `modality` to `get_embeddable_records`
- [ ] `_build_request` (or equivalent) produces text-only, image-only, or both parts based on modality
- [ ] Setting `modality: text` in `dataset.yaml` produces JSONL shards with only a text part
- [ ] Setting `modality: image` produces shards with only an inline_data part
- [ ] Setting `modality: multimodal` produces shards with both parts (existing behaviour)

**Verification:**
- [ ] With `modality: text` in `dataset.yaml`, run `gme ingest --limit 3` then `gme build-shards`; inspect a shard file and confirm no `inline_data` key is present
- [ ] With `modality: multimodal`, confirm both `text` and `inline_data` are present in shard lines
- [ ] `uv run ruff check src/` passes

**Dependencies:** Task 2

**Files touched:**
- `src/gme/state.py`
- `src/gme/batch_builder.py`

**Estimated scope:** Medium

---

### Checkpoint: Phase 3

- [ ] All three modalities produce correct JSONL shard content (manually inspected)
- [ ] Text-only modality does not block on images
- [ ] `uv run ruff check src/` passes

---

## Phase 4: Configurable Qdrant Payload Indexes

### Task 5: Drive `qdrant-init` from `DatasetConfig`

**Description:** Remove the hardcoded `KEYWORD_INDEX_FIELDS` constant from `qdrant_setup.py`. Instead, load `DatasetConfig` from `dataset.yaml` and iterate `payload.indexes` to create each index with the correct type. Map the YAML type strings (`keyword`, `integer`, `float`, `bool`, `text`, `geo`, `datetime`, `uuid`) to `qdrant_client.models.PayloadSchemaType` enum values.

**Acceptance criteria:**
- [ ] `KEYWORD_INDEX_FIELDS` constant is removed from `qdrant_setup.py`
- [ ] `run_qdrant_init` loads `DatasetConfig` and creates one payload index per entry in `payload.indexes` with the declared type
- [ ] All 8 supported type strings map correctly to `PayloadSchemaType` values; an unrecognised type string raises a clear `ValueError`
- [ ] `gme qdrant-init` creates the correct indexes for the HM dataset as declared in `dataset.yaml`

**Verification:**
- [ ] `uv run gme qdrant-init` succeeds; verify in Qdrant dashboard or via `qdrant_client.get_collection()` that indexes match `dataset.yaml`
- [ ] `uv run ruff check src/` passes

**Dependencies:** Task 1

**Files touched:**
- `src/gme/qdrant_setup.py`

**Estimated scope:** Small

---

## Phase 5: CLI Flags and `gme inspect`

### Task 6: Add CLI overrides to `gme ingest` and implement `gme inspect`

**Description:** Add `--dataset`, `--split`, `--id-col`, `--text-template`, `--image-col`, `--modality`, and `--config` flags to `gme ingest`. Flags override the values loaded from `dataset.yaml`. Implement `gme inspect` as a new Typer command: load dataset features (no rows), print a Rich table of column names, types, and notes ("good ID candidate", "not embeddable directly"), and optionally write a starter `dataset.yaml` via `--generate-config`. If `dataset.yaml` already exists, print a unified diff instead of overwriting.

**Acceptance criteria:**
- [ ] All 7 CLI flags are present on `gme ingest` and correctly override `DatasetConfig` values
- [ ] `gme ingest --dataset some/dataset --text-template "{title}" --modality text --limit 3` works without a `dataset.yaml`
- [ ] `gme inspect --dataset squad` prints a Rich table with all columns, their types, and notes
- [ ] "Good ID candidate" note appears for columns named `id`, `_id`, `uuid`, `key` that pass the 100-row uniqueness sample
- [ ] "Not embeddable directly" note appears for non-string, non-Image columns (sequences, structs, etc.)
- [ ] `gme inspect --dataset squad --generate-config` writes `dataset.yaml` when it doesn't exist
- [ ] `gme inspect --dataset squad --generate-config` refuses to overwrite and prints a unified diff when `dataset.yaml` already exists

**Verification:**
- [ ] `uv run gme inspect --dataset squad` completes and prints a table (no rows loaded, fast)
- [ ] `uv run gme ingest --dataset Qdrant/hm_ecommerce_products --id-col article_id --text-template "{text_to_embed}" --image-col image_url --modality multimodal --limit 3` succeeds without a `dataset.yaml`
- [ ] `uv run ruff check src/` passes

**Dependencies:** Tasks 2, 3, 4, 5

**Files touched:**
- `src/gme/cli.py`

**Estimated scope:** Medium

---

### Checkpoint: Phase 5 — Full System

- [ ] `uv run gme inspect --dataset <any-hf-dataset>` works
- [ ] `uv run gme ingest` (with `dataset.yaml`) works for HM dataset end-to-end
- [ ] `uv run gme ingest <flags>` works without `dataset.yaml`
- [ ] All three modalities verified
- [ ] `uv run ruff check src/` passes
- [ ] `uv run gme status` shows correct counts after a pilot run

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `datasets.Image` feature type detection is unreliable across HF dataset builders | High | Test Task 3 with 2-3 real PIL datasets before closing; fall back to duck-typing (`hasattr(feature, 'decode_example')`) if needed |
| `str.format_map()` silently accepts extra keys, masking typos in `text_template` | Low | Log the rendered text for the first record during ingest so users can verify output |
| `pyyaml` adds a new security surface (YAML deserialization) | Low | Use `yaml.safe_load()` only — never `yaml.load()` |
| Removing `.env` dataset fields breaks existing user environments | Low | The `extra="ignore"` setting in `Settings` silently drops unknown env vars; old `.env` files won't cause errors |

## Open Questions (decided)

- `gme inspect --generate-config` refuses to overwrite existing `dataset.yaml` and prints a diff
- `dataset.yaml` is committed to the repo (reproducibility artifact)
