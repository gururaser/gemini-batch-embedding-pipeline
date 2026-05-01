# Generic HuggingFace Dataset Source

## Problem Statement

How might we let any developer point the pipeline at any HuggingFace dataset — choosing their own ID column, text columns, and image columns — and get a working Qdrant vector store out, without touching source code?

## Recommended Direction

Introduce a `HuggingFaceAdapter` abstraction and a `dataset.yaml` config file. The adapter normalises any HF dataset into a `NormalizedRecord` shape the existing 8-phase pipeline already understands. PIL-type images are written to `data/images/` at ingest time; URL-type images follow the existing download path. The `download-images` phase has nothing to do when all images are PIL-sourced. A new `gme inspect` command (not a pipeline phase) introspects any HF dataset schema and can generate a starter `dataset.yaml`.

This approach keeps all 8 pipeline phases intact and requires meaningful changes to only 6 files. No backwards compatibility with the old hardcoded `.env` dataset fields is maintained — everything dataset-specific moves to `dataset.yaml`.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  DatasetConfig                      │
│  (id_column, text_template, image_column, modality) │
│  Sources: dataset.yaml + CLI flag overrides          │
└────────────────────┬────────────────────────────────┘
                     │
              ┌──────▼──────┐
              │HuggingFace  │   new adapter in src/gme/adapters.py
              │  Adapter    │   replaces hardcoded logic in dataset.py
              └──────┬──────┘
                     │ yields NormalizedRecord
                     ▼
┌─────────────────────────────────────────────────────┐
│              Existing 8-phase pipeline               │
│  ingest → download-images → build-shards → ...      │
│  (phases are modality-aware; no structural change)  │
└─────────────────────────────────────────────────────┘

+ gme inspect  (new optional helper, not a pipeline phase)
```

## DatasetConfig

All dataset-specific configuration lives in `dataset.yaml` at the project root. The existing `hf_dataset`, `dataset_split`, and `exclude_columns` fields are removed from `Settings` / `.env`.

**`dataset.yaml` schema:**
```yaml
dataset: Qdrant/hm_ecommerce_products
split: train

id_column: article_id          # omit → row index (warning printed)
text_template: "{text_to_embed}"
image_column: image_url        # omit → text-only modality
modality: multimodal           # text | image | multimodal

payload:
  include:                     # columns to store in Qdrant payload
    - article_id               # omit 'include' entirely → store all non-embedding columns
    - product_type_name
    - colour_group_name
    - detail_desc
    - price
  indexes:                     # subset of 'include' that get a Qdrant payload index
    article_id: keyword        # keyword | integer | float | bool | text | geo | datetime | uuid
    product_type_name: keyword
    colour_group_name: keyword
    detail_desc: text
    price: float
```

**Rules:**
- `payload.include` omitted → all columns except `id_column`, `image_column`, and columns referenced in `text_template` are stored in the payload
- A column in `indexes` must also appear in `include` (or `include` must be omitted) — validated at startup
- `text_template` uses Python `str.format_map()` — no external templating dependency

**CLI flags on `gme ingest`** override top-level fields only:
```
--dataset / -d        HuggingFace dataset name
--split               Dataset split (default: train)
--id-col              ID column name
--text-template       Format string, e.g. "{title}\n{description}"
--image-col           Image column name
--modality            text | image | multimodal
--config              Path to dataset.yaml (default: ./dataset.yaml)
```

Payload config is config-file-only — too structured for CLI flags.

**Precedence:** CLI flags → `dataset.yaml` → built-in defaults (`split=train`, `modality=multimodal`, no ID column → row index with warning).

## HuggingFaceAdapter

New file: `src/gme/adapters.py`.

### NormalizedRecord

```python
@dataclass
class NormalizedRecord:
    record_id: str
    text: str           # rendered from text_template; empty string if no text columns
    image_url: str      # non-empty if image column is a URL string
    image_path: Path | None  # non-None if image written to disk at ingest time (PIL path)
    payload: dict       # columns selected for Qdrant payload
```

### Image type detection

Detected once at adapter initialisation via `ds.features[image_column]`:

| Feature type | Path |
|---|---|
| `datasets.Image` | PIL path — decoded, normalised to JPEG ≤512px, written to `data/images/<sha256>.jpg` during ingest; `image_status` set to `ok` immediately |
| `Value("string")` | URL path — stored in DB as `image_url`; `download-images` phase handles the rest |
| Anything else | Hard error: `"image_column 'X' has unsupported feature type Y"` |

### Text rendering

```python
text = text_template.format_map(row)
# KeyError → clear error naming the missing column (caught at startup via schema check)
```

### ID column

```python
if id_column is None:
    record_id = str(row_index)
    # yellow warning printed once before ingest begins
else:
    record_id = str(row[id_column])
```

`dataset.py` becomes a thin wrapper: `run_ingest` instantiates `HuggingFaceAdapter` from `DatasetConfig` and iterates `NormalizedRecord`s into the existing DB insert path.

## gme inspect

Optional helper command — not a pipeline phase, writes no pipeline state.

**Usage:**
```
gme inspect --dataset <name> [--split train] [--generate-config]
```

**Output:**
```
Dataset: squad  (train split, 87,599 rows)

┌──────────────┬──────────────────────┬─────────────────────────┐
│ Column       │ Type                 │ Notes                   │
├──────────────┼──────────────────────┼─────────────────────────┤
│ id           │ string               │ good ID candidate       │
│ title        │ string               │                         │
│ context      │ string               │                         │
│ question     │ string               │                         │
│ answers      │ sequence<string>     │ not embeddable directly │
└──────────────┴──────────────────────┴─────────────────────────┘
```

`--generate-config` writes a starter `dataset.yaml` with best-guess defaults:
- First `string` column named `id` / `_id` / `uuid` / `key` as `id_column`
- All `string` columns in `payload.include`
- Any `Image`-typed column as `image_column`

**Implementation:** uses `ds.features` only — loads no actual rows. "Good ID candidate" heuristic checks column name and performs a 100-row uniqueness sample.

## Phase Changes

### Modified phases

**`ingest` (phase 1)**
- Reads `DatasetConfig` from `dataset.yaml` + CLI overrides
- Instantiates `HuggingFaceAdapter`; iterates `NormalizedRecord`s into DB
- PIL-type images: written to `data/images/` immediately, `image_status='ok'`
- All hardcoded column references removed

**`download-images` (phase 2)**
- Unchanged in code — processes only `image_status='pending'` records
- PIL-type datasets produce no such records; phase exits with "0 images pending"

**`build-shards` (phase 3)**
- Reads `modality` from `DatasetConfig`
- `text`: text part only; `get_embeddable_records` drops `image_status='ok'` requirement
- `image`: image part only; text part dropped from request
- `multimodal`: current behaviour unchanged
- `get_embeddable_records` gains a `modality` parameter:
  ```sql
  -- text-only
  WHERE embed_status = 'pending' AND embed_attempts < 3

  -- image or multimodal
  WHERE embed_status = 'pending' AND image_status = 'ok' AND embed_attempts < 3
  ```

**`qdrant-init` (phase 6)**
- Reads `payload.indexes` from `DatasetConfig`
- Creates one payload index per entry with the specified type
- `KEYWORD_INDEX_FIELDS` constant removed

### Unchanged phases

`submit`, `collect`, `qdrant-upsert`, `verify`, `cleanup` — no changes needed.

## Error Handling

All schema errors are caught at `gme ingest` startup before any rows are processed.

| Condition | Message |
|---|---|
| `dataset.yaml` not found and no `--dataset` flag | `"No dataset configured. Run 'gme inspect --dataset <name> --generate-config' to get started."` |
| `id_column` not in dataset | `"id_column 'X' not found. Available columns: [...]"` |
| `text_template` references missing column | `"text_template references column 'X' which doesn't exist. Available columns: [...]"` |
| `image_column` not in dataset | `"image_column 'X' not found. Available columns: [...]"` |
| `image_column` has unsupported feature type | `"image_column 'X' has unsupported type Y. Supported: string (URL) or Image (PIL)."` |
| `modality` is `image`/`multimodal` but no `image_column` | `"modality 'multimodal' requires image_column to be set."` |
| `indexes` column not in `include` | `"indexes column 'X' is not in payload.include."` |

**Runtime (per-record, non-fatal):**
- `text_template.format_map` failure → skip record, log warning with `record_id` and missing key
- PIL image decode failure → set `image_status='failed'`, continue

**Warning (non-fatal, printed once):**
- `id_column` not set → yellow warning before ingest begins

## File Change Summary

| File | Change |
|---|---|
| `src/gme/adapters.py` | **New** — `DatasetConfig`, `NormalizedRecord`, `HuggingFaceAdapter` |
| `src/gme/dataset.py` | Modified — thin wrapper over `HuggingFaceAdapter` |
| `src/gme/config.py` | Modified — remove `hf_dataset`, `dataset_split`, `exclude_columns` |
| `src/gme/batch_builder.py` | Modified — modality-aware request building |
| `src/gme/qdrant_setup.py` | Modified — read indexes from `DatasetConfig` |
| `src/gme/cli.py` | Modified — new flags on `ingest`, new `inspect` command |
| `src/gme/state.py` | Modified — `get_embeddable_records` gains `modality` param |
| `dataset.yaml` | **New** — project-root config file (replaces `.env` dataset fields) |
| `src/gme/batch_submit.py` | Unchanged |
| `src/gme/batch_collect.py` | Unchanged |
| `src/gme/qdrant_upsert.py` | Unchanged |
| `src/gme/verify.py` | Unchanged |
| `src/gme/images.py` | Unchanged |
| `src/gme/cleanup.py` | Unchanged |

## Key Assumptions to Validate

- [ ] `datasets.features` reliably exposes `datasets.Image` for PIL-typed columns across all HF dataset builders — verify with 2-3 real datasets before implementing the type detection branch
- [ ] A single image column is sufficient — no known HF dataset requires multiple image columns for a single embedding
- [ ] `str.format_map()` is expressive enough for all reasonable text column combinations — no dataset requires conditional logic in the template

## Not Doing (and Why)

- **Multiple image columns per record** — no clear use case; adds complexity to the shard format
- **Non-HuggingFace sources** (local CSV, JSONL) — out of scope; the adapter pattern leaves the door open without requiring it now
- **Backwards compatibility with `.env` dataset fields** — removed cleanly; old config will produce a clear missing-field error that points to `dataset.yaml`
- **Jinja templating** — `str.format_map()` covers all identified use cases without the dependency
- **Per-record image type detection** — type is detected once from the schema; mixed-type columns within a single dataset are not supported

## Open Questions

- Should `gme inspect --generate-config` overwrite an existing `dataset.yaml` or refuse and print a diff? (Recommend: refuse and print diff — avoid silent data loss)
- Should `dataset.yaml` be committed to the repo or added to `.gitignore`? (Recommend: committed — it's the reproducibility artifact for this dataset run)
