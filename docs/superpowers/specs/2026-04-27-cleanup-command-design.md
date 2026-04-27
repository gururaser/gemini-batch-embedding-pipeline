# Design: `gme cleanup` — State-Aware Artifact Cleanup Command

**Date:** 2026-04-27

## Problem

Pipeline intermediates accumulate indefinitely with no cleanup mechanism. For a full 105K-record run:

| Artifact | Location | Estimated size |
|---|---|---|
| JSONL shard files | `data/batches/in/` | ~914 MB |
| Gemini result files | `data/batches/out/` | ~105 MB |
| Normalized JPEGs | `data/images/` | ~20 MB |

No cleanup logic existed before this change. Manual `rm` was the only option.

## Decision

Add `gme cleanup` — a state-aware command that queries `state.db` to determine which intermediate files are safe to delete, with a `--dry-run` mode that shows sizes before committing.

Images (`data/images/`) are always kept: they're cheap (~20 MB), SHA256-named, and prevent re-downloads on future runs. `state.db` and `vectors.parquet` are never touched.

## Deletion Rules

| Artifact | Safe to delete when |
|---|---|
| `data/batches/in/shard_XXXXX.jsonl` | All records with that `shard_id` have `embed_status = 'ok'` |
| `data/batches/out/batches_<id>.jsonl` | All records with that `batch_id` have `upsert_status = 'ok'` |
| `data/images/` | Never |
| `data/state.db` | Never |
| `data/vectors.parquet` | Never |

Partially-completed shards and in-progress batches are silently skipped — the command is safe to run at any pipeline stage.

## Usage

```bash
# Preview what would be deleted:
uv run gme cleanup --dry-run

# Actually delete:
uv run gme cleanup
```

Dry-run output:
```
Shards (data/batches/in):
  26 of 26 shards fully embedded — eligible for deletion (23.0 MB)
  0 shards skipped (still in progress)

Results (data/batches/out):
  26 of 26 result files fully upserted — eligible for deletion (18.4 MB)
  0 result files skipped (not yet upserted)

Total reclaimable: 41.4 MB

Run without --dry-run to delete.
```

## Implementation

- **`src/gme/cleanup.py`** — all cleanup logic; two SQL queries determine eligibility, then `Path.unlink()` deletes
- **`src/gme/cli.py`** — registers `cleanup` subcommand with `--dry-run` flag

State DB is opened read-only for eligibility queries (no writes occur during cleanup).
