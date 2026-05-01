.PHONY: up down sync pilot full verify cleanup lint batch-jobs batch-cancel batch-delete

up:
	docker compose up -d

down:
	docker compose down

sync:
	uv sync

# Full pilot run (500 records)
pilot:
	uv run gme ingest --limit 500
	uv run gme download-images
	uv run gme build-shards
	uv run gme submit
	uv run gme collect
	uv run gme qdrant-init
	uv run gme qdrant-upsert
	uv run gme verify

# Full ingestion run
full:
	uv run gme ingest
	uv run gme download-images
	uv run gme build-shards
	uv run gme submit
	uv run gme collect
	uv run gme qdrant-init
	uv run gme qdrant-upsert
	uv run gme verify

verify:
	uv run gme verify

cleanup:
	uv run gme cleanup

lint:
	uv run ruff check src/
	uv run ruff format --check src/

# Batch job management
# Usage: make batch-cancel BATCH_ID=batches/123456
# Usage: make batch-delete BATCH_ID=batches/123456
batch-jobs:
	uv run gme batch-jobs

batch-cancel:
	@test -n "$(BATCH_ID)" || (echo "Usage: make batch-cancel BATCH_ID=batches/123456" && exit 1)
	uv run gme batch-cancel $(BATCH_ID) --yes

batch-delete:
	@test -n "$(BATCH_ID)" || (echo "Usage: make batch-delete BATCH_ID=batches/123456" && exit 1)
	uv run gme batch-delete $(BATCH_ID) --yes
