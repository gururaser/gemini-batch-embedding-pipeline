.PHONY: up down sync pilot full verify cleanup lint

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
