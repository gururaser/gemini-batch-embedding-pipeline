import base64
import json
from pathlib import Path

from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn

from gme.config import Settings, get_settings
from gme.state import (
    assign_to_shard,
    get_conn,
    get_embeddable_records,
    insert_batch,
)


def _build_request(
    article_id: str,
    text: str,
    image_path: Path,
    embedding_dim: int,
) -> dict:
    """Construct a Gemini batch request dictionary for a single product."""
    image_bytes = image_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "key": article_id,
        "request": {
            "output_dimensionality": embedding_dim,
            "content": {
                "parts": [
                    {"text": text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            },
        },
    }


def _get_text_to_embed(payload_json: str) -> str:
    """Extract the text to be embedded from a JSON payload string."""
    payload = json.loads(payload_json)
    return payload.get("text_to_embed", "")


def run_build_shards(settings: Settings | None = None) -> None:
    """
    Groups embeddable records into JSONL shards ready for Gemini batch submission.
    Each shard is tracked in the state database.
    """
    if settings is None:
        settings = get_settings()

    settings.ensure_dirs()

    with get_conn(settings.state_db) as conn:
        records = get_embeddable_records(conn)

    if not records:
        print("No records ready for sharding (embed_status=pending, image_status=ok).")
        return

    print(f"Building shards for {len(records)} records (shard_size={settings.records_per_shard})…")

    shard_id = _get_next_shard_id(settings)
    shard: list[dict] = []
    article_ids_in_shard: list[str] = []
    shards_written = 0

    with Progress(
        SpinnerColumn(), TextColumn("[bold]{task.description}"), transient=True
    ) as progress:
        task = progress.add_task("Building shards…", total=len(records))

        for row in records:
            article_id = str(row["article_id"])
            sha256 = row["image_sha256"]
            image_path = settings.images_dir / f"{sha256}.jpg"

            if not image_path.exists():
                progress.advance(task)
                continue

            payload_json = row["payload_json"] or "{}"
            payload = json.loads(payload_json)
            text = payload.get("text_to_embed", "")

            request_line = _build_request(article_id, text, image_path, settings.embedding_dim)
            shard.append(request_line)
            article_ids_in_shard.append(article_id)

            if len(shard) >= settings.records_per_shard:
                _flush_shard(shard, article_ids_in_shard, shard_id, settings)
                shards_written += 1
                shard_id += 1
                shard = []
                article_ids_in_shard = []

            progress.advance(task)

        if shard:
            _flush_shard(shard, article_ids_in_shard, shard_id, settings)
            shards_written += 1

    print(f"Built {shards_written} shards in {settings.batches_in_dir}")


def _flush_shard(
    shard: list[dict],
    article_ids: list[str],
    shard_id: int,
    settings: Settings,
) -> None:
    """Write a shard to a JSONL file and record it in the state database."""
    shard_path = settings.batches_in_dir / f"shard_{shard_id:05d}.jsonl"
    with open(shard_path, "w") as f:
        for line in shard:
            f.write(json.dumps(line) + "\n")

    est_tokens = len(shard) * settings.tokens_per_record_estimate

    with get_conn(settings.state_db) as conn:
        insert_batch(conn, shard_id, est_tokens, len(shard), str(shard_path))
        assign_to_shard(conn, article_ids, shard_id)


def _get_next_shard_id(settings: Settings) -> int:
    """Determine the next available shard ID from the database."""
    with get_conn(settings.state_db) as conn:
        row = conn.execute("SELECT MAX(shard_id) FROM batches").fetchone()
        if row[0] is None:
            return 0
        return int(row[0]) + 1
