import base64
import json
from pathlib import Path

from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn

from gme.adapters import DatasetConfig
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
    image_path: Path | None,
    embedding_dim: int,
    modality: str,
) -> dict:
    """Build a Gemini batch embedding request line for one record.

    Includes only the parts (text, image) that the modality requires.
    """
    parts: list[dict] = []
    if modality in ("text", "multimodal") and text:
        parts.append({"text": text})
    if modality in ("image", "multimodal") and image_path and image_path.exists():
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
    return {
        "key": article_id,
        "request": {
            "output_dimensionality": embedding_dim,
            "content": {"parts": parts},
        },
    }


def run_build_shards(settings: Settings | None = None) -> None:
    """Phase 3: partition pending embeddable records into JSONL shard files."""
    if settings is None:
        settings = get_settings()

    cfg = DatasetConfig.from_yaml("dataset.yaml")
    settings.ensure_dirs()

    with get_conn(settings.state_db) as conn:
        records = get_embeddable_records(conn, modality=cfg.modality)

    if not records:
        print("No records ready for sharding (embed_status=pending, image ready).")
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
            image_path = settings.images_dir / f"{sha256}.jpg" if sha256 else None

            payload = json.loads(row["payload_json"] or "{}")
            text = cfg.text_template.format_map(payload) if cfg.text_template else ""

            request_line = _build_request(
                article_id, text, image_path, settings.embedding_dim, cfg.modality
            )
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
    """Write shard JSONL to disk and record the batch + shard assignments in the state DB."""
    shard_path = settings.batches_in_dir / f"shard_{shard_id:05d}.jsonl"
    with open(shard_path, "w") as f:
        for line in shard:
            f.write(json.dumps(line) + "\n")

    est_tokens = len(shard) * settings.tokens_per_record_estimate

    with get_conn(settings.state_db) as conn:
        insert_batch(conn, shard_id, est_tokens, len(shard), str(shard_path))
        assign_to_shard(conn, article_ids, shard_id)


def _get_next_shard_id(settings: Settings) -> int:
    """Return the next available shard_id (max existing + 1, or 0 if none)."""
    with get_conn(settings.state_db) as conn:
        row = conn.execute("SELECT MAX(shard_id) FROM batches").fetchone()
        if row[0] is None:
            return 0
        return int(row[0]) + 1
