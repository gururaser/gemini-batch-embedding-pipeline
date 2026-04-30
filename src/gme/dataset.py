import json
import uuid
from pathlib import Path

from rich import print
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from gme.adapters import DatasetConfig, HuggingFaceAdapter
from gme.config import Settings, get_settings
from gme.state import get_conn, init_db, insert_records, set_image_status

PAYLOAD_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def _make_point_uuid(record_id: str) -> str:
    """Return a deterministic UUID5 for a record_id, making Qdrant upserts idempotent."""
    return str(uuid.uuid5(PAYLOAD_UUID_NAMESPACE, record_id))


def _flush_to_db(
    db_path: Path,
    batch: list[dict],
    pil_ok: list[tuple[str, str]],
    pil_failed: list[str],
) -> int:
    """Write a batch of records and image statuses to the state DB; return new-insert count."""
    with get_conn(db_path) as conn:
        count = insert_records(conn, batch)
        for article_id, sha256 in pil_ok:
            set_image_status(conn, article_id, "ok", sha256)
        for article_id in pil_failed:
            set_image_status(conn, article_id, "failed", None)
    return count


def run_ingest(
    limit: int | None = None,
    settings: Settings | None = None,
    cfg: DatasetConfig | None = None,
) -> None:
    """Phase 1: stream HuggingFace dataset rows into the SQLite state DB."""
    if settings is None:
        settings = get_settings()
    if cfg is None:
        cfg = DatasetConfig.from_yaml("dataset.yaml")

    settings.ensure_dirs()
    init_db(settings.state_db)

    adapter = HuggingFaceAdapter(cfg, settings.images_dir, settings.image_max_side_px)
    adapter.load()

    print(f"Loading dataset {cfg.dataset!r} ({cfg.split} split, {adapter.total:,} rows)…")

    batch: list[dict] = []
    pil_ok: list[tuple[str, str]] = []
    pil_failed: list[str] = []
    inserted_total = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Ingesting records…", total=adapter.total)

        for record in adapter.iter_records(limit=limit):
            batch.append({
                "article_id": record.record_id,
                "point_uuid": _make_point_uuid(record.record_id),
                "image_url": record.image_url,
                "payload_json": json.dumps(record.payload),
            })

            if record.image_path is not None:
                pil_ok.append((record.record_id, record.image_path.stem))
            elif record.pil_failed:
                pil_failed.append(record.record_id)

            if len(batch) >= 500:
                inserted_total += _flush_to_db(settings.state_db, batch, pil_ok, pil_failed)
                batch.clear()
                pil_ok.clear()
                pil_failed.clear()

            progress.advance(task)

    if batch:
        inserted_total += _flush_to_db(settings.state_db, batch, pil_ok, pil_failed)

    print(
        f"[green]Ingest complete.[/green] {adapter.total} rows processed, "
        f"{inserted_total} new records inserted."
    )
