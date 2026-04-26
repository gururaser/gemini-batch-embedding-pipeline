import json
import uuid
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from gme.config import Settings, get_settings
from gme.state import get_conn, init_db, insert_records

PAYLOAD_UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def _make_point_uuid(article_id: str) -> str:
    return str(uuid.uuid5(PAYLOAD_UUID_NAMESPACE, article_id))


def _make_payload(row: dict, exclude: list[str]) -> dict:
    return {k: v for k, v in row.items() if k not in exclude and not _is_embedding_col(k)}


def _is_embedding_col(key: str) -> bool:
    return False  # filtering done via exclude list in config


def run_ingest(limit: Optional[int] = None, settings: Optional[Settings] = None) -> None:
    if settings is None:
        settings = get_settings()

    settings.ensure_dirs()
    init_db(settings.state_db)

    from datasets import load_dataset  # lazy import

    print(f"Loading dataset {settings.hf_dataset!r} ({settings.dataset_split} split)…")
    ds = load_dataset(settings.hf_dataset, split=settings.dataset_split)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    exclude_set = set(settings.exclude_columns) | {"text_to_embed", "image_url"}
    total = len(ds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Ingesting records…", total=total)

        batch_size = 500
        inserted_total = 0

        with get_conn(settings.state_db) as conn:
            batch: list[dict] = []

            for row in ds:
                row = dict(row)
                article_id = str(row["article_id"])
                image_url = row.get("image_url", "")

                payload = {
                    k: v
                    for k, v in row.items()
                    if k not in exclude_set and k not in settings.exclude_columns
                }
                # Serialize non-primitive values
                payload_clean: dict = {}
                for k, v in payload.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        payload_clean[k] = v
                    else:
                        payload_clean[k] = str(v)

                batch.append({
                    "article_id": article_id,
                    "point_uuid": _make_point_uuid(article_id),
                    "image_url": image_url,
                    "payload_json": json.dumps(payload_clean),
                })

                if len(batch) >= batch_size:
                    with get_conn(settings.state_db) as inner_conn:
                        inserted_total += insert_records(inner_conn, batch)
                    batch.clear()

                progress.advance(task)

            if batch:
                with get_conn(settings.state_db) as inner_conn:
                    inserted_total += insert_records(inner_conn, batch)

    print(f"[green]Ingest complete.[/green] {total} rows processed, {inserted_total} new records inserted.")
