import json
import uuid
from typing import Optional

import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from gme.config import Settings, get_settings
from gme.qdrant_setup import get_qdrant_client
from gme.state import get_conn, get_upsert_pending, mark_upsert_ok

UPSERT_BATCH_SIZE = 256


def run_qdrant_upsert(settings: Optional[Settings] = None) -> None:
    if settings is None:
        settings = get_settings()

    client = get_qdrant_client(settings)

    if not settings.vectors_parquet.exists():
        print("vectors.parquet not found — run 'gme collect' first.")
        return

    # Build vector lookup: article_id → vector
    print("Loading vectors from parquet…")
    tbl = pq.read_table(settings.vectors_parquet)
    vector_map: dict[str, list[float]] = {}
    for batch in tbl.to_batches():
        aids = batch.column("article_id").to_pylist()
        vecs = batch.column("vector").to_pylist()
        for aid, vec in zip(aids, vecs):
            vector_map[aid] = vec

    print(f"Loaded {len(vector_map)} vectors.")

    with get_conn(settings.state_db) as conn:
        pending = get_upsert_pending(conn)

    if not pending:
        print("No records pending upsert.")
        return

    # Filter to records with vectors
    to_upsert = [r for r in pending if str(r["article_id"]) in vector_map]
    skipped = len(pending) - len(to_upsert)
    if skipped:
        print(f"[yellow]Skipping {skipped} records without vectors in parquet.[/yellow]")

    print(f"Upserting {len(to_upsert)} points to '{settings.qdrant_collection}'…")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Upserting to Qdrant…", total=len(to_upsert))

        for i in range(0, len(to_upsert), UPSERT_BATCH_SIZE):
            chunk = to_upsert[i : i + UPSERT_BATCH_SIZE]
            points: list[PointStruct] = []
            article_ids: list[str] = []

            for row in chunk:
                article_id = str(row["article_id"])
                point_uuid_str = str(row["point_uuid"])
                payload = json.loads(row["payload_json"] or "{}")
                vector = vector_map[article_id]

                points.append(
                    PointStruct(
                        id=point_uuid_str,
                        vector=vector,
                        payload=payload,
                    )
                )
                article_ids.append(article_id)

            client.upload_points(
                collection_name=settings.qdrant_collection,
                points=points,
                parallel=4,
                max_retries=3,
            )

            with get_conn(settings.state_db) as conn:
                mark_upsert_ok(conn, article_ids)

            progress.advance(task, len(chunk))

    print("Upsert complete.")
