"""
Collect completed batch results: download result files, parse vectors,
write to vectors.parquet, update state.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from google import genai
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from gme.config import Settings, get_settings
from gme.state import get_conn, get_succeeded_batches, mark_embed_failed, mark_embed_ok


def _download_result_file(client: genai.Client, file_name: str, dest: Path) -> None:
    if dest.exists():
        return
    content = client.files.download(file=file_name)
    dest.write_bytes(bytes(content))


def _parse_result_line(line: str) -> tuple[Optional[str], Optional[list[float]], Optional[str]]:
    """Returns (article_id, vector, error_msg). error_msg is set on failure."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        return None, None, f"JSON parse error: {e}"

    key = obj.get("key")
    if key is None:
        return None, None, "missing key"

    if "error" in obj:
        return key, None, str(obj["error"])

    try:
        # Result structure: response.embedding.values
        response = obj.get("response", {})
        embedding = response.get("embedding", {})
        values = embedding.get("values", [])
        if not values:
            return key, None, "empty values"
        return key, values, None
    except (KeyError, TypeError) as e:
        return key, None, f"parse error: {e}"


def _append_to_parquet(rows: list[dict], path: Path, dim: int) -> None:
    schema = pa.schema([
        pa.field("article_id", pa.string()),
        pa.field("point_uuid", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])
    table = pa.table(
        {
            "article_id": [r["article_id"] for r in rows],
            "point_uuid": [r["point_uuid"] for r in rows],
            "vector": pa.array(
                [r["vector"] for r in rows],
                type=pa.list_(pa.float32(), dim),
            ),
        },
        schema=schema,
    )
    if path.exists():
        existing = pq.read_table(path)
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, path)
    else:
        pq.write_table(table, path)


def run_collect(settings: Optional[Settings] = None) -> None:
    if settings is None:
        settings = get_settings()

    settings.ensure_dirs()
    client = genai.Client(api_key=settings.gemini_api_key)

    with get_conn(settings.state_db) as conn:
        succeeded = get_succeeded_batches(conn)

    if not succeeded:
        print("No succeeded batches to collect.")
        return

    print(f"Collecting {len(succeeded)} completed batch(es)…")

    total_ok = total_failed = 0
    token_counts: list[int] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Collecting batch results…", total=len(succeeded))

        # Build point_uuid lookup from state DB
        with get_conn(settings.state_db) as conn:
            uuid_map: dict[str, str] = {
                row["article_id"]: row["point_uuid"]
                for row in conn.execute("SELECT article_id, point_uuid FROM records").fetchall()
            }

        for row in succeeded:
            batch_id = row["batch_id"]
            result_file_name = row["result_file"]

            if not result_file_name:
                progress.advance(task)
                continue

            dest = settings.batches_out_dir / f"{batch_id.replace('/', '_')}.jsonl"

            try:
                _download_result_file(client, result_file_name, dest)
            except Exception as e:
                print(f"[red]Failed to download result for {batch_id}: {e}[/red]")
                progress.advance(task)
                continue

            batch_rows: list[dict] = []
            ok_ids: list[str] = []
            fail_ids: list[str] = []

            with open(dest) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    article_id, vector, err = _parse_result_line(line)

                    if err or vector is None:
                        if article_id:
                            fail_ids.append(article_id)
                        continue

                    # Validate dimension
                    if len(vector) != settings.embedding_dim:
                        fail_ids.append(article_id)
                        continue

                    # Validate normalization (auto-normalized by Matryoshka)
                    norm = float(np.linalg.norm(vector))
                    if not (0.95 <= norm <= 1.05):
                        print(f"[yellow]Warning: vector for {article_id} has norm={norm:.4f}[/yellow]")

                    point_uuid = uuid_map.get(article_id, "")
                    batch_rows.append({
                        "article_id": article_id,
                        "point_uuid": point_uuid,
                        "vector": [float(v) for v in vector],
                    })
                    ok_ids.append(article_id)

            if batch_rows:
                _append_to_parquet(batch_rows, settings.vectors_parquet, settings.embedding_dim)

            with get_conn(settings.state_db) as conn:
                for aid in ok_ids:
                    mark_embed_ok(conn, aid)
                for aid in fail_ids:
                    mark_embed_failed(conn, aid)

            total_ok += len(ok_ids)
            total_failed += len(fail_ids)
            progress.advance(task)

    print(f"Collect complete. Vectors written: {total_ok}, failures: {total_failed}")
    if settings.vectors_parquet.exists():
        tbl = pq.read_table(settings.vectors_parquet)
        print(f"vectors.parquet total rows: {len(tbl)}")
