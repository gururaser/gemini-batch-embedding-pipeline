import random

import pyarrow.parquet as pq
from google import genai
from google.genai import types as genai_types
from rich.console import Console
from rich.table import Table

from gme.config import Settings, get_settings
from gme.qdrant_setup import get_qdrant_client
from gme.state import get_conn, get_counts

console = Console()


def run_verify(settings: Settings | None = None) -> None:
    """
    Perform multiple verification checks to ensure data integrity and search
    relevance across DB, Parquet, and Qdrant.
    """
    if settings is None:
        settings = get_settings()

    client = get_qdrant_client(settings)
    gemini = genai.Client(api_key=settings.gemini_api_key)
    collection = settings.qdrant_collection

    results: dict[str, str] = {}

    # 1. Count match
    with get_conn(settings.state_db) as conn:
        counts = get_counts(conn)
        db_upserted = counts["upserted"]

    qdrant_count = client.count(collection_name=collection, exact=True).count
    match = db_upserted == qdrant_count
    results["Count match"] = (
        f"[green]PASS[/green] (DB={db_upserted}, Qdrant={qdrant_count})"
        if match
        else f"[red]FAIL[/red] (DB={db_upserted} ≠ Qdrant={qdrant_count})"
    )

    # 2. Spot-check 100 random article_ids
    with get_conn(settings.state_db) as conn:
        all_ids = conn.execute(
            "SELECT article_id, point_uuid FROM records WHERE upsert_status = 'ok' LIMIT 10000"
        ).fetchall()

    sample_100 = random.sample(list(all_ids), min(100, len(all_ids)))
    point_ids = [row["point_uuid"] for row in sample_100]

    retrieved = client.retrieve(
        collection_name=collection,
        ids=point_ids,
        with_payload=True,
        with_vectors=False,
    )
    found = len(retrieved)
    results["Spot-check 100"] = (
        f"[green]PASS[/green] ({found}/100 found)"
        if found == len(point_ids)
        else f"[yellow]WARN[/yellow] ({found}/{len(point_ids)} found)"
    )

    # 3. Self-search sanity: top-1 should be self with score > 0.99
    if settings.vectors_parquet.exists():
        tbl = pq.read_table(settings.vectors_parquet)
        aids = tbl.column("article_id").to_pylist()
        vecs = tbl.column("vector").to_pylist()
        puuids = tbl.column("point_uuid").to_pylist()

        sample_20_idx = random.sample(range(len(aids)), min(20, len(aids)))
        self_search_ok = 0

        for idx in sample_20_idx:
            vec = [float(v) for v in vecs[idx]]
            results_q = client.query_points(
                collection_name=collection,
                query=vec,
                limit=1,
                with_payload=False,
            ).points
            if results_q and str(results_q[0].id) == puuids[idx] and results_q[0].score > 0.99:
                self_search_ok += 1

        total_self = min(20, len(aids))
        results["Self-search (top-1=self)"] = (
            f"[green]PASS[/green] ({self_search_ok}/{total_self})"
            if self_search_ok == total_self
            else f"[yellow]WARN[/yellow] ({self_search_ok}/{total_self})"
        )
    else:
        results["Self-search (top-1=self)"] = "[yellow]SKIP[/yellow] (no vectors.parquet)"

    # 4. Cross-modal: text queries should return same product_type_name in top-5
    with get_conn(settings.state_db) as conn:
        sample_rows = conn.execute(
            """
            SELECT article_id, payload_json FROM records
            WHERE upsert_status = 'ok'
            ORDER BY RANDOM() LIMIT 5
            """
        ).fetchall()

    cross_modal_ok = 0
    for row in sample_rows:
        import json
        payload = json.loads(row["payload_json"] or "{}")
        prod_name = payload.get("prod_name", "")
        expected_type = payload.get("product_type_name", "")
        if not prod_name or not expected_type:
            continue

        try:
            emb_result = gemini.models.embed_content(
                model=settings.gemini_model,
                contents=prod_name,
                config=genai_types.EmbedContentConfig(
                    output_dimensionality=settings.embedding_dim
                ),
            )
            query_vec = emb_result.embeddings[0].values
            search_results = client.query_points(
                collection_name=collection,
                query=list(query_vec),
                limit=5,
                with_payload=True,
            ).points
            types_in_top5 = [
                r.payload.get("product_type_name", "")
                for r in search_results
                if r.payload
            ]
            if expected_type in types_in_top5:
                cross_modal_ok += 1
        except Exception as e:
            console.log(f"[yellow]Cross-modal query failed: {e}[/yellow]")

    results["Cross-modal sanity"] = (
        f"[green]PASS[/green] ({cross_modal_ok}/{len(sample_rows)} matched product_type)"
        if cross_modal_ok >= len(sample_rows) // 2
        else f"[yellow]WARN[/yellow] ({cross_modal_ok}/{len(sample_rows)} matched product_type)"
    )

    # 5. Summary stats table
    _print_summary(counts, results)


def _print_summary(counts: dict, checks: dict) -> None:
    """Print summary statistics and check results in a formatted table."""
    # Pipeline summary
    summary = Table(title="Pipeline Summary", show_lines=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")
    for k, v in counts.items():
        summary.add_row(k.replace("_", " ").title(), str(v) if v is not None else "0")
    console.print(summary)

    # Verification checks
    checks_table = Table(title="Verification Checks", show_lines=True)
    checks_table.add_column("Check", style="bold")
    checks_table.add_column("Result")
    for check, result in checks.items():
        checks_table.add_row(check, result)
    console.print(checks_table)
