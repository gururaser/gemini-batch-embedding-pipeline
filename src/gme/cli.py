from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="gme",
    help="Gemini Multimodal Embeddings ETL pipeline",
    add_completion=False,
)
console = Console()


@app.command("ingest")
def ingest(
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Row limit for pilot runs"),
) -> None:
    """Phase 1: Load HF dataset → SQLite state DB."""
    from gme.dataset import run_ingest
    run_ingest(limit=limit)


@app.command("download-images")
def download_images() -> None:
    """Phase 2: Download & cache product images locally."""
    from gme.images import run_download_images
    run_download_images()


@app.command("build-shards")
def build_shards() -> None:
    """Phase 3: Partition embeddable records into JSONL shard files."""
    from gme.batch_builder import run_build_shards
    run_build_shards()


@app.command("submit")
def submit() -> None:
    """Phase 4: Submit shards to Gemini Batch API and poll until done."""
    from gme.batch_submit import run_submit
    run_submit()


@app.command("collect")
def collect() -> None:
    """Phase 5: Download batch results and write vectors to parquet."""
    from gme.batch_collect import run_collect
    run_collect()


@app.command("qdrant-init")
def qdrant_init() -> None:
    """Phase 6: Create Qdrant collection and payload indexes."""
    from gme.qdrant_setup import run_qdrant_init
    run_qdrant_init()


@app.command("qdrant-upsert")
def qdrant_upsert() -> None:
    """Phase 7: Upsert embeddings and payloads into Qdrant."""
    from gme.qdrant_upsert import run_qdrant_upsert
    run_qdrant_upsert()


@app.command("verify")
def verify() -> None:
    """Phase 8: Verify pipeline correctness end-to-end."""
    from gme.verify import run_verify
    run_verify()


@app.command("status")
def status() -> None:
    """Show current pipeline progress from state DB."""
    from gme.config import get_settings
    from gme.state import get_conn, get_counts
    from rich.table import Table

    settings = get_settings()
    if not settings.state_db.exists():
        console.print("[yellow]No state DB found. Run 'gme ingest' first.[/yellow]")
        raise typer.Exit(1)

    with get_conn(settings.state_db) as conn:
        counts = get_counts(conn)
        batch_states = conn.execute(
            "SELECT state, COUNT(*) as n FROM batches GROUP BY state"
        ).fetchall()

    table = Table(title="Pipeline Status", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    for k, v in counts.items():
        table.add_row(k.replace("_", " ").title(), str(v) if v is not None else "0")

    console.print(table)

    if batch_states:
        btable = Table(title="Batch Jobs", show_lines=True)
        btable.add_column("State", style="bold")
        btable.add_column("Count", justify="right")
        for row in batch_states:
            btable.add_row(row["state"], str(row["n"]))
        console.print(btable)


if __name__ == "__main__":
    app()
