from pathlib import Path

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
    limit: int | None = typer.Option(None, "--limit", "-n", help="Row limit for pilot runs"),
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="HuggingFace dataset name"),
    split: str | None = typer.Option(None, "--split", help="Dataset split"),
    id_col: str | None = typer.Option(None, "--id-col", help="ID column name"),
    text_template: str | None = typer.Option(
        None, "--text-template", "-t", help='Text format string, e.g. "{title} {body}"'
    ),
    image_col: str | None = typer.Option(None, "--image-col", help="Image column name"),
    modality: str | None = typer.Option(
        None, "--modality", help="Embedding modality: text | image | multimodal"
    ),
    config: Path = typer.Option(Path("dataset.yaml"), "--config", help="Path to dataset.yaml"),
) -> None:
    """Phase 1: Load HF dataset → SQLite state DB."""
    from gme.adapters import DatasetConfig
    from gme.dataset import run_ingest

    if config.exists():
        cfg = DatasetConfig.from_yaml(config)
    elif dataset is not None:
        cfg = DatasetConfig(dataset=dataset)
    else:
        console.print(
            "[red]No dataset configured. "
            "Run 'gme inspect --dataset <name> --generate-config' to get started.[/red]"
        )
        raise typer.Exit(1)

    cfg.apply_overrides(
        dataset=dataset,
        split=split,
        id_column=id_col,
        text_template=text_template,
        image_column=image_col,
        modality=modality,
    )

    run_ingest(limit=limit, cfg=cfg)


@app.command("download-images")
def download_images() -> None:
    """Phase 2: Download & cache product images locally."""
    from gme.images import run_download_images

    run_download_images()


@app.command("build-shards")
def build_shards(
    config: Path = typer.Option(Path("dataset.yaml"), "--config", help="Path to dataset.yaml"),
) -> None:
    """Phase 3: Partition embeddable records into JSONL shard files."""
    from gme.adapters import DatasetConfig
    from gme.batch_builder import run_build_shards

    cfg = DatasetConfig.from_yaml(config) if config.exists() else None
    run_build_shards(cfg=cfg)


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
def qdrant_init(
    config: Path = typer.Option(Path("dataset.yaml"), "--config", help="Path to dataset.yaml"),
) -> None:
    """Phase 6: Create Qdrant collection and payload indexes."""
    from gme.adapters import DatasetConfig
    from gme.qdrant_setup import run_qdrant_init

    cfg = DatasetConfig.from_yaml(config) if config.exists() else None
    run_qdrant_init(cfg=cfg)


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


@app.command("cleanup")
def cleanup(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be deleted without deleting"
    ),
) -> None:
    """Remove intermediate files (shards, results) that are no longer needed."""
    from gme.cleanup import run_cleanup

    run_cleanup(dry_run=dry_run)


@app.command("inspect")
def inspect(
    dataset: str = typer.Option(..., "--dataset", "-d", help="HuggingFace dataset name"),
    split: str = typer.Option("train", "--split", help="Dataset split"),
    generate_config: bool = typer.Option(
        False, "--generate-config", help="Write a starter dataset.yaml"
    ),
    config: Path = typer.Option(
        Path("dataset.yaml"), "--config", help="Output path for --generate-config"
    ),
) -> None:
    """Inspect a HuggingFace dataset schema; optionally generate dataset.yaml."""
    from gme.adapters import run_inspect

    run_inspect(
        dataset=dataset,
        split=split,
        generate_config=generate_config,
        config_path=config,
    )


@app.command("status")
def status() -> None:
    """Show current pipeline progress from state DB."""
    from rich.table import Table

    from gme.config import get_settings
    from gme.state import get_conn, get_counts

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
