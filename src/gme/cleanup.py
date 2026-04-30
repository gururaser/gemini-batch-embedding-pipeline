from pathlib import Path


def _fmt_bytes(n: int) -> str:
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _eligible_shards(conn) -> tuple[list[tuple[Path, int]], int]:
    """Return (eligible_files, total_shard_count) where eligible files are fully embedded."""
    total = conn.execute(
        "SELECT COUNT(*) FROM batches WHERE request_file IS NOT NULL"
    ).fetchone()[0]
    rows = conn.execute(
        """
        SELECT b.request_file
        FROM batches b
        JOIN records r ON r.shard_id = b.shard_id
        WHERE b.request_file IS NOT NULL
        GROUP BY b.shard_id
        HAVING COUNT(r.article_id) > 0
           AND COUNT(r.article_id) = SUM(r.embed_status = 'ok')
        """
    ).fetchall()
    eligible = []
    for row in rows:
        p = Path(row["request_file"])
        if p.exists():
            eligible.append((p, p.stat().st_size))
    return eligible, total


def _eligible_results(conn, batches_out_dir: Path) -> tuple[list[tuple[Path, int]], int]:
    """Return (eligible_files, total_result_count) where eligible files are fully upserted."""
    total = conn.execute(
        "SELECT COUNT(*) FROM batches WHERE state = 'SUCCEEDED'"
    ).fetchone()[0]
    rows = conn.execute(
        """
        SELECT b.batch_id
        FROM batches b
        JOIN records r ON r.batch_id = b.batch_id
        WHERE b.state = 'SUCCEEDED'
        GROUP BY b.batch_id
        HAVING COUNT(r.article_id) > 0
           AND COUNT(r.article_id) = SUM(r.upsert_status = 'ok')
        """
    ).fetchall()
    eligible = []
    for row in rows:
        p = batches_out_dir / f"{row['batch_id'].replace('/', '_')}.jsonl"
        if p.exists():
            eligible.append((p, p.stat().st_size))
    return eligible, total


def run_cleanup(dry_run: bool = False) -> None:
    import typer
    from rich.console import Console

    from gme.config import get_settings
    from gme.state import get_conn

    console = Console()
    settings = get_settings()

    if not settings.state_db.exists():
        console.print("[yellow]No state DB found. Run 'gme ingest' first.[/yellow]")
        raise typer.Exit(1)

    with get_conn(settings.state_db) as conn:
        shards, total_shards = _eligible_shards(conn)
        results, total_results = _eligible_results(conn, settings.batches_out_dir)

    shard_bytes = sum(s for _, s in shards)
    result_bytes = sum(s for _, s in results)
    total_bytes = shard_bytes + result_bytes

    console.print(f"\n[bold]Shards[/bold] ({settings.batches_in_dir}):")
    console.print(
        f"  {len(shards)} of {total_shards} shards fully embedded"
        f" — eligible for deletion ({_fmt_bytes(shard_bytes)})"
    )
    console.print(f"  {total_shards - len(shards)} shards skipped (still in progress)")

    console.print(f"\n[bold]Results[/bold] ({settings.batches_out_dir}):")
    console.print(
        f"  {len(results)} of {total_results} result files fully upserted"
        f" — eligible for deletion ({_fmt_bytes(result_bytes)})"
    )
    console.print(f"  {total_results - len(results)} result files skipped (not yet upserted)")

    console.print(f"\nTotal reclaimable: [green]{_fmt_bytes(total_bytes)}[/green]")

    if dry_run:
        console.print("\nRun without [bold]--dry-run[/bold] to delete.")
        return

    deleted = 0
    for p, _ in shards + results:
        p.unlink()
        deleted += 1

    console.print(f"\nDeleted {deleted} files, reclaimed [green]{_fmt_bytes(total_bytes)}[/green].")
