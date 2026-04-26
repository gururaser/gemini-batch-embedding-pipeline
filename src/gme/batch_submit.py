"""
Submit JSONL shards to Gemini Batch API and poll until completion.

Token/concurrency caps (Tier 1 @ 90%):
  - MAX_ENQUEUED_TOKENS: 432,000 across all in-flight jobs
  - MAX_CONCURRENT_JOBS: 9
"""
import asyncio
import time
from pathlib import Path
from typing import Optional

from google import genai
from rich.console import Console
from rich.live import Live
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential

from gme.config import Settings, get_settings
from gme.state import (
    get_active_batches,
    get_conn,
    get_pending_shards,
    update_batch_state,
    update_batch_submitted,
)

console = Console()

# Gemini Batch job states
TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
ACTIVE_STATES = {"JOB_STATE_PENDING", "JOB_STATE_RUNNING"}

# Map Gemini state strings → our DB states
STATE_MAP = {
    "JOB_STATE_PENDING": "SUBMITTED",
    "JOB_STATE_RUNNING": "RUNNING",
    "JOB_STATE_SUCCEEDED": "SUCCEEDED",
    "JOB_STATE_FAILED": "FAILED",
    "JOB_STATE_CANCELLED": "FAILED",
    "JOB_STATE_EXPIRED": "EXPIRED",
}


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=32))
def _upload_file(client: genai.Client, path: Path) -> str:
    uploaded = client.files.upload(
        file=path,
        config={"mime_type": "application/jsonl"},
    )
    return uploaded.name


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=32))
def _create_batch(client: genai.Client, file_name: str, display_name: str, model: str) -> str:
    job = client.batches.create_embeddings(
        model=model,
        src={"file_name": file_name},
        config={"display_name": display_name},
    )
    return job.name


def _get_job_state(client: genai.Client, batch_id: str) -> tuple[str, Optional[str]]:
    job = client.batches.get(name=batch_id)
    state = str(job.state.name) if hasattr(job.state, "name") else str(job.state)
    result_file: Optional[str] = None
    if state == "JOB_STATE_SUCCEEDED":
        try:
            result_file = job.dest.file_name if job.dest else None
        except AttributeError:
            result_file = None
    return state, result_file


def _make_status_table(in_flight: dict, pending_count: int) -> Table:
    table = Table(title="Batch Submit Status", show_lines=False)
    table.add_column("Batch ID", style="cyan", no_wrap=True, max_width=40)
    table.add_column("Shard", justify="right")
    table.add_column("State")
    table.add_column("Est Tokens", justify="right")

    for batch_id, info in in_flight.items():
        table.add_row(
            batch_id[-20:] if len(batch_id) > 20 else batch_id,
            str(info["shard_id"]),
            info["state"],
            f"{info['est_tokens']:,}",
        )

    table.caption = f"Pending shards: {pending_count}"
    return table


def run_submit(settings: Optional[Settings] = None) -> None:
    if settings is None:
        settings = get_settings()

    client = genai.Client(api_key=settings.gemini_api_key)

    # Recover any previously submitted-but-untracked jobs
    _recover_active_jobs(client, settings)

    # in_flight: batch_id → {shard_id, est_tokens, state}
    in_flight: dict[str, dict] = {}

    with get_conn(settings.state_db) as conn:
        for row in get_active_batches(conn):
            in_flight[row["batch_id"]] = {
                "shard_id": row["shard_id"],
                "est_tokens": row["est_tokens"],
                "state": row["state"],
            }

    with Live(console=console, refresh_per_second=0.2) as live:
        while True:
            # ── Poll in-flight jobs ────────────────────────────────────────
            for batch_id in list(in_flight.keys()):
                try:
                    gemini_state, result_file = _get_job_state(client, batch_id)
                except Exception as e:
                    console.log(f"[yellow]Poll error for {batch_id}: {e}[/yellow]")
                    continue

                db_state = STATE_MAP.get(gemini_state, "RUNNING")

                if gemini_state in TERMINAL_STATES:
                    with get_conn(settings.state_db) as conn:
                        update_batch_state(conn, batch_id, db_state, result_file)
                    console.log(f"[{'green' if db_state == 'SUCCEEDED' else 'red'}]Batch {batch_id[-12:]} → {db_state}[/]")
                    del in_flight[batch_id]
                else:
                    in_flight[batch_id]["state"] = db_state

            # ── Compute budget headroom ────────────────────────────────────
            active_tokens = sum(j["est_tokens"] for j in in_flight.values())
            slots_free = settings.max_concurrent_jobs - len(in_flight)

            # ── Submit next pending shard if budget allows ─────────────────
            if slots_free > 0:
                with get_conn(settings.state_db) as conn:
                    pending = get_pending_shards(conn)

                submitted_any = False
                for row in pending:
                    shard_id = row["shard_id"]
                    est_tokens = row["est_tokens"]
                    request_file = row["request_file"]

                    if len(in_flight) >= settings.max_concurrent_jobs:
                        break
                    if active_tokens + est_tokens > settings.max_enqueued_tokens:
                        continue  # try smaller shards if any

                    try:
                        file_name = _upload_file(client, Path(request_file))
                        batch_id = _create_batch(
                            client,
                            file_name,
                            f"gme-shard-{shard_id:05d}",
                            settings.gemini_model,
                        )
                    except Exception as e:
                        console.log(f"[red]Failed to submit shard {shard_id}: {e}[/red]")
                        continue

                    with get_conn(settings.state_db) as conn:
                        update_batch_submitted(conn, shard_id, batch_id)

                    in_flight[batch_id] = {
                        "shard_id": shard_id,
                        "est_tokens": est_tokens,
                        "state": "SUBMITTED",
                    }
                    active_tokens += est_tokens
                    console.log(f"[blue]Submitted shard {shard_id} → batch {batch_id[-12:]} (~{est_tokens:,} tokens)[/blue]")
                    submitted_any = True

            # ── Check if done ──────────────────────────────────────────────
            with get_conn(settings.state_db) as conn:
                pending_count = conn.execute(
                    "SELECT COUNT(*) FROM batches WHERE state = 'PENDING'"
                ).fetchone()[0]

            live.update(_make_status_table(in_flight, pending_count))

            if not in_flight and pending_count == 0:
                console.log("[green bold]All shards submitted and completed.[/green bold]")
                break

            time.sleep(60)


def _recover_active_jobs(client: genai.Client, settings: Settings) -> None:
    """Re-check any batches stuck in SUBMITTED/RUNNING state on startup."""
    with get_conn(settings.state_db) as conn:
        stuck = conn.execute(
            "SELECT batch_id, shard_id FROM batches WHERE state IN ('SUBMITTED', 'RUNNING')"
        ).fetchall()

    for row in stuck:
        batch_id = row["batch_id"]
        if not batch_id:
            continue
        try:
            gemini_state, result_file = _get_job_state(client, batch_id)
            db_state = STATE_MAP.get(gemini_state, "RUNNING")
            if gemini_state in TERMINAL_STATES:
                with get_conn(settings.state_db) as conn:
                    update_batch_state(conn, batch_id, db_state, result_file)
                console.log(f"Recovered stuck batch {batch_id[-12:]} → {db_state}")
        except Exception as e:
            console.log(f"[yellow]Could not recover {batch_id}: {e}[/yellow]")
