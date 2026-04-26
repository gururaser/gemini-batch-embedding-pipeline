import asyncio
import hashlib
import io
from pathlib import Path

import httpx
from PIL import Image
from rich import print
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from gme.config import Settings, get_settings
from gme.state import get_conn, get_pending_image_records, set_image_status


def _is_transient(exc: BaseException) -> bool:
    """Determine if an exception is transient and should be retried."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in (404, 410)
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))


def _sha256(data: bytes) -> str:
    """Calculate the SHA256 hash of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def _normalize_image(data: bytes, max_side: int) -> bytes:
    """
    Open an image, normalize it to RGB if needed, resize it to fit within max_side,
    and return the JPEG bytes.
    """
    img = Image.open(io.BytesIO(data))
    img.verify()
    img = Image.open(io.BytesIO(data))  # re-open after verify (verify closes stream)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


async def _download_one(
    client: httpx.AsyncClient,
    article_id: str,
    image_url: str,
    images_dir: Path,
    max_side: int,
    semaphore: asyncio.Semaphore,
    progress_callback,
) -> tuple[str, str, str | None]:
    """Download and process a single image."""
    async with semaphore:
        try:
            status, sha256 = await _fetch_and_save(client, image_url, images_dir, max_side)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 410):
                status, sha256 = "failed_404", None
            else:
                status, sha256 = "failed_other", None
        except Exception:
            status, sha256 = "failed_other", None
        finally:
            progress_callback()
    return article_id, status, sha256


@retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=32),
)
async def _fetch_and_save(
    client: httpx.AsyncClient, image_url: str, images_dir: Path, max_side: int
) -> tuple[str, str]:
    """Fetch an image from a URL, normalize it, and save it to the images directory."""
    response = await client.get(image_url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    raw = response.content
    normalized = _normalize_image(raw, max_side)
    sha = _sha256(normalized)
    dest = images_dir / f"{sha}.jpg"
    if not dest.exists():
        dest.write_bytes(normalized)
    return "ok", sha


async def _run_async(
    records: list,
    images_dir: Path,
    max_side: int,
    concurrency: int,
    update_fn,
    advance_fn,
) -> None:
    """Run the asynchronous download loop for all records."""
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency // 2)

    async with httpx.AsyncClient(http2=True, limits=limits) as client:
        tasks = [
            _download_one(
                client,
                row["article_id"],
                row["image_url"],
                images_dir,
                max_side,
                semaphore,
                advance_fn,
            )
            for row in records
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    for article_id, status, sha256 in results:
        update_fn(article_id, status, sha256)


def run_download_images(settings: Settings | None = None) -> None:
    """
    Orchestrate the download of pending images, updating the database with
    results and progress.
    """
    if settings is None:
        settings = get_settings()

    settings.ensure_dirs()

    with get_conn(settings.state_db) as conn:
        records = get_pending_image_records(conn)

    if not records:
        print("No pending images to download.")
        return

    print(f"Downloading {len(records)} images (concurrency={settings.image_download_concurrency})…")

    results_buffer: list[tuple[str, str, str | None]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Downloading images…", total=len(records))

        def advance():
            """Advance the progress bar."""
            progress.advance(task)

        def collect(article_id, status, sha256):
            """Collect results into the local buffer."""
            results_buffer.append((article_id, status, sha256))

        asyncio.run(
            _run_async(
                list(records),
                settings.images_dir,
                settings.image_max_side_px,
                settings.image_download_concurrency,
                collect,
                advance,
            )
        )

    # Flush to DB in batches
    ok = failed_404 = failed_other = 0
    batch_size = 200
    for i in range(0, len(results_buffer), batch_size):
        chunk = results_buffer[i : i + batch_size]
        with get_conn(settings.state_db) as conn:
            for article_id, status, sha256 in chunk:
                set_image_status(conn, article_id, status, sha256)
                if status == "ok":
                    ok += 1
                elif status == "failed_404":
                    failed_404 += 1
                else:
                    failed_other += 1

    print(
        f"Image download complete. ok={ok}, failed_404={failed_404}, failed_other={failed_other}"
    )
