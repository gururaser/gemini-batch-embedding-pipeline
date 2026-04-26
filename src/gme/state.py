import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

CREATE_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS records (
    article_id      TEXT PRIMARY KEY,
    point_uuid      TEXT NOT NULL,
    image_url       TEXT,
    image_sha256    TEXT,
    image_status    TEXT NOT NULL DEFAULT 'pending',
    embed_status    TEXT NOT NULL DEFAULT 'pending',
    embed_attempts  INTEGER NOT NULL DEFAULT 0,
    batch_id        TEXT,
    shard_id        INTEGER,
    upsert_status   TEXT NOT NULL DEFAULT 'pending',
    payload_json    TEXT,
    updated_at      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS ix_embed   ON records(embed_status);
CREATE INDEX IF NOT EXISTS ix_upsert  ON records(upsert_status);
CREATE INDEX IF NOT EXISTS ix_batch   ON records(batch_id);
CREATE INDEX IF NOT EXISTS ix_image   ON records(image_status);

CREATE TABLE IF NOT EXISTS batches (
    batch_id        TEXT PRIMARY KEY,
    shard_id        INTEGER UNIQUE,
    state           TEXT NOT NULL DEFAULT 'PENDING',
    est_tokens      INTEGER NOT NULL DEFAULT 0,
    record_count    INTEGER NOT NULL DEFAULT 0,
    request_file    TEXT,
    result_file     TEXT,
    submitted_at    INTEGER,
    finished_at     INTEGER
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(CREATE_SQL)


@contextmanager
def get_conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def now() -> int:
    return int(time.time())


# ── records helpers ──────────────────────────────────────────────────────────

def insert_records(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    ts = now()
    conn.executemany(
        """
        INSERT OR IGNORE INTO records
            (article_id, point_uuid, image_url, payload_json, updated_at)
        VALUES
            (:article_id, :point_uuid, :image_url, :payload_json, :updated_at)
        """,
        [{**r, "updated_at": ts} for r in rows],
    )
    return conn.rowcount


def set_image_status(
    conn: sqlite3.Connection,
    article_id: str,
    status: str,
    sha256: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE records
        SET image_status = ?, image_sha256 = ?, updated_at = ?
        WHERE article_id = ?
        """,
        (status, sha256, now(), article_id),
    )


def get_pending_image_records(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT article_id, image_url FROM records WHERE image_status = 'pending'"
    ).fetchall()


def get_embeddable_records(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT article_id, point_uuid, image_url, image_sha256, payload_json
        FROM records
        WHERE embed_status = 'pending'
          AND image_status = 'ok'
          AND embed_attempts < 3
        ORDER BY article_id
        """
    ).fetchall()


def assign_to_shard(
    conn: sqlite3.Connection, article_ids: list[str], shard_id: int
) -> None:
    ts = now()
    conn.executemany(
        "UPDATE records SET shard_id = ?, embed_status = 'in_batch', updated_at = ? WHERE article_id = ?",
        [(shard_id, ts, aid) for aid in article_ids],
    )


def get_pending_shards(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT shard_id, est_tokens, record_count, request_file FROM batches WHERE state = 'PENDING'"
    ).fetchall()


def get_active_batches(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT batch_id, shard_id, est_tokens, state FROM batches WHERE state IN ('RUNNING', 'SUBMITTED')"
    ).fetchall()


def get_succeeded_batches(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT batch_id, shard_id, result_file FROM batches WHERE state = 'SUCCEEDED'"
    ).fetchall()


def insert_batch(conn: sqlite3.Connection, shard_id: int, est_tokens: int, record_count: int, request_file: str) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO batches (shard_id, est_tokens, record_count, request_file, state)
        VALUES (?, ?, ?, ?, 'PENDING')
        """,
        (shard_id, est_tokens, record_count, request_file),
    )


def update_batch_submitted(conn: sqlite3.Connection, shard_id: int, batch_id: str) -> None:
    conn.execute(
        """
        UPDATE batches SET batch_id = ?, state = 'SUBMITTED', submitted_at = ?
        WHERE shard_id = ?
        """,
        (batch_id, now(), shard_id),
    )
    conn.execute(
        "UPDATE records SET batch_id = ? WHERE shard_id = ?",
        (batch_id, shard_id),
    )


def update_batch_state(
    conn: sqlite3.Connection, batch_id: str, state: str, result_file: str | None = None
) -> None:
    ts = now()
    conn.execute(
        "UPDATE batches SET state = ?, result_file = ?, finished_at = ? WHERE batch_id = ?",
        (state, result_file, ts, batch_id),
    )
    if state in ("FAILED", "EXPIRED"):
        conn.execute(
            """
            UPDATE records
            SET embed_status = 'pending', batch_id = NULL, shard_id = NULL,
                embed_attempts = embed_attempts + 1, updated_at = ?
            WHERE batch_id = ?
            """,
            (ts, batch_id),
        )


def mark_embed_ok(conn: sqlite3.Connection, article_id: str) -> None:
    conn.execute(
        "UPDATE records SET embed_status = 'ok', updated_at = ? WHERE article_id = ?",
        (now(), article_id),
    )


def mark_embed_failed(conn: sqlite3.Connection, article_id: str) -> None:
    conn.execute(
        """
        UPDATE records
        SET embed_status = 'failed', embed_attempts = embed_attempts + 1, updated_at = ?
        WHERE article_id = ?
        """,
        (now(), article_id),
    )


def get_upsert_pending(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT article_id, point_uuid, payload_json
        FROM records
        WHERE upsert_status = 'pending' AND embed_status = 'ok'
        """
    ).fetchall()


def mark_upsert_ok(conn: sqlite3.Connection, article_ids: list[str]) -> None:
    conn.executemany(
        "UPDATE records SET upsert_status = 'ok', updated_at = ? WHERE article_id = ?",
        [(now(), aid) for aid in article_ids],
    )


def get_counts(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(image_status = 'ok') AS image_ok,
            SUM(image_status LIKE 'failed%') AS image_failed,
            SUM(embed_status = 'ok') AS embed_ok,
            SUM(embed_status = 'failed') AS embed_failed,
            SUM(upsert_status = 'ok') AS upserted
        FROM records
        """
    ).fetchone()
    return dict(row)
