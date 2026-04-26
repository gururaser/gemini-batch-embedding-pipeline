import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from gme.state import (
    get_conn,
    get_counts,
    init_db,
    insert_records,
    mark_embed_ok,
    mark_upsert_ok,
    set_image_status,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    init_db(path)
    return path


def test_insert_records(db_path: Path) -> None:
    records = [
        {
            "article_id": "A001",
            "point_uuid": "00000000-0000-0000-0000-000000000001",
            "image_url": "https://example.com/a.jpg",
            "payload_json": json.dumps({"prod_name": "Shirt"}),
        },
        {
            "article_id": "A002",
            "point_uuid": "00000000-0000-0000-0000-000000000002",
            "image_url": "https://example.com/b.jpg",
            "payload_json": json.dumps({"prod_name": "Pants"}),
        },
    ]
    with get_conn(db_path) as conn:
        insert_records(conn, records)

    with get_conn(db_path) as conn:
        counts = get_counts(conn)

    assert counts["total"] == 2
    assert counts["image_ok"] == 0


def test_insert_idempotent(db_path: Path) -> None:
    rec = [
        {
            "article_id": "A001",
            "point_uuid": "00000000-0000-0000-0000-000000000001",
            "image_url": "https://example.com/a.jpg",
            "payload_json": "{}",
        }
    ]
    with get_conn(db_path) as conn:
        insert_records(conn, rec)
        insert_records(conn, rec)  # duplicate

    with get_conn(db_path) as conn:
        counts = get_counts(conn)

    assert counts["total"] == 1


def test_image_status_flow(db_path: Path) -> None:
    with get_conn(db_path) as conn:
        insert_records(
            conn,
            [{"article_id": "A001", "point_uuid": "uuid1", "image_url": "u", "payload_json": "{}"}],
        )
        set_image_status(conn, "A001", "ok", sha256="abc123")

    with get_conn(db_path) as conn:
        counts = get_counts(conn)

    assert counts["image_ok"] == 1


def test_embed_and_upsert_flow(db_path: Path) -> None:
    with get_conn(db_path) as conn:
        insert_records(
            conn,
            [{"article_id": "A001", "point_uuid": "uuid1", "image_url": "u", "payload_json": "{}"}],
        )
        set_image_status(conn, "A001", "ok", sha256="abc123")
        mark_embed_ok(conn, "A001")

    with get_conn(db_path) as conn:
        counts = get_counts(conn)
    assert counts["embed_ok"] == 1

    with get_conn(db_path) as conn:
        mark_upsert_ok(conn, ["A001"])

    with get_conn(db_path) as conn:
        counts = get_counts(conn)
    assert counts["upserted"] == 1
