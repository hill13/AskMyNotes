"""Shared DB fixture. Tests run against a SEPARATE database, because they
TRUNCATE -- pointed at the real one, they would wipe ingested notes."""
from __future__ import annotations

import os

import pytest

from askmynotes import db

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://askmynotes:askmynotes@localhost:5434/askmynotes_test",
)


@pytest.fixture(scope="module")
def conn():
    try:
        ctx = db.connect(TEST_DATABASE_URL)
        c = ctx.__enter__()
    except Exception as exc:
        pytest.skip(f"test database not available: {exc}")

    # Guard OUTSIDE the try: a misconfigured URL must fail loudly, not be
    # swallowed into a skip that looks like "Docker is down".
    name = c.execute("SELECT current_database()").fetchone()[0]
    if not name.endswith("_test"):
        ctx.__exit__(None, None, None)
        pytest.fail(f"refusing to run destructive tests against '{name}'")

    try:
        db.init_schema(c)
        yield c
    finally:
        ctx.__exit__(None, None, None)
