"""Shared pytest fixtures.

Every test gets its own temporary SQLite file, so tests never touch data/app.db.
B: once scripts/load_data.py exists, add a `seeded_db` fixture here that loads
fixtures/sample_200.csv into the temporary database (docs/PROJECT_PLAN.md, Section 6).
"""

import pytest
from fastapi.testclient import TestClient

import main
from src.config import DB_PATH_ENV_VAR


@pytest.fixture
def test_db_path(tmp_path, monkeypatch):
    """Point the app at a fresh temporary database for this test."""
    db_path = tmp_path / "test_app.db"
    monkeypatch.setenv(DB_PATH_ENV_VAR, str(db_path))
    return db_path


@pytest.fixture
def client(test_db_path):
    """A FastAPI TestClient that uses the temporary database."""
    with TestClient(main.app) as test_client:
        yield test_client
