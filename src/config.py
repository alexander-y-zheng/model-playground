"""Shared paths and settings.

Paths are read through functions (not module-level constants) so tests can
point the app at a temporary database with an environment variable.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_CSV_PATH = DATA_DIR / "diabetes_sample.csv"

DB_PATH_ENV_VAR = "MODEL_PLAYGROUND_DB"
DEFAULT_DB_PATH = DATA_DIR / "app.db"

TARGET_COLUMN = "Diabetes_binary"


def get_db_path() -> Path:
    """Return the SQLite path, overridable with MODEL_PLAYGROUND_DB."""
    return Path(os.getenv(DB_PATH_ENV_VAR, str(DEFAULT_DB_PATH)))
