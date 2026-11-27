"""SQLite helpers shared by the whole app.

Owners (see docs/PROJECT_PLAN.md, Section 6):
- B adds table creation for `diabetes_sample` and `features`, plus `load_xy(features)`.
- D adds table creation for `training_runs`, plus `save_run(...)`.
"""

import sqlite3
from pathlib import Path

from src.config import get_db_path


def get_connection(db_path=None) -> sqlite3.Connection:
    """Return a fresh connection with rows accessible by column name."""
    db_path = Path(db_path) if db_path is not None else get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection
