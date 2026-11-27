"""HTTP client for the FastAPI backend.

Owner: C, who adds get_features(), post_train() and get_runs()
(docs/PROJECT_PLAN.md, Section 6). A added get_health() for the Home page.
"""

import os

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 30


def get_health() -> dict | None:
    """Return the API's /health response, or None if the API isn't reachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None
