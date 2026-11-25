"""Training run history endpoints. Owner: C (docs/PROJECT_PLAN.md, Section 6)."""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/runs", tags=["Runs"])


@router.get("")
def list_runs():
    """Past training runs, newest first, optionally filtered by model type."""
    raise HTTPException(status_code=501, detail="Not implemented yet: GET /runs (Member C)")


# Keep /compare above /{run_id}; otherwise FastAPI tries to parse "compare" as a run_id.
@router.get("/compare")
def compare_runs():
    """Side-by-side metrics for several runs (stretch: Compare Models page)."""
    raise HTTPException(status_code=501, detail="Not implemented yet: GET /runs/compare (Member C)")


@router.get("/{run_id}")
def get_run(run_id: int):
    """One run's full settings and metrics."""
    raise HTTPException(status_code=501, detail="Not implemented yet: GET /runs/{run_id} (Member C)")
