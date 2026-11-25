"""Dataset endpoints. Owner: B (docs/PROJECT_PLAN.md, Section 6)."""

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Dataset"])


@router.get("/features")
def list_features():
    """List the 21 input features with label, description and kind."""
    raise HTTPException(status_code=501, detail="Not implemented yet: GET /features (Member B)")


@router.get("/dataset/summary")
def dataset_summary():
    """Row count, class balance and per-feature mean by class."""
    raise HTTPException(status_code=501, detail="Not implemented yet: GET /dataset/summary (Member B)")
