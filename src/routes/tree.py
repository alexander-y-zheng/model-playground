"""Decision tree training endpoint. Owner: C (docs/PROJECT_PLAN.md, Section 7C)."""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/train", tags=["Decision Tree"])


@router.post("/tree")
def train_tree():
    raise HTTPException(status_code=501, detail="Not implemented yet: POST /train/tree (Member C)")
