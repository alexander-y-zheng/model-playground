"""KNN training endpoint. Owner: B (docs/PROJECT_PLAN.md, Section 7B)."""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/train", tags=["KNN"])


@router.post("/knn")
def train_knn():
    raise HTTPException(status_code=501, detail="Not implemented yet: POST /train/knn (Member B)")
