"""Naive Bayes training endpoint. Owner: D (docs/PROJECT_PLAN.md, Section 7D)."""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/train", tags=["Naive Bayes"])


@router.post("/naive_bayes")
def train_naive_bayes():
    raise HTTPException(status_code=501, detail="Not implemented yet: POST /train/naive_bayes (Member D)")
