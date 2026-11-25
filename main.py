"""FastAPI entry point. Run with: uvicorn main:app --reload"""

import logging

from fastapi import FastAPI

from src.middleware import log_requests
from src.routes import data, knn, logistic, naive_bayes, runs, tree
from src.utils import setup_logging

setup_logging(verbose=False)
logger = logging.getLogger(__name__)

API_VERSION = "0.1.0"

app = FastAPI(
    title="Model Playground API",
    description="Train and inspect classification models on CDC diabetes survey data.",
    version=API_VERSION,
)
app.middleware("http")(log_requests)


@app.get("/health", tags=["Health"])
def health_check():
    """Return the application status and version."""
    return {"status": "ok", "version": API_VERSION}


for module in (data, runs, logistic, knn, tree, naive_bayes):
    app.include_router(module.router)
