"""Streamlit entry point. Run with: streamlit run streamlit_app/Home.py"""

import streamlit as st

from components.api_client import API_URL, get_health

st.set_page_config(page_title="Model Playground", page_icon="🧪", layout="wide")

st.title("🧪 Model Playground")
st.markdown(
    "Learn how four classic classification models work by training them yourself. "
    "Pick features and hyperparameters, train a model on CDC diabetes survey data, "
    "and see how it performs and *why* it behaves the way it does."
)

health = get_health()
if health:
    st.success(f"API connected ({API_URL}, version {health['version']})")
else:
    st.error(
        f"Can't reach the API at {API_URL}. "
        "Start it in another terminal with `uvicorn main:app --reload`."
    )

st.subheader("Models")
st.markdown(
    """
| Page | What you'll learn |
|---|---|
| **Logistic Regression** | How feature weights, regularization and the decision threshold work |
| **KNN** | How *k* trades overfitting for smoothness, and why scaling matters |
| **Decision Tree** | How a tree splits the data, and how depth leads to overfitting |
| **Naive Bayes** | How priors and per-class likelihoods combine into a prediction |

Choose a model from the sidebar to start.
"""
)

st.caption(
    "Educational tool only. Predictions are not medical advice. "
    "Data: CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015."
)

# TODO (A/B): dataset summary from GET /dataset/summary once B's endpoint is merged.
