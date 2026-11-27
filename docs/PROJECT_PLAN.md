# Model Playground: Project Plan

> **One-line pitch:** An interactive tool that teaches students how four traditional classification models work. Users choose features and hyperparameters, train each model live on diabetes survey data, and see how it performs and *why* it behaves that way.

**Repo:** `model-playground` · **Track:** ML Model Integration · **Team size:** 4 · **Stack:** Streamlit → FastAPI → scikit-learn + SQLite

| Member | Model page they own | Shared foundation task |
|---|---|---|
| **A** — Alex Zheng | Logistic Regression | Repo setup and app skeletons + shared decision-boundary helper and chart |
| **B** — David Alberino | K-Nearest Neighbors | Dataset sample + feature metadata |
| **C** — Megan Anderson | Decision Tree | Shared Streamlit components + run history endpoints (+ Compare page stretch) |
| **D** — Joel Guerra | Naive Bayes | Schemas, shared training pipeline, cache |

> **Assign D to whoever is most available over Thanksgiving.** D's schemas and `run_training()` are on the critical path: every model page depends on them.

---

## 1. How the app works

```
 Streamlit page (one per model)
   │  user picks features + hyperparameters, clicks "Train"
   │  POST /train/<model_type>   (JSON)
   ▼
 FastAPI
   ├─ validates the request (Pydantic)            → 422 if invalid
   ├─ checks the training cache                    → instant if same request seen before
   ├─ loads the dataset sample from SQLite
   ├─ splits into train/test, fits the model, evaluates it
   ├─ builds model-specific "explain" data (coefficients, tree, k-curve, …)
   ├─ computes a 2-D decision boundary if exactly 2 non-binary features are chosen
   ├─ saves the run to SQLite (training_runs table)
   └─ returns one JSON response (same shape for every model)
   ▼
 Streamlit renders it
   ├─ shared panels: metrics, confusion matrix, ROC curve, decision boundary   (same on every page)
   └─ model-specific charts: the actual lesson                                 (owned by each member)
```

Models are **trained on demand** for each request, so we do **not** need saved `.joblib` model files. Each run's settings and results are stored in SQLite.

---

## 2. Why Naive Bayes instead of SVM

| | Naive Bayes | SVM |
|---|---|---|
| Training time (4k rows) | Milliseconds | ~1–3 s with RBF kernel; probability output (`probability=True`) adds an internal 5-fold CV, which makes it several times slower |
| Feature scaling needed | No | Yes, and results depend heavily on it |
| Hyperparameter sensitivity | Low (always gives a reasonable result) | High (bad `C`/`gamma` combinations can hang or give useless results) |
| Implementation | `GaussianNB` / `BernoulliNB`, a few lines | Needs scaling, probability calibration, careful parameter bounds |
| Teaching value | Strong: priors, likelihoods, and the "naive" independence assumption are easy to visualize | Strong, but harder to show without 2-D plots |

The interactive app needs training to finish quickly so pages feel responsive, which makes Naive Bayes the safer choice.

---

## 3. Dataset

**Source:** CDC Diabetes Health Indicators (BRFSS 2015), via UCI ML Repository (`ucimlrepo`, dataset id `891`). Also on Kaggle as "Diabetes Health Indicators Dataset."

**Target:** `Diabetes_binary` → 0 = no diabetes, 1 = prediabetes or diabetes.

**Shortened sample (what the app uses):**
1. Download the full dataset (~253k rows).
2. Clean it with our MP1 `data_pipeline` functions: drop duplicate rows (~24k), check value ranges, cast floats to ints.
3. Take a **balanced random sample of 5,000 rows**: 2,500 per class, `random_state=42`.
4. Save the sample as `data/diabetes_sample.csv`. **Commit this file**: it's only ~250 KB, and committing it means nobody (including graders) has to download anything.
5. `scripts/load_data.py` loads the CSV into SQLite (`data/app.db`, gitignored).

Why a balanced sample: it keeps training fast (well under 1 s for every model), and accuracy is meaningful when the classes are 50/50. In the real population only ~14% are positive. The Naive Bayes prior slider teaches this point (Section 7D).

### The 21 input features

| Kind | Columns | Values |
|---|---|---|
| **Binary** (0/1) | `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex` | 0 = no, 1 = yes (`Sex`: 0 = female, 1 = male) |
| **Ordinal** | `GenHlth` (1 = excellent … 5 = poor), `Age` (1 = 18–24 … 13 = 80+), `Education` (1–6), `Income` (1 = <$10k … 8 = ≥$75k) | integers |
| **Count** | `MentHlth`, `PhysHlth` | days in past 30 (0–30) |
| **Continuous** | `BMI` | ~12–98 |

The `kind` column (binary / ordinal / count / continuous) is stored in the `features` table and used for validation. For example, Bernoulli NB only accepts binary features, and the decision-boundary chart needs two non-binary features.

---

## 4. SQLite schema

```sql
-- The 5,000-row sample
CREATE TABLE diabetes_sample (
    id INTEGER PRIMARY KEY,
    HighBP INTEGER, HighChol INTEGER, ... , Income INTEGER,   -- all 21 features
    Diabetes_binary INTEGER NOT NULL
);

-- Human-readable info for the feature picker
CREATE TABLE features (
    name TEXT PRIMARY KEY,        -- "BMI"
    label TEXT NOT NULL,          -- "Body Mass Index"
    description TEXT NOT NULL,    -- "Computed from height and weight"
    kind TEXT NOT NULL,           -- binary | ordinal | count | continuous
    min_value REAL, max_value REAL
);

-- Every training request
CREATE TABLE training_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,     -- logistic | knn | tree | naive_bayes
    features_json TEXT NOT NULL,
    params_json TEXT NOT NULL,
    test_size REAL NOT NULL,
    random_state INTEGER NOT NULL,
    metrics_json TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    created_at TEXT NOT NULL      -- ISO timestamp
);
```

---

## 5. API contract (everyone codes against this)

### Endpoints

| Method | Path | Owner | Purpose |
|---|---|---|---|
| GET | `/health` | A | Health check |
| GET | `/features` | B | List of features with label, description, kind |
| GET | `/dataset/summary` | B | Row count, class balance, per-feature mean by class (for the Home page) |
| POST | `/train/logistic` | A | Train logistic regression |
| POST | `/train/knn` | B | Train KNN |
| POST | `/train/tree` | C | Train decision tree |
| POST | `/train/naive_bayes` | D | Train Naive Bayes |
| GET | `/runs?model_type=&limit=` | C | Training run history |
| GET | `/runs/{run_id}` | C | One run's details |

### Request (shared base + model-specific `params`)

```json
POST /train/knn
{
  "features": ["BMI", "Age", "HighBP"],
  "test_size": 0.2,
  "random_state": 42,
  "params": { "n_neighbors": 5, "metric": "euclidean", "weights": "uniform", "scale": true }
}
```

```python
# src/schemas.py: base classes (D writes this first; everyone imports it)
class TrainRequestBase(BaseModel):
    features: list[str] = Field(min_length=1)          # validator: known names, no duplicates
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    random_state: int = Field(42, ge=0)

class Metrics(BaseModel):
    accuracy: float          # test set
    train_accuracy: float    # for spotting overfitting on every page
    precision: float
    recall: float
    f1: float
    roc_auc: float

class TrainResponse(BaseModel):
    run_id: int
    model_type: str
    features: list[str]
    params: dict
    n_train: int
    n_test: int
    metrics: Metrics
    confusion_matrix: list[list[int]]      # [[TN, FP], [FN, TP]]
    roc_curve: dict                        # {"fpr": [...], "tpr": [...]}, ≤100 points
    duration_ms: float
    cached: bool
    decision_boundary: dict | None         # shared, see Section 6 (A); null unless exactly 2 non-binary features
    model_details: dict                    # model-specific, defined in Section 7
```

### Errors
| Situation | Status | Example `detail` |
|---|---|---|
| Missing/invalid field, out-of-range param, unknown feature | **422** (Pydantic) | `"n_neighbors must be between 1 and 50"` |
| Valid JSON but impossible request | **400** | `"Bernoulli Naive Bayes only accepts binary features; remove: BMI, Age"` |
| Unexpected server error | **500** | logged with our MP2 middleware |

The positive class is always `1` (diabetes). Precision, recall and ROC use `predict_proba[:, 1]`.

---

## 6. Shared foundation (build first)

**Goal:** merged into `main` by **Sun 11/29** (hard deadline: start of class **Tue 12/1**). While foundation PRs are in review, model owners can already write their model code as plain functions (see Section 7), because the contract above is fixed.

### A: Repo setup (`setup/project-structure`), merge this one first
- Folder structure (Section 9), `requirements.txt`, `.gitignore` (`data/app.db`, `.venv/`, `__pycache__/`)
- `main.py` FastAPI app with all routers registered (empty stubs returning 501 are fine), MP2 logging middleware
- `streamlit_app/Home.py` + empty page files for each model
- `tests/conftest.py`: a temporary SQLite DB seeded with ~200 rows from `fixtures/sample_200.csv`, and a FastAPI `TestClient` fixture
- README skeleton with the run commands

### A (part 2): Shared decision boundary (`feature/decision-boundary`), after setup merges
Works for **any** fitted model with `predict_proba`, so every page gets one. Comparing them side by side is a key lesson: logistic regression draws a straight line, KNN a jagged one, the tree draws boxes, and Naive Bayes draws curves.
- `src/ml/boundary.py`:
  ```python
  def compute_decision_boundary(model, X_train, y_train, features, feature_kinds) -> dict | None:
      # Returns None unless exactly 2 features are selected and both are non-binary.
      # Builds a 60×60 grid over each feature's range (original units),
      # predicts P(diabetes) on the grid, and samples ≤400 training points for the scatter.
  ```
  Output (goes in `TrainResponse.decision_boundary`):
  ```json
  { "x_feature": "BMI", "y_feature": "Age",
    "x": [...], "y": [...],            // 60 grid values each
    "z": [[...], ...],                 // 60×60 predicted P(diabetes)
    "points": [{"x": 31, "y": 9, "label": 1}, ...] }
  ```
  D calls this inside `run_training()`, so model owners don't need to do anything.
- `streamlit_app/components/boundary_chart.py`: `decision_boundary_chart(boundary)`. It draws a Plotly contour with scatter points on top. If `boundary` is `null`, it shows "Select exactly 2 non-binary features (e.g., BMI + Age) to see the decision boundary."
- **Tests (`tests/test_boundary.py`):** 2 non-binary features → 60×60 grid with values in [0, 1] · 3 features → `None` · a binary feature (`HighBP`) → `None` · works with all 4 model types

### B: Data (`data/load-sample`)
- `scripts/make_sample.py`: download from UCI → clean with MP1 functions → balanced 5k sample → `data/diabetes_sample.csv` (run once; CSV is committed)
- `scripts/load_data.py`: CSV → `diabetes_sample` table; feature metadata → `features` table
- `GET /features`, `GET /dataset/summary`
- Data access helper in `src/database.py`: `load_xy(features) -> (X: DataFrame, y: Series)`

### C: Shared Streamlit components (`setup/ui-components`)
In `streamlit_app/components/`:
- `api_client.py`: `get_features()`, `post_train(model_type, payload)`, `get_runs()`. Reads `API_URL` (default `http://localhost:8000`). Shows `st.error` with the API's `detail` for 4xx/5xx errors and a friendly message if the API isn't running.
- `feature_picker(default)`: multiselect showing labels and descriptions from `/features`
- `split_settings()`: test-size slider + random-seed input
- `metrics_panel(metrics)`: `st.metric` row, including train vs. test accuracy
- `confusion_matrix_chart(cm)` and `roc_chart(roc_curve, auc)`, built with Plotly
- `model_intro(title, explanation, try_this: list[str])`: the "what this model does" box with "Try this" prompts

**Important Streamlit detail:** store the latest training response in `st.session_state[f"{model_type}_result"]`. Streamlit reruns the whole script on every widget change, so any slider that only changes the chart (like the logistic threshold slider) must read from session state instead of calling the API again.

### C (part 2): Run history endpoints (`feature/run-history`), after D's pipeline merges
- `src/routes/runs.py`:
  - `GET /runs?model_type=&limit=`: newest first, optional filter by model type, `limit` 1–100 (default 20)
  - `GET /runs/{run_id}`: one run's full settings and metrics, 404 if it doesn't exist
- `get_runs()` in `api_client.py` (already listed above)
- **Tests (`tests/test_runs.py`):** after training, the run appears in `/runs` · `model_type` filter returns only matching runs · unknown `run_id` → 404 · invalid `model_type` → 422
- These endpoints are what the stretch **Compare Models** page reads, so C owns that page too (Section 7).

### D: Schemas, training pipeline, cache (`setup/training-pipeline`)
- `src/schemas.py`: the base classes in Section 5. Open this as its **own small PR first** (day 1), so nobody waits for the bigger pipeline PR.
- `src/ml/common.py`:
  ```python
  def run_training(model_type, request, build_model, explain) -> TrainResponse:
      # 1. cache lookup        (key built from the full request; see below)
      # 2. X, y = load_xy(request.features)
      # 3. stratified train_test_split(test_size, random_state)
      # 4. model = build_model(); model.fit(X_train, y_train)
      # 5. metrics, confusion matrix, ROC (downsampled to ≤100 points)
      # 6. details = explain(model, data)       ← each member's function
      # 7. boundary = compute_decision_boundary(...)   ← A's helper (use None until it merges)
      # 8. save to training_runs, store in cache, return TrainResponse
  ```
  Each model owner only writes `build_model()` and `explain()`.
- `src/cache.py`: reuse our MP3 fakeredis cache pattern
- `save_run(...)` helper in `src/database.py` that inserts into `training_runs` (C's `/runs` endpoints read this table)
- **Tests (`tests/test_cache.py`):** the same request twice → second response has `cached: true` · changing any param or feature → cache miss · bumping `dataset_version` → cache miss

### Caching (explain this in the README and the presentation)
- **What:** the full `TrainResponse` for a training request.
- **Key:** `train:{model_type}:{sha256(sorted features, sorted params, test_size, random_state, dataset_version)}`. Every input that changes the result is in the key.
- **Why:** students often re-click "Train" with the same settings or flip between settings they've tried. A cached response comes back instantly.
- **Expiry/invalidation:** TTL of 1 hour. `dataset_version` is stored in SQLite and bumped whenever `load_data.py` reloads the data, so old cached results stop matching.
- Cached responses return `"cached": true` so the UI can show a small "⚡ cached" badge.

---

## 7. Individual model features

Each member owns their model **end to end**: the endpoint, `build_model()`, `explain()`, the Streamlit page, its charts, and its tests. Write `build_model` and `explain` in `src/ml/<model>.py` and the route in `src/routes/<model>.py`.

### 7A. Logistic Regression: Member A (`feature/logistic-regression`)

**Lesson:** a linear model gives each feature a weight. Regularization shrinks weights, and the decision threshold trades precision against recall.

| Param | Widget | Range / options | Default |
|---|---|---|---|
| `C` (inverse regularization) | log slider | 0.001 – 100 | 1.0 |
| `penalty` | radio | `l1`, `l2` | `l2` |

- **Model:** `Pipeline([StandardScaler(), LogisticRegression(C, penalty, solver="liblinear")])`. Scaling makes the coefficients comparable across features.
- **`model_details`:**
  ```json
  { "intercept": -0.12,
    "coefficients": [{"feature": "HighBP", "value": 0.71}, ...],         // sorted by |value|
    "threshold_curve": [{"threshold": 0.05, "precision": 0.52, "recall": 0.99}, ...] }  // 0.05..0.95 step 0.05
  ```
- **Charts:**
  1. Horizontal coefficient bar chart (red = raises risk, blue = lowers risk)
  2. **Threshold slider** (outside the form, reads from session state): moves a marker on a precision/recall-vs-threshold line chart and updates the confusion matrix counts
- **Try this:** "Set penalty = L1 and C = 0.01: which coefficients become exactly 0?" · "Slide the threshold to 0.3: what happens to recall?" · "Pick BMI + Age: notice the decision boundary is a straight line."
- **Tests:** `C = 0` → 422 · response contains one coefficient per selected feature · L1 with very small C produces at least one zero coefficient

### 7B. K-Nearest Neighbors: Member B (`feature/knn`)

**Lesson:** KNN predicts by majority vote of the *k* closest training points. Small *k* overfits, large *k* oversmooths, and unscaled features distort distance.

| Param | Widget | Range / options | Default |
|---|---|---|---|
| `n_neighbors` | slider | 1 – 50 | 5 |
| `metric` | radio | `euclidean`, `manhattan` | `euclidean` |
| `weights` | radio | `uniform`, `distance` | `uniform` |
| `scale` | checkbox | true / false | true |

- **Model:** `Pipeline([StandardScaler() if scale else "passthrough", KNeighborsClassifier(...)])`
- **`model_details`:**
  ```json
  { "k_curve": [{"k": 1, "train_accuracy": 0.99, "test_accuracy": 0.66}, ...],     // k = 1,3,5,...,49
    "scaled": true,
    "feature_ranges": [{"feature": "BMI", "min": 13, "max": 79}, {"feature": "HighBP", "min": 0, "max": 1}, ...] }
  ```
  The decision boundary comes from A's shared helper (top-level `decision_boundary` field), so KNN doesn't compute it.
- **Charts:**
  1. Train vs. test accuracy over *k* (with the chosen *k* marked)
  2. Feature-range bar chart ("why scaling matters"): unscaled, BMI's range (~65) dwarfs a 0/1 feature, so BMI dominates the distance
  3. Shared decision boundary chart: KNN's lesson is how the boundary gets smoother as *k* grows
- **Try this:** "Set k = 1: compare train vs. test accuracy." · "Pick BMI + Age, set k = 1, then k = 49: how does the boundary change?" · "Uncheck *scale* and add `HighBP`: what happens to accuracy?"
- **Tests:** `n_neighbors = 0` → 422 · `metric = "cosine"` → 422 · `k_curve` has 25 entries and at k = 1 train accuracy ≥ test accuracy
- **Performance note:** the k-curve is ~25 predictions over the test and train sets (4k rows), about 1 s. Fine, and the cache makes repeat requests instant.

### 7C. Decision Tree: Member C (`feature/decision-tree`)

**Lesson:** a tree asks a sequence of yes/no questions. Deeper trees memorize the training data (overfitting), and the tree's structure is fully readable.

| Param | Widget | Range / options | Default |
|---|---|---|---|
| `max_depth` | slider | 1 – 20 | 4 |
| `min_samples_leaf` | slider | 1 – 200 | 1 |
| `criterion` | radio | `gini`, `entropy` | `gini` |

- **Model:** `DecisionTreeClassifier(max_depth, min_samples_leaf, criterion, random_state)` (no scaling needed)
- **`model_details`:**
  ```json
  { "tree_dot": "digraph Tree { ... }",     // sklearn export_graphviz(..., max_depth=4, filled=True)
    "n_leaves": 14, "actual_depth": 4,
    "feature_importances": [{"feature": "GenHlth", "value": 0.38}, ...],
    "depth_curve": [{"depth": 1, "train_accuracy": 0.71, "test_accuracy": 0.70}, ...] }   // depths 1–20
  ```
- **Charts:**
  1. Tree diagram via `st.graphviz_chart(tree_dot)`. This renders a DOT string directly, with no Graphviz install needed. Display depth is capped at 4 so the diagram stays readable.
  2. Train vs. test accuracy across depths (the overfitting gap)
  3. Feature importance bar chart
- **Try this:** "Set max_depth = 20: why is train accuracy almost 100% but test accuracy lower?" · "Which feature does the tree split on first?" · "Pick BMI + Age: why is the decision boundary made of rectangles?"
- **Tests:** `max_depth = 0` → 422 · `max_depth = 1` → `n_leaves == 2` · importances sum to ~1

### 7D. Naive Bayes: Member D (`feature/naive-bayes`)

**Lesson:** Naive Bayes combines a *prior* (how common diabetes is) with *likelihoods* (how each feature looks within each class), and "naively" assumes the features are independent.

| Param | Widget | Range / options | Default |
|---|---|---|---|
| `variant` | radio | `gaussian`, `bernoulli` | `gaussian` |
| `var_smoothing` (Gaussian) | log slider | 1e-12 – 1e-1 | 1e-9 |
| `alpha` (Bernoulli) | slider | 0.01 – 10 | 1.0 |
| `prior_diabetes` | slider + "learn from data" checkbox | 0.01 – 0.99, or `null` | `null` (learned ≈ 0.5) |

- **Model:** `GaussianNB(var_smoothing, priors=[1-p, p])` or `BernoulliNB(alpha, class_prior=[1-p, p])`
- **Validation:** Bernoulli + any non-binary feature → **400** with the list of features to remove. The page's feature picker can also pre-filter to binary features when Bernoulli is selected.
- **`model_details`:**
  ```json
  { "class_prior": [0.5, 0.5],
    "gaussian": {"BMI": {"mean": [27.1, 31.9], "var": [30.2, 45.8]}, ...} | null,
    "bernoulli": [{"feature": "HighBP", "p_given_no": 0.38, "p_given_yes": 0.75}, ...] | null }
  ```
- **Charts:**
  1. **Gaussian:** pick a feature → two bell curves (no diabetes vs. diabetes) from the learned mean and variance
  2. **Bernoulli:** grouped bars of P(feature = 1 | class) for each feature
  3. Prior slider: the effect on the confusion matrix and recall
- **Try this:** "Set the prior to 0.14 (the real US rate): how does recall change, and why?" · "Gaussian NB treats `HighBP` (0/1) as a bell curve. Is that reasonable? Compare with Bernoulli." · "Pick BMI + Age (Gaussian): why is the boundary curved?"
- **Tests:** Bernoulli with `BMI` → 400 · `prior_diabetes = 1.5` → 422 · a higher prior produces at least as many positive predictions as a lower prior

### Stretch: Compare Models page: Member C (`feature/compare-models`, 12/5–12/7)
C owns this because it's built on C's `/runs` endpoints and C's Decision Tree page is the lightest model page. Others can help if they finish early.
- `GET /runs/compare?ids=3,7,12` → side-by-side metrics
- `streamlit_app/pages/5_Compare_Models.py`: table of past runs from SQLite, select runs to compare, grouped bar chart of accuracy, recall, F1 and AUC. If all selected runs used the same 2 non-binary features, show their decision boundaries side by side.
- Lesson: "Which model does best on the *same* features, and which is easiest to explain?"

---

## 8. Each Streamlit model page follows this template

```python
st.title("K-Nearest Neighbors")
model_intro(title, explanation, try_this=[...])      # shared (C)

with st.form("knn_form"):
    features = feature_picker(default=["BMI", "Age"])  # shared (C)
    test_size, seed = split_settings()                 # shared (C)
    k = st.slider("k (neighbors)", 1, 50, 5)           # model-specific
    ...
    submitted = st.form_submit_button("Train model")

if submitted:
    st.session_state["knn_result"] = post_train("knn", payload)   # shared (C)

result = st.session_state.get("knn_result")
if result:
    metrics_panel(result["metrics"])                   # shared (C)
    col1, col2 = st.columns(2)
    confusion_matrix_chart(result["confusion_matrix"]) # shared (C)
    roc_chart(result["roc_curve"], result["metrics"]["roc_auc"])
    decision_boundary_chart(result["decision_boundary"])   # shared (A)
    st.divider()
    # model-specific charts from result["model_details"]  ← owner builds these
```

Using `st.form` means the API is only called when the user clicks **Train**, not on every slider move.

---

## 9. Repository structure

```
model-playground/
├── main.py                     # FastAPI app, registers routers, middleware   (A)
├── data_pipeline/              # reused MP1 cleaning/validation code
├── src/
│   ├── schemas.py              # Pydantic contract                            (D)
│   ├── database.py             # connection, init, load_xy (B); save_run (D)
│   ├── cache.py                # MP3 fakeredis pattern                        (D)
│   ├── middleware.py           # MP2 logging                                  (A)
│   ├── ml/
│   │   ├── common.py           # run_training, evaluate                       (D)
│   │   ├── boundary.py         # compute_decision_boundary                    (A)
│   │   ├── logistic.py         # build_model + explain                        (A)
│   │   ├── knn.py                                                             (B)
│   │   ├── tree.py                                                            (C)
│   │   └── naive_bayes.py                                                     (D)
│   └── routes/
│       ├── data.py             # /features, /dataset/summary                  (B)
│       ├── runs.py             # /runs, /runs/{id}, /runs/compare             (C)
│       ├── logistic.py  knn.py  tree.py  naive_bayes.py                       (each owner)
├── streamlit_app/
│   ├── Home.py                 # intro + dataset summary                      (A/B)
│   ├── components/             # api_client, pickers, shared charts           (C)
│   │   └── boundary_chart.py   # decision_boundary_chart                      (A)
│   └── pages/
│       ├── 1_Logistic_Regression.py                                           (A)
│       ├── 2_KNN.py                                                           (B)
│       ├── 3_Decision_Tree.py                                                 (C)
│       ├── 4_Naive_Bayes.py                                                   (D)
│       └── 5_Compare_Models.py                                                (C, stretch)
├── scripts/
│   ├── make_sample.py          # UCI → clean → 5k balanced CSV (run once)     (B)
│   └── load_data.py            # CSV → SQLite                                 (B)
├── data/
│   ├── diabetes_sample.csv     # committed (~250 KB)
│   └── app.db                  # gitignored, built by load_data.py
├── fixtures/sample_200.csv     # small test dataset
├── tests/
│   ├── conftest.py             # temp DB + TestClient                         (A)
│   ├── test_data.py  test_boundary.py  test_runs.py  test_cache.py            (B, A, C, D)
│   └── test_logistic.py  test_knn.py  test_tree.py  test_naive_bayes.py       (each owner)
├── requirements.txt
└── README.md
```

**Expected run commands (for the README):**
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python scripts/load_data.py       # builds data/app.db from the committed CSV
uvicorn main:app --reload         # terminal 1 → http://localhost:8000/docs
streamlit run streamlit_app/Home.py   # terminal 2
pytest                            # run all tests
```

**Libraries to add on top of the MP3 requirements:** `scikit-learn`, `streamlit`, `plotly`, `requests`, `ucimlrepo`. Pin exact versions (`pip freeze`) so everyone gets the same results.

---

## 10. Timeline

| Date | Milestone |
|---|---|
| **Tue 11/24** | Plan agreed. D opens the small `src/schemas.py` PR; A opens the setup PR. |
| **Wed 11/25 – Sun 11/29** | Foundation PRs (A setup → D schemas → D pipeline → B data → C components) reviewed and merged. Model owners prototype `build_model`/`explain` in a notebook or script. |
| **Tue 12/1** (work session) | All feature branches created from updated `main`. Endpoints working in `/docs`. A starts the decision-boundary helper; C starts `/runs`. |
| **Fri 12/4** | **Progress report.** Goal: all 4 `/train/*` endpoints + basic pages (shared panels) merged, plus the decision boundary and `/runs`. |
| **Sat 12/5 – Sun 12/6** | Model-specific charts, "Try this" prompts, tests, Compare page (C), README sections. |
| **12/7 or 12/10** | **Demo testing** on the presenting laptop. Run every page from a fresh clone using only the README. |
| **12/8 or 12/11** | **Presentation.** Each member demos their own model page. |
| **Mon 12/14** | **Final deliverable.** README complete, all tests pass, all PRs merged. |

---

## 11. Git workflow

- Branch from an up-to-date `main`: `git checkout main && git pull origin main && git checkout -b feature/<name>`
- While working, bring in teammates' merged work: `git fetch origin && git merge origin/main`
- **Review rotation:** A reviews B's PRs · B reviews C's · C reviews D's · D reviews A's
- Keep PRs small. Each foundation task and each model feature is its own PR.
- Never commit `data/app.db`, `.venv/`, or API keys.

### Definition of done for each model feature
- [ ] `POST /train/<model>` works in `/docs` and returns the shared `TrainResponse` shape
- [ ] Parameter bounds enforced (422) and domain errors return 400 with a helpful message
- [ ] Page shows shared panels **and** at least 2 model-specific charts
- [ ] "What this model does" text + 2–3 "Try this" prompts
- [ ] At least 3 tests in `tests/test_<model>.py`, all passing
- [ ] PR reviewed by your rotation partner, merged
- [ ] Row filled in the README feature table

---

## 12. README feature table (fill in as features merge)

| Member | Feature | Branch / PR | What a user can do | Tests |
|---|---|---|---|---|
| A | Logistic Regression page + `/train/logistic` | `feature/logistic-regression` #_ | Tune C and penalty, see coefficients, move the threshold to trade precision vs. recall | `tests/test_logistic.py` |
| A | Shared decision boundary (all model pages) | `feature/decision-boundary` #_ | Pick 2 features and see how the model divides the space | `tests/test_boundary.py` |
| B | KNN page + `/train/knn` | `feature/knn` #_ | Tune k, metric, scaling, see the accuracy-vs-k curve and why scaling matters | `tests/test_knn.py` |
| C | Decision Tree page + `/train/tree` | `feature/decision-tree` #_ | Tune depth and leaf size, see the tree diagram and overfitting curve | `tests/test_tree.py` |
| C | Run history + Compare Models page | `feature/run-history`, `feature/compare-models` #_ | Browse past training runs and compare models side by side | `tests/test_runs.py` |
| D | Naive Bayes page + `/train/naive_bayes` | `feature/naive-bayes` #_ | Switch Gaussian/Bernoulli, set the prior, see learned class distributions | `tests/test_naive_bayes.py` |
| D | Training cache | `setup/training-pipeline` #_ | Re-run a previous setting and get the result instantly (⚡ cached) | `tests/test_cache.py` |

---

## 13. Limitations to mention in the presentation

- **Educational, not medical:** the app teaches how models behave. It does not diagnose anyone.
- **Balanced sample ≠ real world:** the sample is 50/50, while the real US rate is ~14%. Real-world precision would be lower (see the NB prior demo).
- **Self-reported 2015 survey data:** answers can be inaccurate, the features are coarse (mostly yes/no, bucketed age and income), and the data is US-only.
- **One train/test split:** results depend on `random_state`. Cross-validation would give more stable estimates (possible future work).
- **Decision boundaries are 2-D only:** they show the model restricted to two features, not the full model.
