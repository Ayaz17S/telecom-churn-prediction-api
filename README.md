# Telecom Customer Churn Prediction Service

An end-to-end ML service that predicts which telecom customers are likely to leave,
served as a REST API with prediction logging in SQL and data-drift monitoring.

**Stack:** Python, pandas, scikit-learn, FastAPI, Pydantic, SQLite, pytest, Docker

## Problem
Telecom companies lose roughly 1 in 4 customers a year. Keeping a customer is much
cheaper than finding a new one, so the retention team needs a list of customers who are
about to leave, early enough to offer them a better plan.

Dataset: IBM Telco Customer Churn, 7,043 customers, 19 features, 26.5% churn.

## Architecture
```
telco_churn.csv -> features.py (cleaning) -> train.py (compare 3 models, tune threshold)
                                                   |
                                    models/model.joblib + metrics.json + drift_reference.json
                                                   |
client --POST /predict--> api.py (FastAPI + Pydantic validation) --> model --> response
                                  |
                                  +--> db.py: every prediction logged in SQLite
                                                   |
                         GET /monitoring/drift  <--+  drift.py: PSI vs training data
                         GET /monitoring/stats  <--+  SQL aggregate queries
```

## Results (held-out test set, 1,409 customers)
| Model (5-fold CV ROC-AUC) | Score |
|---|---|
| Logistic Regression | 0.846 |
| Random Forest | 0.846 |
| **Gradient Boosting (selected)** | **0.848** |

Test set: ROC-AUC **0.843**. At the tuned threshold (0.35): churn recall **71%**
(the default 0.5 threshold only reached 52%), precision 56%.

## Key design decisions (what to say in the interview)
1. **One sklearn `Pipeline`** holds imputation, scaling, one-hot encoding and the model.
   The API loads that single object, so training and serving can never preprocess
   differently (training/serving skew).
2. **Class imbalance**: only 26% churn. Used `class_weight="balanced"` for the linear and
   forest models, stratified splits, and ROC-AUC instead of accuracy (a model that says
   "nobody churns" is 74% accurate and useless).
3. **Threshold tuning**: missing a churner costs more than a wasted retention offer, so
   the threshold is tuned for F1 on *out-of-fold training predictions*. The test set is
   never used for any decision, so the reported numbers are honest.
4. **Input validation**: Pydantic `Literal` types and ranges reject bad input with
   HTTP 422 before it reaches the model.
5. **Data drift monitoring**: the Population Stability Index compares live requests with
   the training distribution. PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 retrain.
6. **SQL logging** with parameterised queries (prevents SQL injection); gives an audit
   trail and business stats.

## Run it
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m app.train                 # trains and saves the model (about 30 s)
uvicorn app.api:app --reload        # open http://127.0.0.1:8000/docs
python -m pytest -q                 # 9 tests
python -m app.simulate_drift        # demo: drift detector catches a price hike
```
Docker: `docker build -t churn-api . && docker run -p 8000:8000 churn-api`

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness check |
| GET | `/model/info` | model name, threshold, test metrics |
| POST | `/predict` | one customer -> probability, yes/no, risk level |
| POST | `/predict/batch` | up to 1,000 customers, returns latency |
| GET | `/monitoring/stats` | SQL aggregates of logged predictions |
| GET | `/monitoring/drift` | PSI per numeric feature |

Drift demo output: real traffic gives PSI about 0.01-0.02 (stable); after a simulated
40% price rise and a wave of new customers, PSI jumps above 2 (significant).

## Limitations and next steps
- Explainability: add SHAP values so the retention team sees *why* a customer is at risk.
- Retraining pipeline triggered automatically when drift is significant.
- Move SQLite to PostgreSQL and add authentication for a multi-user deployment.
