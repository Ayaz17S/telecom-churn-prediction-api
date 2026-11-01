import pytest
from fastapi.testclient import TestClient

from app.api import app

CUSTOMER = {
    "tenure": 2, "MonthlyCharges": 95.5, "TotalCharges": 191.0, "SeniorCitizen": 0,
    "gender": "Female", "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
}

LOYAL = {**CUSTOMER, "tenure": 70, "MonthlyCharges": 25.0, "TotalCharges": 1750.0,
         "InternetService": "No", "OnlineSecurity": "No internet service",
         "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
         "TechSupport": "No internet service", "StreamingTV": "No internet service",
         "StreamingMovies": "No internet service", "Contract": "Two year",
         "PaymentMethod": "Bank transfer (automatic)"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs the startup/shutdown lifespan
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_valid_probability(client):
    body = client.post("/predict", json=CUSTOMER).json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}


def test_model_ranks_risky_customer_above_loyal_one(client):
    risky = client.post("/predict", json=CUSTOMER).json()["churn_probability"]
    loyal = client.post("/predict", json=LOYAL).json()["churn_probability"]
    assert risky > loyal


def test_invalid_input_is_rejected(client):
    bad = {**CUSTOMER, "Contract": "Ten year"}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", json={**CUSTOMER, "tenure": -5}).status_code == 422


def test_batch_and_stats(client):
    resp = client.post("/predict/batch", json={"customers": [CUSTOMER, LOYAL] * 30})
    assert resp.json()["count"] == 60
    stats = client.get("/monitoring/stats").json()
    assert stats["total_predictions"] >= 60


def test_drift_endpoint(client):
    body = client.get("/monitoring/drift").json()
    assert set(body["features"]) == {"tenure", "MonthlyCharges", "TotalCharges"}
    assert body["overall_status"] in {"stable", "moderate", "significant"}
