"""Data loading and feature definitions shared by training and serving.

Keeping this in one place guarantees the API sees exactly the same columns
and cleaning steps the model was trained on (avoids training/serving skew).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"

TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known data-quality issues in the raw Telco dataset."""
    df = df.copy()
    # TotalCharges is stored as text and is blank for brand-new customers
    # (tenure == 0). Convert to numbers; blanks become NaN and are imputed later.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = clean(pd.read_csv(path))
    df = df.drop(columns=["customerID"])  # an ID carries no predictive signal
    y = (df[TARGET] == "Yes").astype(int)
    return df[ALL_FEATURES], y
