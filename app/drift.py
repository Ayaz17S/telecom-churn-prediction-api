"""Data drift detection with the Population Stability Index (PSI).

Idea: at training time we record how each numeric feature is distributed
(which share of customers falls in each of 10 bins). In production we bin the
incoming requests the same way and compare the two distributions:

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Rule of thumb used in banking/telecom model monitoring:
    PSI < 0.10  -> stable
    0.10-0.25   -> moderate shift, investigate
    PSI > 0.25  -> significant shift, consider retraining
"""
import numpy as np
import pandas as pd

DRIFT_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
N_BINS = 10
EPS = 1e-4  # avoids log(0) / division by zero for empty bins


def _bin_shares(values: np.ndarray, edges: list[float]) -> np.ndarray:
    # Open-ended outer bins so values outside the training range still count.
    inner = np.asarray(edges[1:-1])
    counts = np.bincount(np.searchsorted(inner, values, side="right"),
                         minlength=len(edges) - 1)
    return counts / max(len(values), 1)


def build_reference(X_train: pd.DataFrame) -> dict:
    ref = {}
    for col in DRIFT_FEATURES:
        values = X_train[col].dropna().to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
        ref[col] = {"edges": edges.tolist(),
                    "expected": _bin_shares(values, edges.tolist()).tolist()}
    return ref


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def status(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


def drift_report(reference: dict, live: pd.DataFrame) -> dict:
    report = {}
    for col, ref in reference.items():
        values = live[col].dropna().to_numpy()
        actual = _bin_shares(values, ref["edges"])
        score = psi(np.asarray(ref["expected"]), actual)
        report[col] = {"psi": round(score, 4), "status": status(score)}
    return report
