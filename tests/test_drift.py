import numpy as np
import pandas as pd

from app.drift import build_reference, drift_report, psi


def make_df(rng, tenure_mean):
    n = 2000
    return pd.DataFrame({
        "tenure": rng.normal(tenure_mean, 10, n).clip(0, 72),
        "MonthlyCharges": rng.normal(65, 20, n).clip(18, 120),
        "TotalCharges": rng.normal(2000, 800, n).clip(0, None),
    })


def test_psi_is_zero_for_identical_distributions():
    shares = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(shares, shares) == 0.0


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    ref = build_reference(make_df(rng, 30))
    report = drift_report(ref, make_df(rng, 30))
    assert all(f["status"] == "stable" for f in report.values())


def test_shifted_distribution_is_flagged():
    rng = np.random.default_rng(0)
    ref = build_reference(make_df(rng, 30))
    report = drift_report(ref, make_df(rng, 55))  # customers suddenly much older
    assert report["tenure"]["status"] == "significant"
    assert report["MonthlyCharges"]["status"] == "stable"
