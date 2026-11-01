"""Demo: show the drift monitor catching a change in incoming data.

Run:  python -m app.simulate_drift

1. Sends 300 real customers to the API  -> drift should be "stable".
2. Sends 300 customers whose MonthlyCharges were raised 40% (e.g. a price
   hike) and who are all new (tenure <= 6)   -> drift should be flagged.
"""
import os
import tempfile

os.environ["CHURN_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "demo.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402
from app.features import ALL_FEATURES, DATA_PATH, clean  # noqa: E402

import pandas as pd  # noqa: E402


def to_payload(df: pd.DataFrame) -> list[dict]:
    df = df[ALL_FEATURES].copy()
    df["TotalCharges"] = df["TotalCharges"].astype(object).where(df["TotalCharges"].notna(), None)
    return df.to_dict(orient="records")


def main() -> None:
    data = clean(pd.read_csv(DATA_PATH)).sample(600, random_state=7)
    normal, shifted = data.iloc[:300].copy(), data.iloc[300:].copy()
    shifted["MonthlyCharges"] = (shifted["MonthlyCharges"] * 1.4).round(2)
    shifted["tenure"] = shifted["tenure"] % 7
    shifted["TotalCharges"] = (shifted["tenure"] * shifted["MonthlyCharges"]).round(2)

    with TestClient(app) as client:
        client.post("/predict/batch", json={"customers": to_payload(normal)})
        print("After normal traffic: ", client.get("/monitoring/drift?window=300").json())

        client.post("/predict/batch", json={"customers": to_payload(shifted)})
        print("After shifted traffic:", client.get("/monitoring/drift?window=300").json())
        print("Stats:", client.get("/monitoring/stats").json())


if __name__ == "__main__":
    main()
