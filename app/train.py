"""Train, compare and save a churn model.

Run:  python -m app.train
Outputs (in models/):
    model.joblib        full sklearn Pipeline (preprocessing + classifier)
    metrics.json        test-set metrics + cross-validation comparison
    drift_reference.json  training distribution of numeric features, for drift checks
"""
import json

import numpy as np

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.drift import build_reference
from app.features import CATEGORICAL_FEATURES, MODEL_DIR, NUMERIC_FEATURES, load_dataset

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": an unseen category at serving time becomes
        # all zeros instead of crashing the API.
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def candidate_models() -> dict:
    # class_weight="balanced" because only ~26% of customers churn; without it
    # the model would favour predicting "no churn" and miss real churners.
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def main() -> None:
    X, y = load_dataset()
    print(f"Loaded {len(X)} rows, churn rate = {y.mean():.1%}")

    # Hold out a test set that is never used for model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    for name, clf in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_scores[name] = {"mean_roc_auc": scores.mean(), "std": scores.std()}
        print(f"  {name:22s} CV ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(cv_scores, key=lambda n: cv_scores[n]["mean_roc_auc"])
    print(f"Best model: {best_name}")

    model = Pipeline([("prep", build_preprocessor()),
                      ("clf", candidate_models()[best_name])])

    # Tune the decision threshold. 0.5 is arbitrary; missing a churner (lost
    # customer) usually costs more than a wasted retention offer, so we pick the
    # threshold that maximises F1 on out-of-fold training predictions. The test
    # set is not touched, so the test metrics stay honest.
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    candidates = np.arange(0.20, 0.71, 0.01)
    threshold = float(max(candidates, key=lambda t: f1_score(y_train, oof >= t)))
    print(f"Tuned decision threshold: {threshold:.2f}")

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "model": best_name,
        "threshold": round(threshold, 2),
        "test_roc_auc": roc_auc_score(y_test, proba),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "cv_comparison": cv_scores,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print(classification_report(y_test, pred, target_names=["stay", "churn"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "drift_reference.json").write_text(
        json.dumps(build_reference(X_train), indent=2))
    print(f"Saved model and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
