"""Streamlit web UI for the churn model.

Run locally:  streamlit run streamlit_app.py
Deployed on Streamlit Community Cloud straight from this GitHub repo.
"""
import json

import joblib
import pandas as pd
import streamlit as st

from app.drift import drift_report
from app.features import ALL_FEATURES, DATA_PATH, MODEL_DIR, clean

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner="Training the model (first start only, about a minute)...")
def load_model():
    # The model file is not committed to Git, so a fresh deployment trains it once.
    if not (MODEL_DIR / "model.joblib").exists():
        from app.train import main as train
        train()
    model = joblib.load(MODEL_DIR / "model.joblib")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    drift_ref = json.loads((MODEL_DIR / "drift_reference.json").read_text())
    return model, metrics, drift_ref


@st.cache_data
def load_data() -> pd.DataFrame:
    return clean(pd.read_csv(DATA_PATH))


model, metrics, drift_ref = load_model()
threshold = metrics["threshold"]

st.title("📉 Telecom Customer Churn Predictor")
st.write("Predicts how likely a telecom customer is to cancel, so the retention team can act early. "
         "Trained on 7,043 real customers (IBM Telco dataset).")

tab_predict, tab_model, tab_drift = st.tabs(["Predict a customer", "Model performance", "Drift monitoring demo"])

# ---------------------------------------------------------------- Predict
with tab_predict:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Customer details")
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (months with us)", 0, 72, 2)
        monthly = c2.number_input("Monthly charges", 18.0, 120.0, 95.5, step=0.5)
        contract = c3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        internet = c1.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment = c2.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = c3.radio("Paperless billing", ["Yes", "No"], horizontal=True)

        phone = c1.radio("Phone service", ["Yes", "No"], horizontal=True)
        multiple = c2.radio("Multiple lines", ["No", "Yes"], horizontal=True, disabled=phone == "No")
        senior = c3.radio("Senior citizen", ["No", "Yes"], horizontal=True)

        gender = c1.radio("Gender", ["Female", "Male"], horizontal=True)
        partner = c2.radio("Has partner", ["No", "Yes"], horizontal=True)
        dependents = c3.radio("Has dependents", ["No", "Yes"], horizontal=True)

        st.caption("Add-on services")
        has_net = internet != "No"
        addons = {}
        cols = st.columns(6)
        for col, name in zip(cols, ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                    "TechSupport", "StreamingTV", "StreamingMovies"]):
            checked = col.checkbox(name, value=name.startswith("Streaming"), disabled=not has_net)
            addons[name] = ("Yes" if checked else "No") if has_net else "No internet service"

    customer = {
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": round(tenure * monthly, 2),
        "SeniorCitizen": int(senior == "Yes"), "gender": gender, "Partner": partner,
        "Dependents": dependents, "PhoneService": phone,
        "MultipleLines": multiple if phone == "Yes" else "No phone service",
        "InternetService": internet, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, **addons,
    }
    proba = float(model.predict_proba(pd.DataFrame([customer])[ALL_FEATURES])[0, 1])
    risk = "high" if proba >= 0.6 else "medium" if proba >= 0.3 else "low"

    with right:
        st.subheader("Prediction")
        st.metric("Churn probability", f"{proba:.0%}")
        st.progress(proba)
        if proba >= threshold:
            st.error(f"⚠️ Likely to churn — {risk} risk. Recommend a retention offer.")
        else:
            st.success(f"✅ Likely to stay — {risk} risk.")
        st.caption(f"Decision threshold: {threshold:.2f} (tuned to catch more leaving customers "
                   "than the default 0.50).")
        st.info("Try it: raise tenure to 60 and choose a two-year contract — the risk drops sharply.")

# ---------------------------------------------------------------- Model
with tab_model:
    st.subheader("How good is the model?")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC (test)", f"{metrics['test_roc_auc']:.3f}")
    m2.metric("Recall — churners caught", f"{metrics['test_recall']:.0%}")
    m3.metric("Precision", f"{metrics['test_precision']:.0%}")
    m4.metric("F1", f"{metrics['test_f1']:.3f}")

    a, b = st.columns(2)
    with a:
        st.markdown("**Model comparison (5-fold cross-validation ROC-AUC)**")
        cv = pd.DataFrame(metrics["cv_comparison"]).T.rename(
            columns={"mean_roc_auc": "mean ROC-AUC", "std": "std dev"})
        st.dataframe(cv.style.format("{:.4f}"), use_container_width=True)
        st.caption(f"Selected: **{metrics['model']}** (best mean score).")
    with b:
        st.markdown(f"**Confusion matrix on {metrics['n_test']} unseen customers**")
        cm = metrics["confusion_matrix"]
        st.dataframe(pd.DataFrame(cm, index=["Actually stayed", "Actually churned"],
                                  columns=["Predicted stay", "Predicted churn"]),
                     use_container_width=True)
        st.caption("Bottom-right = churners we caught; bottom-left = churners we missed.")

# ---------------------------------------------------------------- Drift
with tab_drift:
    st.subheader("Data drift: does incoming data still look like the training data?")
    st.write("Models learn from the past. If customers change (e.g. after a price rise), predictions "
             "can silently get worse. The **Population Stability Index (PSI)** compares each feature's "
             "distribution now vs. during training: below 0.10 stable, 0.10–0.25 moderate, above 0.25 "
             "significant (consider retraining).")
    c1, c2 = st.columns(2)
    price_rise = c1.slider("Simulated price rise (%)", 0, 60, 40)
    new_only = c2.checkbox("Incoming customers are all new (tenure ≤ 6 months)", value=True)

    sample = load_data().sample(300, random_state=7).copy()
    sample["MonthlyCharges"] *= 1 + price_rise / 100
    if new_only:
        sample["tenure"] = sample["tenure"] % 7
    sample["TotalCharges"] = sample["tenure"] * sample["MonthlyCharges"]

    report = drift_report(drift_ref, sample)
    colour = {"stable": "🟢", "moderate": "🟠", "significant": "🔴"}
    cols = st.columns(len(report))
    for col, (feature, r) in zip(cols, report.items()):
        col.metric(feature, f"PSI {r['psi']:.3f}")
        col.write(f"{colour[r['status']]} {r['status']}")
    st.caption("Set the price rise to 0 and untick the box to see the real data come out stable.")

st.divider()
st.caption("Built with scikit-learn and Streamlit · "
           "[Source code](https://github.com/Ayaz17S/telecom-churn-prediction-api)")
