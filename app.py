"""
Adult Census Income – Streamlit Deployment App
===============================================
Predicts whether a person earns >50K or <=50K per year.

Run locally:
    pip install streamlit scikit-learn pandas numpy joblib
    streamlit run app.py

The app expects the following artefacts (produced by the training
script) to live next to app.py in an `artifacts/` folder:
    artifacts/model.pkl
    artifacts/scaler.pkl
    artifacts/imputer.pkl
    artifacts/label_encoders.pkl
    artifacts/label_encoder_y.pkl
    artifacts/feature_cols.pkl
"""

import os
import pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
ART_DIR    = BASE_DIR / "artifacts"

# ── Load artefacts (cached so they load only once) ────────────────────────────
@st.cache_resource
def load_artifacts():
    model          = joblib.load(ART_DIR / "model.pkl")
    scaler         = joblib.load(ART_DIR / "scaler.pkl")
    imputer        = joblib.load(ART_DIR / "imputer.pkl")
    label_encoders = joblib.load(ART_DIR / "label_encoders.pkl")
    le_y           = joblib.load(ART_DIR / "label_encoder_y.pkl")
    feature_cols   = joblib.load(ART_DIR / "feature_cols.pkl")
    return model, scaler, imputer, label_encoders, le_y, feature_cols

# ── Option lists (from training data) ────────────────────────────────────────
WORKCLASS_OPTIONS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov",
    "Without-pay", "Never-worked",
]
MARITAL_OPTIONS = [
    "Married-civ-spouse", "Divorced", "Never-married",
    "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse",
]
OCCUPATION_OPTIONS = [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces",
]
RELATIONSHIP_OPTIONS = [
    "Wife", "Own-child", "Husband",
    "Not-in-family", "Other-relative", "Unmarried",
]
RACE_OPTIONS = [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
    "Other", "Black",
]
SEX_OPTIONS      = ["Male", "Female"]
EDUCATION_OPTIONS = [
    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
    "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
    "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
]
EDUCATION_NUM_MAP = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9,
    "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16,
}

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(inputs: dict) -> tuple[str, float]:
    """Return (label, probability) for the >50K class."""
    model, scaler, imputer, label_encoders, le_y, feature_cols = load_artifacts()

    # Build a one-row DataFrame with ALL original columns
    # (imputer expects occupation / workclass columns)
    raw = pd.DataFrame([{
        "age":            inputs["age"],
        "workclass":      inputs["workclass"],
        "fnlwgt":         inputs["fnlwgt"],
        "education.num":  inputs["education_num"],
        "marital.status": inputs["marital_status"],
        "occupation":     inputs["occupation"],
        "relationship":   inputs["relationship"],
        "race":           inputs["race"],
        "sex":            inputs["sex"],
        "capital.gain":   inputs["capital_gain"],
        "capital.loss":   inputs["capital_loss"],
        "hours.per.week": inputs["hours_per_week"],
        # native.country is imputed then dropped, so keep a dummy
        "native.country": np.nan,
    }])

    # Impute (workclass / occupation / native.country)
    impute_cols = ["occupation", "workclass", "native.country"]
    raw[impute_cols] = imputer.transform(raw[impute_cols])

    # Drop columns not used in training
    raw.drop(["native.country"], axis=1, inplace=True)

    # Encode categoricals
    for col, le in label_encoders.items():
        if col in raw.columns:
            val = raw[col].iloc[0]
            raw[col] = le.transform([val])[0] if val in le.classes_ else -1

    # Reorder columns
    raw = raw[feature_cols]

    # Scale
    raw_s = scaler.transform(raw)

    # Predict
    prob   = model.predict_proba(raw_s)[0]
    cls_id = model.predict(raw_s)[0]
    label  = le_y.inverse_transform([cls_id])[0]
    conf   = prob[cls_id]
    return label, conf


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adult Census Income Predictor",
    page_icon="💰",
    layout="centered",
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("💰 Adult Census Income Predictor")
st.markdown(
    "This app predicts whether a person earns **>50K** or **≤50K** per year "
    "based on census data. Fill in the details below and click **Predict**."
)
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    age           = st.slider("Age", 17, 90, 35)
    education     = st.selectbox("Education Level", EDUCATION_OPTIONS,
                                  index=EDUCATION_OPTIONS.index("Bachelors"))
    education_num = EDUCATION_NUM_MAP[education]
    marital_status= st.selectbox("Marital Status", MARITAL_OPTIONS)
    occupation    = st.selectbox("Occupation", OCCUPATION_OPTIONS)
    relationship  = st.selectbox("Relationship", RELATIONSHIP_OPTIONS)

with col2:
    workclass     = st.selectbox("Work Class", WORKCLASS_OPTIONS)
    race          = st.selectbox("Race", RACE_OPTIONS)
    sex           = st.selectbox("Sex", SEX_OPTIONS)
    hours_per_week= st.slider("Hours per Week", 1, 99, 40)
    capital_gain  = st.number_input("Capital Gain ($)", 0, 99999, 0, step=100)
    capital_loss  = st.number_input("Capital Loss ($)", 0, 4356, 0, step=100)

# fnlwgt is a census weight – set a reasonable default; advanced users can tweak
with st.expander("Advanced — Census Weight (fnlwgt)"):
    fnlwgt = st.number_input("fnlwgt", min_value=10000, max_value=1500000,
                              value=189778, step=1000,
                              help="Statistical weight assigned by Census Bureau.")

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔮  Predict Income", use_container_width=True, type="primary"):
    if not ART_DIR.exists():
        st.error(
            "Artifacts folder not found. Please run the training script first "
            "to generate `artifacts/` next to `app.py`."
        )
    else:
        inputs = dict(
            age=age,
            workclass=workclass,
            fnlwgt=fnlwgt,
            education_num=education_num,
            marital_status=marital_status,
            occupation=occupation,
            relationship=relationship,
            race=race,
            sex=sex,
            capital_gain=capital_gain,
            capital_loss=capital_loss,
            hours_per_week=hours_per_week,
        )
        try:
            label, confidence = predict(inputs)
            if label == ">50K":
                st.success(f"### 🟢 Predicted Income: **{label}**")
            else:
                st.info(f"### 🔵 Predicted Income: **{label}**")

            st.metric("Model Confidence", f"{confidence * 100:.1f}%")
            st.caption(
                "Model: Random Forest Classifier  •  "
                "Test accuracy: 85.88%  •  "
                "Dataset: UCI Adult Census Income"
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small>Built with Streamlit · scikit-learn · Adult Census Income dataset (UCI)</small>",
    unsafe_allow_html=True,
)
