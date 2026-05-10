"""
Adult Census Income – Streamlit App
=====================================
- Trains the model on first run and saves .pkl files next to app.py
- On subsequent runs loads the saved files (no retraining needed)
- No artifacts folder — all files sit in the same directory as app.py

Folder structure:
    app.py
    Adult_Census_Income.csv
    (model.pkl, scaler.pkl, imputer.pkl,
     label_encoders.pkl, label_encoder_y.pkl, feature_cols.pkl
     → created automatically on first run)

Run:
    pip install streamlit scikit-learn pandas numpy joblib
    streamlit run app.py
"""

import pathlib
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ── Paths (all in the same directory as app.py) ───────────────────────────────
BASE        = pathlib.Path(__file__).parent
CSV_PATH    = BASE / "Adult_Census_Income.csv"
MODEL_PATH  = BASE / "model.pkl"
SCALER_PATH = BASE / "scaler.pkl"
IMPUTER_PATH= BASE / "imputer.pkl"
LE_COLS_PATH= BASE / "label_encoders.pkl"
LE_Y_PATH   = BASE / "label_encoder_y.pkl"
FEAT_PATH   = BASE / "feature_cols.pkl"

# ── Constants ─────────────────────────────────────────────────────────────────
WORKCLASS_OPTIONS    = ['Federal-gov', 'Local-gov', 'Never-worked', 'Private',
                         'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']
MARITAL_OPTIONS      = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
                         'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
OCCUPATION_OPTIONS   = ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                         'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
                         'Other-service', 'Priv-house-serv', 'Prof-specialty',
                         'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
RELATIONSHIP_OPTIONS = ['Husband', 'Not-in-family', 'Other-relative',
                         'Own-child', 'Unmarried', 'Wife']
RACE_OPTIONS         = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
SEX_OPTIONS          = ['Female', 'Male']
EDUCATION_OPTIONS    = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th',
                         '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc',
                         'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']
EDUCATION_NUM_MAP    = {e: i + 1 for i, e in enumerate(EDUCATION_OPTIONS)}
CAT_COLS             = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
IMPUTE_COLS          = ['occupation', 'workclass', 'native.country']
DROP_COLS            = ['education', 'native.country']


# ── Train & save (runs only when .pkl files are missing) ──────────────────────
def train_and_save():
    st.info("🔧 First run — training model and saving to disk…")

    df = pd.read_csv(CSV_PATH, na_values=['?'])
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    X = df.drop('income', axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=15, stratify=y)

    for frame in [X_train, X_test, y_train, y_test]:
        frame.reset_index(drop=True, inplace=True)

    # Impute
    imputer = SimpleImputer(strategy='most_frequent')
    X_train[IMPUTE_COLS] = imputer.fit_transform(X_train[IMPUTE_COLS])
    X_test[IMPUTE_COLS]  = imputer.transform(X_test[IMPUTE_COLS])

    # Drop high-cardinality columns
    X_train.drop(DROP_COLS, axis=1, inplace=True)
    X_test.drop(DROP_COLS,  axis=1, inplace=True)

    # Encode categoricals — one LabelEncoder per column
    label_encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col]  = X_test[col].map(
            lambda x, _le=le: int(_le.transform([x])[0]) if x in _le.classes_ else -1)
        label_encoders[col] = le

    # Encode target
    le_y = LabelEncoder()
    y_train_enc = le_y.fit_transform(y_train)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # Train Random Forest
    model = RandomForestClassifier(random_state=15, n_estimators=100)
    model.fit(X_train_s, y_train_enc)

    feature_cols = X_train.columns.tolist()

    # Save all files directly next to app.py (no subfolder)
    joblib.dump(model,         MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(imputer,       IMPUTER_PATH)
    joblib.dump(label_encoders,LE_COLS_PATH)
    joblib.dump(le_y,          LE_Y_PATH)
    joblib.dump(feature_cols,  FEAT_PATH)

    st.success("✅ Model trained and saved!")


# ── Load saved pipeline ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_pipeline():
    # Train and save if any file is missing
    files = [MODEL_PATH, SCALER_PATH, IMPUTER_PATH, LE_COLS_PATH, LE_Y_PATH, FEAT_PATH]
    if not all(f.exists() for f in files):
        if not CSV_PATH.exists():
            st.error(f"CSV not found: place `{CSV_PATH.name}` next to `app.py`.")
            st.stop()
        train_and_save()

    return (
        joblib.load(MODEL_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(IMPUTER_PATH),
        joblib.load(LE_COLS_PATH),
        joblib.load(LE_Y_PATH),
        joblib.load(FEAT_PATH),
    )


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(inputs: dict):
    model, scaler, imputer, label_encoders, le_y, feature_cols = load_pipeline()

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
        "native.country": np.nan,
    }])

    raw[IMPUTE_COLS] = imputer.transform(raw[IMPUTE_COLS])
    raw.drop(["native.country"], axis=1, inplace=True)

    for col, le in label_encoders.items():
        if col in raw.columns:
            val = raw[col].iloc[0]
            raw[col] = int(le.transform([val])[0]) if val in le.classes_ else -1

    raw   = raw[feature_cols]
    raw_s = scaler.transform(raw)

    cls_id = model.predict(raw_s)[0]
    prob   = model.predict_proba(raw_s)[0]
    label  = le_y.inverse_transform([cls_id])[0]
    return label, prob[cls_id]


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Income Predictor", page_icon="💰", layout="centered")
st.title("💰 Adult Census Income Predictor")
st.markdown(
    "Predicts whether a person earns **>50K** or **≤50K** per year "
    "based on census data. Fill in the details and click **Predict**."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    age            = st.slider("Age", 17, 90, 35)
    education      = st.selectbox("Education Level", EDUCATION_OPTIONS,
                                   index=EDUCATION_OPTIONS.index("Bachelors"))
    education_num  = EDUCATION_NUM_MAP[education]
    marital_status = st.selectbox("Marital Status", MARITAL_OPTIONS,
                                   index=MARITAL_OPTIONS.index("Never-married"))
    occupation     = st.selectbox("Occupation", OCCUPATION_OPTIONS,
                                   index=OCCUPATION_OPTIONS.index("Prof-specialty"))
    relationship   = st.selectbox("Relationship", RELATIONSHIP_OPTIONS,
                                   index=RELATIONSHIP_OPTIONS.index("Not-in-family"))

with col2:
    workclass      = st.selectbox("Work Class", WORKCLASS_OPTIONS,
                                   index=WORKCLASS_OPTIONS.index("Private"))
    race           = st.selectbox("Race", RACE_OPTIONS,
                                   index=RACE_OPTIONS.index("White"))
    sex            = st.selectbox("Sex", SEX_OPTIONS)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    capital_gain   = st.number_input("Capital Gain ($)", 0, 99999, 0, step=100)
    capital_loss   = st.number_input("Capital Loss ($)", 0, 4356,  0, step=100)

with st.expander("Advanced — Census Weight (fnlwgt)"):
    fnlwgt = st.number_input("fnlwgt", 10000, 1500000, 189778, step=1000,
                              help="Statistical weight assigned by the Census Bureau.")

st.divider()

if st.button("🔮  Predict Income", use_container_width=True, type="primary"):
    with st.spinner("Predicting…"):
        label, confidence = predict(dict(
            age=age, workclass=workclass, fnlwgt=fnlwgt,
            education_num=education_num, marital_status=marital_status,
            occupation=occupation, relationship=relationship,
            race=race, sex=sex, capital_gain=capital_gain,
            capital_loss=capital_loss, hours_per_week=hours_per_week,
        ))

    if label == ">50K":
        st.success(f"### 🟢 Predicted Income: **{label}**")
    else:
        st.info(f"### 🔵 Predicted Income: **{label}**")

    st.metric("Model Confidence", f"{confidence * 100:.1f}%")
    st.caption("Model: Random Forest · Test accuracy: 85.88% · Dataset: UCI Adult Census Income")

st.divider()
st.markdown(
    "<small>Built with Streamlit · scikit-learn · Adult Census Income (UCI)</small>",
    unsafe_allow_html=True,
)
