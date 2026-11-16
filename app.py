from pathlib import Path

import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Cancer Recurrence Prediction",
    page_icon="🏥",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    artifact_dir = Path(__file__).resolve().parent
    model = joblib.load(artifact_dir / "cancer_model.pkl")
    scaler = joblib.load(artifact_dir / "scaler.pkl")
    means = joblib.load(artifact_dir / "feature_means.pkl")
    feature_names = joblib.load(artifact_dir / "feature_names.pkl")

    if list(model.feature_names_in_) != feature_names:
        raise ValueError("Model features do not match feature_names.pkl.")

    return model, scaler, means, feature_names


model, scaler, means, feature_names = load_artifacts()

# Title
st.title("🏥 Cancer Recurrence Prediction Model")
st.markdown("Enter patient details below to predict the risk of cancer recurrence after treatment.")
st.markdown("---")

# Input form
st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age at Diagnosis", min_value=20, max_value=100, value=50)
    tumor_size = st.number_input("Tumor Size (mm)", min_value=1, max_value=200, value=25)
    lymph_nodes = st.number_input("Lymph Nodes Examined Positive", min_value=0, max_value=30, value=0)
    mutation_count = st.number_input("Mutation Count", min_value=0, max_value=100, value=5)
    npi = st.number_input("Nottingham Prognostic Index", min_value=1.0, max_value=7.0, value=3.0)

with col2:
    er_status = st.selectbox("ER Status", ["Positive", "Negative"])
    her2_status = st.selectbox("HER2 Status", ["Negative", "Positive"])
    pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
    surgery = st.selectbox("Type of Breast Surgery", ["Mastectomy", "Breast Conserving"])
    grade = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3])

with col3:
    chemo = st.selectbox("Chemotherapy", ["Yes", "No"])
    hormone = st.selectbox("Hormone Therapy", ["Yes", "No"])
    radio = st.selectbox("Radio Therapy", ["Yes", "No"])
    cancer_type = st.selectbox("Cancer Type Detailed", [
        "Breast Invasive Ductal Carcinoma",
        "Breast Invasive Lobular Carcinoma",
        "Breast Mixed Ductal and Lobular Carcinoma",
        "Invasive Breast Carcinoma",
        "Breast Invasive Mixed Mucinous Carcinoma",
        "Breast Angiosarcoma",
        "Breast",
        "Metaplastic Breast Cancer"
    ])
    histologic = st.selectbox("Tumor Other Histologic Subtype", [
        "Ductal/NST", "Mixed", "Lobular",
        "Tubular/ cribriform", "Mucinous",
        "Medullary", "Other", "Metaplastic"
    ])

st.markdown("---")

# Predict button
if st.button("🔍 Predict Recurrence Risk", width="stretch"):

    # Encode inputs
    cancer_type_map = {
        "Breast": 0, "Breast Angiosarcoma": 1,
        "Breast Invasive Ductal Carcinoma": 2,
        "Breast Invasive Lobular Carcinoma": 3,
        "Breast Invasive Mixed Mucinous Carcinoma": 4,
        "Breast Mixed Ductal and Lobular Carcinoma": 5,
        "Invasive Breast Carcinoma": 6,
        "Metaplastic Breast Cancer": 7
    }
    histologic_map = {
        "Ductal/NST": 0, "Lobular": 1, "Medullary": 2,
        "Metaplastic": 3, "Mixed": 4, "Mucinous": 5,
        "Other": 6, "Tubular/ cribriform": 7
    }

    # Build input dataframe
    input_data = pd.DataFrame([[
        age,
        1 if surgery == "Mastectomy" else 0,
        cancer_type_map[cancer_type],
        1 if chemo == "Yes" else 0,
        1 if er_status == "Positive" else 0,
        grade,
        1 if her2_status == "Positive" else 0,
        histologic_map[histologic],
        1 if hormone == "Yes" else 0,
        lymph_nodes,
        mutation_count,
        npi,
        1 if pr_status == "Positive" else 0,
        1 if radio == "Yes" else 0,
        tumor_size
    ]], columns=feature_names, dtype=float)

    # Apply the matching scaler statistics; its training data also included an
    # outcome-only field that is not available when making a prediction.
    scale_cols = [
        'Age at Diagnosis', 'Neoplasm Histologic Grade',
        'Lymph nodes examined positive', 'Mutation Count',
        'Nottingham prognostic index', 'Tumor Size'
    ]
    scaler_features = list(scaler.feature_names_in_)
    missing_scale_cols = [name for name in scale_cols if name not in scaler_features]
    if missing_scale_cols:
        raise ValueError(f"Scaler is missing required features: {missing_scale_cols}")

    scale_indices = [scaler_features.index(name) for name in scale_cols]
    input_data[scale_cols] = (
        input_data[scale_cols].to_numpy()
        - scaler.mean_[scale_indices]
    ) / scaler.scale_[scale_indices]

        # Predict
    probability = model.predict_proba(input_data)[0][1]

    # Tier cutoffs from the test-set score distribution (tertiles)
    LOW_CUTOFF = 0.41
    HIGH_CUTOFF = 0.52

    if probability < LOW_CUTOFF:
        risk_label = "LOWER RISK"
    elif probability < HIGH_CUTOFF:
        risk_label = "MODERATE RISK"
    else:
        risk_label = "HIGH RISK"

    # Show result
    st.subheader("Prediction Result")
    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Risk Score", f"{probability*100:.1f}%")

    with col_b:
        if risk_label == "HIGH RISK":
            st.error("⚠️ HIGH RISK: recommend closer follow-up")
        elif risk_label == "MODERATE RISK":
            st.warning("🟠 MODERATE RISK: routine monitoring with review")
        else:
            st.success("✅ LOWER RISK: standard follow-up")

        st.caption(
        "In test data, recurrence rates were about 29% (Lower), 36% (Moderate) and 59% (High). "
        "Tiers rank risk and are not calibrated probabilities. Lower risk does not mean no risk."
    )
    # SHAP explanation
    st.subheader("Why this prediction?")

    # SHAP values for a linear model: coefficient x (value - training mean)
    coef = model.coef_[0]
    x = input_data[feature_names].to_numpy()[0]
    shap_vals = coef * (x - means)
    base_value = float(model.intercept_[0] + coef @ means)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=x,
            feature_names=feature_names,
        ),
        show=False,
    )
    st.pyplot(fig, alt="SHAP waterfall explanation of the prediction")
    plt.close()

    # Disclaimer
    st.markdown("---")
    st.warning("⚠️ Disclaimer: This is a research prototype and not a clinical decision making tool. Always consult a qualified medical professional.")