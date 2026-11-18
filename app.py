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

# ---------- Form defaults and examples ----------
DEFAULTS = {
    "age": 50, "tumor_size": 25, "lymph_nodes": 0, "mutation_count": 5, "npi": 3.0,
    "er_status": "Positive", "her2_status": "Negative", "pr_status": "Positive",
    "surgery": "Mastectomy", "grade": 1,
    "chemo": "Yes", "hormone": "Yes", "radio": "Yes",
    "cancer_type": "Breast Invasive Ductal Carcinoma",
    "histologic": "Ductal/NST",
}

EXAMPLES = {
    "Low": {
        **DEFAULTS, "age": 62, "tumor_size": 10, "lymph_nodes": 0, "mutation_count": 3,
        "npi": 2.0, "surgery": "Breast Conserving", "grade": 1, "chemo": "No",
    },
    "Moderate": dict(DEFAULTS),
    "High": {
        **DEFAULTS, "age": 45, "tumor_size": 60, "lymph_nodes": 8, "mutation_count": 8,
        "npi": 5.5, "er_status": "Negative", "her2_status": "Positive",
        "pr_status": "Negative", "grade": 3, "hormone": "No",
    },
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_example(name):
    for k, v in EXAMPLES[name].items():
        st.session_state[k] = v


# ---------- Header ----------
st.title("🏥 Cancer Recurrence Prediction Model")
st.caption("Breast cancer risk stratification after treatment | METABRIC data | Research prototype")

tab_predict, tab_perf, tab_about = st.tabs(["🔍 Predict", "📊 Model Performance", "ℹ️ About"])

# ---------- Tab 1: Predict ----------
with tab_predict:
    st.markdown("**Quick start:** load an example patient, or enter your own details.")
    b1, b2, b3 = st.columns(3)
    b1.button("🟢 Low-risk example", on_click=load_example, args=("Low",), width="stretch")
    b2.button("🟠 Moderate example", on_click=load_example, args=("Moderate",), width="stretch")
    b3.button("🔴 High-risk example", on_click=load_example, args=("High",), width="stretch")

    st.markdown("---")
    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age at Diagnosis", min_value=20, max_value=100, key="age")
        tumor_size = st.number_input("Tumor Size (mm)", min_value=1, max_value=200, key="tumor_size")
        lymph_nodes = st.number_input("Lymph Nodes Examined Positive", min_value=0, max_value=30, key="lymph_nodes")
        mutation_count = st.number_input("Mutation Count", min_value=0, max_value=100, key="mutation_count")
        npi = st.number_input("Nottingham Prognostic Index", min_value=1.0, max_value=7.0, key="npi")

    with col2:
        er_status = st.selectbox("ER Status", ["Positive", "Negative"], key="er_status")
        her2_status = st.selectbox("HER2 Status", ["Negative", "Positive"], key="her2_status")
        pr_status = st.selectbox("PR Status", ["Positive", "Negative"], key="pr_status")
        surgery = st.selectbox("Type of Breast Surgery", ["Mastectomy", "Breast Conserving"], key="surgery")
        grade = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3], key="grade")

    with col3:
        chemo = st.selectbox("Chemotherapy", ["Yes", "No"], key="chemo")
        hormone = st.selectbox("Hormone Therapy", ["Yes", "No"], key="hormone")
        radio = st.selectbox("Radio Therapy", ["Yes", "No"], key="radio")
        cancer_type = st.selectbox("Cancer Type Detailed", [
            "Breast Invasive Ductal Carcinoma",
            "Breast Invasive Lobular Carcinoma",
            "Breast Mixed Ductal and Lobular Carcinoma",
            "Invasive Breast Carcinoma",
            "Breast Invasive Mixed Mucinous Carcinoma",
            "Breast Angiosarcoma",
            "Breast",
            "Metaplastic Breast Cancer"
        ], key="cancer_type")
        histologic = st.selectbox("Tumor Other Histologic Subtype", [
            "Ductal/NST", "Mixed", "Lobular",
            "Tubular/ cribriform", "Mucinous",
            "Medullary", "Other", "Metaplastic"
        ], key="histologic")

    st.markdown("---")

    if st.button("🔍 Predict Recurrence Risk", width="stretch"):

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
            input_data[scale_cols].to_numpy() - scaler.mean_[scale_indices]
        ) / scaler.scale_[scale_indices]

        probability = model.predict_proba(input_data)[0][1]

        LOW_CUTOFF = 0.41
        HIGH_CUTOFF = 0.52

        if probability < LOW_CUTOFF:
            risk_label = "LOWER RISK"
        elif probability < HIGH_CUTOFF:
            risk_label = "MODERATE RISK"
        else:
            risk_label = "HIGH RISK"

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
                st.success("✅ LOWER RISK: standard follow-up, still not risk-free")

        st.caption(
            "In test data, recurrence rates were about 29% (Lower), 36% (Moderate) and 59% (High). "
            "Tiers rank risk and are not calibrated probabilities. Lower risk does not mean no risk."
        )

        st.subheader("Why this prediction?")

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

        st.markdown("---")
        st.warning("⚠️ Disclaimer: This is a research prototype and not a clinical decision making tool. Always consult a qualified medical professional.")

# ---------- Tab 2: Model Performance ----------
with tab_perf:
    st.subheader("Model comparison (498 held-out patients)")
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Gradient Boosting"],
        "ROC-AUC": [0.659, 0.628, 0.598, 0.596],
    }), hide_index=True, width="stretch")
    st.caption("Logistic Regression was selected. Cox survival model C-index: 0.66.")

    st.subheader("Do the risk tiers separate outcomes?")
    tier_df = pd.DataFrame({
        "Tier": ["Lower", "Moderate", "High"],
        "Patients": [160, 191, 147],
        "Recurrence rate (%)": [29.4, 35.6, 58.5],
    })
    st.dataframe(tier_df, hide_index=True, width="stretch")
    st.bar_chart(tier_df.set_index("Tier")["Recurrence rate (%)"])
    st.caption("Overall recurrence rate in the test set: 40.4%. The High tier separates well; Lower and Moderate are hard to tell apart.")

# ---------- Tab 3: About ----------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        "Ranks breast cancer patients by risk of recurrence after treatment, using 15 clinical "
        "features from the METABRIC dataset (2,488 patients), with a SHAP chart explaining each score."
    )
    st.subheader("Limitations")
    st.markdown(
        "- ROC-AUC of 0.66 means modest discrimination\n"
        "- Scores are rankings, not calibrated probabilities\n"
        "- Trained on one cohort (UK/Canada), not validated elsewhere\n"
        "- Clinical features only, no genomic data\n"
        "- Research prototype, not a clinical tool"
    )
    st.warning("This is a research prototype and not a clinical decision making tool.")