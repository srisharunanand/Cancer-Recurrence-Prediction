# Cancer Recurrence Prediction (Breast Cancer, METABRIC)

Research prototype that ranks breast cancer patients by risk of recurrence after treatment.
**Live demo:** https://cancer-recurrence-prediction-gkggzpjsob5hmfbyvalu3h.streamlit.app/

> Not a clinical tool. Do not use for medical decisions.

## What it does
Enter 15 clinical features and get a risk score, a risk tier (Lower / Moderate / High)
and a SHAP waterfall chart showing which features pushed the score up or down.

## Dataset
METABRIC clinical data, 2,509 patients (2,488 after removing rows with no outcome).

## Pipeline
1. Cleaning: dropped high-missing columns, median/mode imputation, encoding, scaling
2. Removed data leakage (relapse-free months)
3. Compared 4 models: Logistic Regression, Random Forest, XGBoost, Gradient Boosting
4. Tuned with GridSearchCV; Logistic Regression performed best
5. SHAP for explainability
6. Cox Proportional Hazards for time-to-recurrence (in the notebook)

## Results (498 held-out patients)
| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.659 |
| Random Forest | 0.628 |
| XGBoost | 0.598 |
| Gradient Boosting | 0.596 |

Cox model C-index: 0.66

| Risk tier | Patients | Recurrence rate |
|---|---|---|
| Lower | 160 | 29.4% |
| Moderate | 191 | 35.6% |
| High | 147 | 58.5% |

Overall recurrence rate: 40.4%.

## Limitations
- ROC-AUC of 0.66 means modest discrimination
- High tier separates well; Lower and Moderate are hard to tell apart
- Scores are rankings, not calibrated probabilities
- Tier cutoffs were set from the same test set's score distribution
- Trained on one cohort (UK/Canada); not validated on other populations
- Clinical features only, no genomic data

## Tech
Python, scikit-learn, XGBoost, SHAP, lifelines, Streamlit

## Notebook
(add your public Kaggle link here)
