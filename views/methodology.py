import pandas as pd
import streamlit as st

from app_core import ui
from app_core.config import REPO_URL
from app_core.data import load_metadata

meta = load_metadata()

ui.page_header("Methodology",
               "How the model behind Retention Planner was built, validated and how to read its outputs.",
               eyebrow="Reference")

st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#F3F6F8", color="#C9D2DC", fontname="Helvetica", fontsize=11];
  edge [color="#6B7A8C"];
  raw [label="Raw customer record\\n19 inputs"];
  clean [label="TelcoCleaner\\nTotalCharges fix, redundant levels"];
  prep [label="ColumnTransformer\\nmedian imputer | one-hot"];
  model [label="LightGBM\\n112 trees x 8 leaves", fillcolor="#CCE7E4"];
  proba [label="Churn probability\\n(calibrated)"];
  shap [label="SHAP / LIME\\nexplanations"];
  plan [label="Value-based\\ncampaign plan", fillcolor="#F9D9CF"];
  raw -> clean -> prep -> model -> proba -> plan; model -> shap; shap -> plan;
}
""", width="stretch")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("""
#### Data and validation
- Telco Customer Churn, 7,043 customers, 26.5 % churn.
- 80/20 stratified split. The test set was used **once**, after model, hyperparameters and threshold were fixed.
- Every comparison used the same 5 stratified folds. All learned preprocessing is fitted inside each fold (no leakage).
- Cleaning rules live inside the exported pipeline, so the app applies exactly the same transformations.

#### Model selection and optimization
- Compared Dummy, Logistic Regression, Random Forest and LightGBM on PR-AUC.
- Grid, Random and TPE search under the same budget (48 evaluations) were statistically tied; TPE was the most efficient.
- NSGA-II optimized PR-AUC and model complexity; the simplest solution within one standard error of the best was selected.
""")
with right:
    st.markdown(f"""
#### Decision threshold
- Probabilities are well calibrated (expected calibration error ≈ 0.02), so no extra calibration step is used.
- The threshold **{meta['business_threshold']:.2f}** maximizes expected campaign profit on out-of-fold predictions
  under declared hypothetical economics. The planner generalizes this to a value-based rule per customer.

#### Explainability
- SHAP (TreeSHAP, exact) gives additive contributions in log-odds, aggregated from one-hot columns to original features.
- LIME fits a local linear surrogate on the raw inputs; it agrees with SHAP on the main drivers.

#### Limitations
- Predictive association is not causation: a factor that raises risk is not proven to cause churn.
- One static snapshot; no information on past campaigns, so the effect of an offer (uplift) is assumed.
- Monetary values are hypothetical and must be replaced with the operator's real figures.
""")

with st.expander("Final hyperparameters, test metrics and library versions"):
    a, b, c = st.columns(3)
    a.dataframe(pd.Series(meta["final_params"], name="value").to_frame(), width="stretch")
    b.dataframe(pd.DataFrame(meta["test_metrics"]).T.drop(columns=["contacted"], errors="ignore")
                .style.format("{:.3f}"), width="stretch")
    c.dataframe(pd.Series(meta["versions"], name="version").to_frame(), width="stretch")

if "<your-user>" not in REPO_URL:
    st.markdown(f"Source code and notebook: [{REPO_URL}]({REPO_URL})")
