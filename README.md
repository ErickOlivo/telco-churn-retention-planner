# Retention Planner: Telco Churn Risk Intelligence

Retention Planner predicts which telecom customers are about to leave, explains each prediction in plain
language, and turns the predictions into a retention campaign plan that respects a budget.

**Live app:** `https://telco-retention-planner-erick.streamlit.app`

Built for *Aprendizaje Automático, Task 7* (Telco Customer Churn) by Erick Olivo.

## What the app does

| Page | Purpose |
|:--|:--|
| Overview | Headline metrics and how the solution works (score, explain, act). |
| Customer Explorer | Interactive EDA: filter by contract, internet, payment, tenure and senior status; KPIs, churn drivers, tenure and charges, segment matrix, data download. |
| Model Performance | Model comparison, Grid vs Random vs TPE optimization, NSGA-II Pareto front, test evaluation with an interactive threshold and confusion matrix, global importance (SHAP, LIME, permutation, Random Forest impurity). |
| Customer Risk | Form or presets to score a customer: churn probability, risk tier, worth-contacting decision, SHAP and LIME explanations, suggested retention actions. |
| Retention Planner | Business application: value-based targeting under editable (hypothetical) economics, budget limit, policy comparison with backtest, scenario sensitivity and a downloadable action list. |
| Methodology | Pipeline, validation protocol, decisions and limitations. |

## Key results (held-out test set, 1,409 customers)

| Metric | Value |
|:--|:--|
| ROC-AUC | 0.847 |
| PR-AUC | 0.660 (base rate 0.265) |
| Recall at business threshold 0.36 | 0.70 (0.52 at the default 0.5) |
| Model | LightGBM, 112 trees x 8 leaves, selected by NSGA-II with the one-standard-error rule |

Test results match cross-validation, so tuning and selection did not overfit.

## Design decisions

- **The model ranks, the business decides.** Models are compared with threshold-free metrics (PR-AUC); the
  decision threshold is set afterwards from a declared cost model.
- **No leakage.** 80/20 stratified split; the test set was used once. Cleaning rules live inside the
  exported pipeline, and every learned step is fitted within cross-validation folds.
- **Controlled optimizer comparison.** Grid, Random and TPE share the same objective, folds and budget
  (48 evaluations); Random and TPE run with 3 seeds.
- **Multi-objective selection.** NSGA-II maximizes PR-AUC and minimizes model complexity (total leaves); the
  simplest solution within one standard error of the best is selected.
- **Value-based targeting.** The planner contacts a customer when
  `probability x success rate x customer value > contact cost`, instead of a single risk threshold.
- **Inference only.** The app loads `artifacts/model.joblib`; it never retrains.

## Project structure

```
telco-churn-retention-planner/
├── app.py                  # Entry point: page config, navigation, sidebar
├── telco_churn.py          # Shared pipeline module (needed to load the model)
├── app_core/
│   ├── config.py           # Paths, palette, labels, retention playbook
│   ├── data.py             # Cached loading of model, metadata and results; scoring
│   ├── explain.py          # SHAP aggregation, LIME, plain-language explanations
│   ├── business.py         # Campaign economics and targeting policies
│   └── ui.py               # Visual system: CSS, headers, formatting, chart style
├── views/                  # One file per page
├── artifacts/
│   ├── model.joblib        # Fitted pipeline exported by the notebook
│   ├── metadata.json       # Threshold, params, metrics, schema, versions
│   ├── results/*.csv       # Experiment results shown in the app
│   └── data/*.csv          # Dataset and LIME reference sample
├── notebooks/              # Training and analysis notebook (Colab)
├── .streamlit/config.toml  # Theme
└── requirements.txt
```

## Run locally

Requires Python 3.12.

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Reproduce the artifacts

1. Open `notebooks/Telco_Churn_Optimization_XAI.ipynb` in Google Colab.
2. Place `TelcoDataset.zip` in `MyDrive/Aprendizaje_automatico/datasets/`.
3. Run all cells (about 15 minutes). Section 13 writes the artifacts to
   `MyDrive/Aprendizaje_automatico/telco_churn_artifacts/`.
4. Copy `model.joblib`, `metadata.json`, `results/` and `data/` into `artifacts/`, and `telco_churn.py` to the
   repository root.

Library versions in `requirements.txt` match the environment used to export the model; keep them aligned
when re-exporting.

## Assumptions and limitations

- Monetary figures (horizon, margin, contact cost, success rate) are **hypothetical** and editable in the app.
  They illustrate the decision logic and are not estimated from the data.
- Explanations describe how the model uses features; they do not prove causation.
- The dataset is a single snapshot with no record of past campaigns, so the effect of an offer (uplift) is
  assumed rather than measured.
- The Retention Planner uses the held-out test customers as an illustrative portfolio.

## Data

IBM Telco Customer Churn sample dataset (7,043 customers, 21 columns).
