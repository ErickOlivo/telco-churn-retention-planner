"""Static configuration: paths, palette, labels and the retention playbook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPO_URL = "https://github.com/<your-user>/telco-churn-retention-planner"

TEST_SIZE = 0.20  # Must match the notebook split.

COLORS = {
    "primary": "#0F766E",
    "primary_soft": "#CCE7E4",
    "churn": "#E4572E",
    "churn_soft": "#F9D9CF",
    "stay": "#4C6A92",
    "stay_soft": "#DCE4EF",
    "ink": "#0B1F33",
    "muted": "#6B7A8C",
    "grid": "#E6EBF0",
    "positive": "#15803D",
    "accent": "#D9A11B",
}
STRATEGY_COLORS = {"Grid": "#4C6A92", "Random": "#8DA2B8", "TPE": "#0F766E", "NSGA-II": "#D9A11B"}
MODEL_COLORS = {"Dummy": "#C3CCD6", "Logistic Regression": "#4C6A92", "Random Forest": "#8DA2B8",
                "LightGBM": "#0F766E"}

DEFAULT_ECONOMICS = {"horizon_months": 12, "gross_margin": 0.50, "contact_cost": 40.0, "success_rate": 0.30}

FEATURE_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}

SEARCH_SPACE = [
    ("n_estimators", "int", "100 - 600", "linear", "150, 400"),
    ("learning_rate", "float", "0.01 - 0.2", "log", "0.03, 0.1"),
    ("num_leaves", "int", "8 - 64", "linear", "8, 16, 32"),
    ("min_child_samples", "int", "10 - 100", "linear", "20, 60"),
    ("colsample_bytree", "float", "0.6 - 1.0", "linear", "0.6, 0.9"),
    ("subsample", "float", "0.6 - 1.0", "linear", "fixed 1.0"),
    ("reg_lambda", "float", "0.001 - 10", "log", "fixed 0.0"),
]

# Retention playbook: (feature, condition on the customer's value, action). Hypothetical business rules.
PLAYBOOK = [
    ("Contract", lambda v: v == "Month-to-month", "Offer a 12-month contract with a loyalty discount"),
    ("tenure", lambda v: float(v) < 12, "Schedule an onboarding check-in call"),
    ("PaymentMethod", lambda v: v == "Electronic check", "Incentivize automatic payment (card or bank transfer)"),
    ("TechSupport", lambda v: v == "No", "Offer a free 3-month Tech Support trial"),
    ("OnlineSecurity", lambda v: v == "No", "Offer a free 3-month Online Security trial"),
    ("InternetService", lambda v: v == "Fiber optic", "Review fiber plan price and service quality"),
    ("MonthlyCharges", lambda v: float(v) > 70, "Run a plan right-sizing review"),
    ("PaperlessBilling", lambda v: v == "Yes", "Send a personalized bill explanation"),
]
