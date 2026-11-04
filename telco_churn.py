"""Shared components for the Telco churn project (training notebook and Streamlit app)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC_INPUTS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_INPUTS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
INPUT_COLUMNS = CATEGORICAL_INPUTS[:4] + ["tenure"] + CATEGORICAL_INPUTS[4:] + ["MonthlyCharges", "TotalCharges"]
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                   "StreamingTV", "StreamingMovies"]
ENGINEERED = ["avg_monthly_spend", "price_increase", "n_services"]


class TelcoCleaner(BaseEstimator, TransformerMixin):
    """Deterministic cleaning rules derived from the EDA. Learns nothing from data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        total = pd.to_numeric(X["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce")
        X["TotalCharges"] = total.where(~(total.isna() & (X["tenure"] == 0)), 0.0).astype(float)
        X["tenure"] = X["tenure"].astype(float)
        X["MonthlyCharges"] = X["MonthlyCharges"].astype(float)
        X["SeniorCitizen"] = X["SeniorCitizen"].replace({0: "No", 1: "Yes", "0": "No", "1": "Yes"}).astype(str)
        X[INTERNET_ADDONS] = X[INTERNET_ADDONS].replace("No internet service", "No")
        X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")
        return X


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds three business-readable features. Set enabled=False for the ablation study."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.enabled:
            return X
        X = X.copy()
        has_history = X["tenure"] > 0
        X["avg_monthly_spend"] = np.where(has_history, X["TotalCharges"] / X["tenure"].where(has_history, 1),
                                          X["MonthlyCharges"])
        X["price_increase"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
        services = (X[["PhoneService", "MultipleLines"] + INTERNET_ADDONS] == "Yes").sum(axis=1)
        X["n_services"] = (services + (X["InternetService"] != "No")).astype(float)
        return X


def _combine(feature, category):
    return f"{feature}={str(category).replace(' ', '_')}"


def build_preprocessor(scale=False):
    numeric = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False,
                                 feature_name_combiner=_combine)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric), make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ], verbose_feature_names_out=False).set_output(transform="pandas")


def build_pipeline(model, engineered=True, scale=False):
    return Pipeline([
        ("clean", TelcoCleaner()),
        ("features", FeatureBuilder(enabled=engineered)),
        ("prep", build_preprocessor(scale=scale)),
        ("model", model),
    ])


def original_feature(name):
    """Map a transformed column (e.g. 'Contract=Month-to-month') to its original feature."""
    return name.split("=")[0]


def aggregate_by_feature(values, feature_names):
    """Sum contributions of one-hot columns back to their original feature."""
    frame = pd.DataFrame(np.atleast_2d(values), columns=list(feature_names))
    return frame.T.groupby(original_feature).sum().T


def _fmt(value):
    return f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, (int, float, np.floating)) else str(value)


def explain_in_words(contributions, customer, top=3, min_effect=0.05):
    """Plain-language summary from per-feature contributions (positive = raises churn risk).

    Contributions below `min_effect` (log-odds) are ignored as negligible.
    """
    ordered = contributions.sort_values(ascending=False)
    up = [f"{f} = {_fmt(customer.get(f, ''))}" for f, v in ordered.items() if v >= min_effect][:top]
    down = [f"{f} = {_fmt(customer.get(f, ''))}" for f, v in ordered[::-1].items() if v <= -min_effect][:top]
    text = []
    if up:
        text.append("Risk increases mainly because of: " + "; ".join(up) + ".")
    if down:
        text.append("Risk decreases because of: " + "; ".join(down) + ".")
    return " ".join(text)
