"""Cached loading of exported artifacts. The app never retrains: it only loads and predicts."""
import json

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.model_selection import train_test_split

from app_core.config import ARTIFACTS, TEST_SIZE
from telco_churn import ID_COL, INPUT_COLUMNS, TARGET, TelcoCleaner, aggregate_by_feature


@st.cache_resource(show_spinner="Loading model")
def load_model():
    return joblib.load(ARTIFACTS / "model.joblib")


@st.cache_data
def load_metadata():
    return json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))


@st.cache_data
def load_result(name):
    return pd.read_csv(ARTIFACTS / "results" / f"{name}.csv")


@st.cache_data
def load_raw():
    return pd.read_csv(ARTIFACTS / "data" / "telco_churn.csv")


@st.cache_data
def load_clean():
    """Full dataset after the same cleaning rules used by the model."""
    raw = load_raw()
    clean = TelcoCleaner().transform(raw[INPUT_COLUMNS])
    clean.insert(0, ID_COL, raw[ID_COL])
    clean["Churn"] = (raw[TARGET] == "Yes").astype(int)
    return clean


def business_threshold():
    return float(load_metadata()["business_threshold"])


def base_rate():
    return float(load_metadata()["base_rate_train"])


@st.cache_data(show_spinner="Scoring held-out customers")
def scored_portfolio():
    """Held-out test customers (same split as the notebook), scored by the exported model."""
    raw = load_raw()
    X = raw[INPUT_COLUMNS]
    y = (raw[TARGET] == "Yes").astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                            random_state=load_metadata()["seed"])
    portfolio = TelcoCleaner().transform(X_test)
    portfolio.insert(0, ID_COL, raw.loc[X_test.index, ID_COL])
    portfolio["churn_probability"] = load_model().predict_proba(X_test)[:, 1]
    portfolio["actual_churn"] = y_test
    return portfolio


@st.cache_resource
def shap_explainer():
    return shap.TreeExplainer(load_model().named_steps["model"])


def shap_by_feature(frame):
    """SHAP values (log-odds) aggregated to the original features, plus the base value."""
    model = load_model()
    transformed = model[:-1].transform(frame[INPUT_COLUMNS])
    explanation = shap_explainer()(transformed)
    values = explanation.values[:, :, 1] if explanation.values.ndim == 3 else explanation.values
    contributions = aggregate_by_feature(values, transformed.columns)
    contributions.index = frame.index
    return contributions, float(np.ravel(explanation.base_values)[0])


@st.cache_data(show_spinner="Explaining portfolio")
def portfolio_shap():
    return shap_by_feature(scored_portfolio())


def predict(customer):
    frame = pd.DataFrame([customer])[INPUT_COLUMNS]
    return float(load_model().predict_proba(frame)[0, 1])
