"""SHAP and LIME helpers producing business-readable explanations."""
import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

from app_core.config import ARTIFACTS, FEATURE_LABELS, PLAYBOOK
from app_core.data import load_model
from telco_churn import CATEGORICAL_INPUTS, INPUT_COLUMNS

CAT_IDX = [INPUT_COLUMNS.index(c) for c in CATEGORICAL_INPUTS]


def format_value(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return str(value)


def readable(feature, value):
    return f"{FEATURE_LABELS.get(feature, feature)}: {format_value(value)}"


def to_probability(log_odds):
    return 1 / (1 + np.exp(-log_odds))


def recommend(contributions, customer, limit=2):
    """Actions from the playbook for the strongest risk-increasing drivers of this customer."""
    actions = []
    for feature in contributions[contributions > 0].sort_values(ascending=False).index:
        for rule_feature, condition, action in PLAYBOOK:
            if rule_feature == feature and condition(customer[feature]) and action not in actions:
                actions.append(action)
        if len(actions) >= limit:
            break
    return actions


def main_driver(contributions, customer):
    feature = contributions.idxmax()
    return readable(feature, customer[feature]) if contributions.max() > 0 else "No strong risk driver"


@st.cache_resource(show_spinner="Preparing LIME")
def lime_setup():
    reference = pd.read_csv(ARTIFACTS / "data" / "lime_reference.csv")[INPUT_COLUMNS]
    categories = {c: sorted(reference[c].astype(str).unique()) for c in CATEGORICAL_INPUTS}
    explainer = LimeTabularExplainer(
        _encode(reference, categories), feature_names=INPUT_COLUMNS, class_names=["Stays", "Churns"],
        categorical_features=CAT_IDX,
        categorical_names={i: categories[c] for i, c in zip(CAT_IDX, CATEGORICAL_INPUTS)},
        discretize_continuous=True, mode="classification", random_state=42)
    return explainer, categories


def _encode(frame, categories):
    out = frame[INPUT_COLUMNS].copy()
    for c in CATEGORICAL_INPUTS:
        out[c] = out[c].astype(str).map({v: i for i, v in enumerate(categories[c])})
    return out.astype(float).to_numpy()


def lime_explain(customer, num_samples=4000):
    """Returns LIME weights (rule -> weight toward churn) and the surrogate's local prediction."""
    explainer, categories = lime_setup()
    model = load_model()

    def predict_fn(array):
        frame = pd.DataFrame(array, columns=INPUT_COLUMNS)
        for c in CATEGORICAL_INPUTS:
            frame[c] = frame[c].round().astype(int).map(dict(enumerate(categories[c])))
        return model.predict_proba(frame)

    row = _encode(pd.DataFrame([customer]), categories)[0]
    explanation = explainer.explain_instance(row, predict_fn, num_features=10, num_samples=num_samples,
                                             labels=(1,))
    weights = pd.Series(dict(explanation.as_list(label=1)))
    return weights, float(explanation.local_pred[0])


def lime_feature(rule):
    return max((c for c in INPUT_COLUMNS if c in rule), key=len)
