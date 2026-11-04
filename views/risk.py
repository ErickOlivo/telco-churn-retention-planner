import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_core import ui
from app_core.business import customer_value, expected_net_value
from app_core.config import COLORS, DEFAULT_ECONOMICS, FEATURE_LABELS
from app_core.data import base_rate, business_threshold, predict, scored_portfolio, shap_by_feature
from app_core.explain import (lime_explain, lime_feature, readable, recommend, to_probability)
from telco_churn import INPUT_COLUMNS, INTERNET_ADDONS

t_star = business_threshold()
avg_risk = base_rate()

PRESETS = {
    "Typical": dict(gender="Female", SeniorCitizen="No", Partner="Yes", Dependents="No", tenure=29,
                             PhoneService="Yes", MultipleLines="No", InternetService="DSL", OnlineSecurity="Yes",
                             OnlineBackup="No", DeviceProtection="No", TechSupport="No", StreamingTV="No",
                             StreamingMovies="No", Contract="One year", PaperlessBilling="Yes",
                             PaymentMethod="Credit card (automatic)", MonthlyCharges=65.0, TotalCharges=1885.0),
    "High risk": dict(gender="Male", SeniorCitizen="Yes", Partner="Yes", Dependents="No", tenure=1,
                               PhoneService="Yes", MultipleLines="Yes", InternetService="Fiber optic",
                               OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
                               StreamingTV="Yes", StreamingMovies="Yes", Contract="Month-to-month",
                               PaperlessBilling="Yes", PaymentMethod="Electronic check", MonthlyCharges=95.1,
                               TotalCharges=95.1),
    "Loyal": dict(gender="Male", SeniorCitizen="No", Partner="Yes", Dependents="Yes", tenure=55,
                            PhoneService="Yes", MultipleLines="No", InternetService="DSL", OnlineSecurity="Yes",
                            OnlineBackup="Yes", DeviceProtection="No", TechSupport="Yes", StreamingTV="No",
                            StreamingMovies="No", Contract="Two year", PaperlessBilling="No",
                            PaymentMethod="Bank transfer (automatic)", MonthlyCharges=57.55, TotalCharges=3046.4),
}
OPTIONS = {
    "gender": ["Female", "Male"], "SeniorCitizen": ["No", "Yes"], "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"], "PhoneService": ["No", "Yes"], "MultipleLines": ["No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"], "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    **{c: ["No", "Yes"] for c in INTERNET_ADDONS},
}


def load_profile(values, auto_total=False, source=None):
    for col, value in values.items():
        st.session_state[f"c_{col}"] = float(value) if col in ("MonthlyCharges", "TotalCharges") else value
    st.session_state["c_tenure"] = int(values["tenure"])
    st.session_state["c_auto_total"] = auto_total
    st.session_state["c_source"] = source


def on_preset():
    choice = st.session_state.get("c_preset")
    if choice:
        load_profile(PRESETS[choice], source=None)


def on_random():
    row = scored_portfolio().sample(1).iloc[0]
    load_profile({c: row[c] for c in INPUT_COLUMNS},
                 source={"id": row["customerID"], "actual": int(row["actual_churn"])})
    st.session_state["c_preset"] = None


if "c_tenure" not in st.session_state:
    load_profile(PRESETS["Typical"], auto_total=True)
    st.session_state["c_preset"] = "Typical"

ui.page_header("Customer Risk",
               "Enter or load a customer profile to get the churn probability, the factors behind it and a "
               "recommended retention action.", eyebrow="Prediction and explainability")

left, right = st.columns([0.95, 1.25], gap="large")

with left:
    p1, p2 = st.columns([1.6, 1], vertical_alignment="bottom")
    p1.segmented_control("Start from a profile", list(PRESETS), key="c_preset", on_change=on_preset)
    p2.button("Test customer", on_click=on_random, icon=":material/shuffle:", width="stretch",
              help="Load a real customer from the test set, never seen during training.")

    with st.container(border=True):
        st.markdown("**Contract and billing**")
        a, b = st.columns(2)
        a.selectbox("Contract", OPTIONS["Contract"], key="c_Contract")
        b.selectbox("Payment method", OPTIONS["PaymentMethod"], key="c_PaymentMethod")
        a.slider("Tenure (months)", 0, 72, key="c_tenure")
        b.number_input("Monthly charges (USD)", 18.0, 120.0, step=0.5, key="c_MonthlyCharges")
        a.radio("Paperless billing", OPTIONS["PaperlessBilling"], key="c_PaperlessBilling", horizontal=True)
        auto_total = b.toggle("Auto total", key="c_auto_total",
                              help="Uses tenure × monthly charges. Turn off to enter the billed amount.")
        if auto_total:
            st.session_state["c_TotalCharges"] = float(st.session_state.c_tenure * st.session_state.c_MonthlyCharges)
        b.number_input("Total charges (USD)", 0.0, 9000.0, step=10.0, key="c_TotalCharges", disabled=auto_total)

    with st.container(border=True):
        st.markdown("**Services**")
        a, b = st.columns(2)
        a.radio("Phone service", OPTIONS["PhoneService"], key="c_PhoneService", horizontal=True)
        if st.session_state.c_PhoneService == "No":
            st.session_state["c_MultipleLines"] = "No"
        b.radio("Multiple lines", OPTIONS["MultipleLines"], key="c_MultipleLines", horizontal=True,
                disabled=st.session_state.c_PhoneService == "No")
        st.selectbox("Internet service", OPTIONS["InternetService"], key="c_InternetService")
        no_internet = st.session_state.c_InternetService == "No"
        cols = st.columns(3)
        for i, col in enumerate(INTERNET_ADDONS):
            if no_internet:
                st.session_state[f"c_{col}"] = "No"
            cols[i % 3].selectbox(FEATURE_LABELS[col], OPTIONS[col], key=f"c_{col}", disabled=no_internet)
        if no_internet:
            ui.note("Add-on services require internet service.")

    with st.container(border=True):
        st.markdown("**Profile**")
        a, b = st.columns(2)
        a.radio("Gender", OPTIONS["gender"], key="c_gender", horizontal=True)
        b.radio("Senior citizen", OPTIONS["SeniorCitizen"], key="c_SeniorCitizen", horizontal=True)
        a.radio("Partner", OPTIONS["Partner"], key="c_Partner", horizontal=True)
        b.radio("Dependents", OPTIONS["Dependents"], key="c_Dependents", horizontal=True)

customer = {col: st.session_state[f"c_{col}"] for col in INPUT_COLUMNS}
probability = predict(customer)
contributions, base_value = shap_by_feature(pd.DataFrame([customer]))
contributions = contributions.iloc[0]

if probability >= t_star:
    tier, tone, color = "High risk", "high", COLORS["churn"]
elif probability >= t_star / 2:
    tier, tone, color = "Moderate risk", "medium", COLORS["accent"]
else:
    tier, tone, color = "Low risk", "low", COLORS["primary"]

value = customer_value(customer["MonthlyCharges"], DEFAULT_ECONOMICS)
net = expected_net_value(probability, value, DEFAULT_ECONOMICS)

with right:
    source = st.session_state.get("c_source")
    if source:
        outcome = "churned" if source["actual"] else "stayed"
        st.caption(f"Held-out customer {source['id']}: in reality this customer **{outcome}**.")

    r1, r2 = st.columns([1, 1.15])
    with r1:
        st.markdown(
            f'<div class="rp-risk"><div class="label">Churn probability</div>'
            f'<div class="value" style="color:{color}">{probability:.0%}</div>'
            f'{ui.badge(tier, tone)}<div class="label" style="margin-top:10px">'
            f'{probability / avg_risk:.1f}× the historical churn rate ({avg_risk:.1%})</div></div>',
            unsafe_allow_html=True)
    with r2:
        gauge = go.Figure(go.Indicator(
            mode="gauge", value=100 * probability,
            gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=color, thickness=0.35),
                       steps=[dict(range=[0, 50 * t_star], color="#EAF4F3"),
                              dict(range=[50 * t_star, 100 * t_star], color="#FBF3DF"),
                              dict(range=[100 * t_star, 100], color="#FBE6DF")],
                       threshold=dict(line=dict(color=COLORS["ink"], width=3), value=100 * t_star))))
        gauge.update_layout(height=170, margin=dict(l=38, r=38, t=24, b=4), font=dict(family="Inter", size=11))
        ui.show(gauge)
        ui.note(f"Black line: contact threshold ({t_star:.0%}).")

    with st.container(border=True):
        decision = "Worth contacting" if net > 0 else "Not worth contacting now"
        st.markdown(f"**{decision}.** Customer value over 12 months ≈ {ui.md_money(value)}; expected net value of a "
                    f"retention offer ≈ **{ui.md_money(net)}** (default hypothetical economics).")
        ups = contributions[contributions >= 0.05].sort_values(ascending=False).head(3)
        downs = contributions[contributions <= -0.05].sort_values().head(3)
        if len(ups):
            st.markdown("**Raises risk:** " + "; ".join(readable(f, customer[f]) for f in ups.index))
        if len(downs):
            st.markdown("**Lowers risk:** " + "; ".join(readable(f, customer[f]) for f in downs.index))
        actions = recommend(contributions, customer)
        if actions and probability >= t_star / 2:
            st.markdown("**Suggested actions:** " + "; ".join(actions) + ".")

    tab_shap, tab_lime, tab_compare = st.tabs(["Why: SHAP", "Local check: LIME", "SHAP vs LIME"])
    with tab_shap:
        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
        shown = top.head(8)
        rest = top.iloc[8:].sum()
        bars = pd.Series(list(shown.values) + [rest],
                         index=[readable(f, customer[f]) for f in shown.index] + ["Other features"])
        bars = bars.iloc[::-1]
        limit = max(0.4, bars.abs().max() * 1.35)
        fig = go.Figure(go.Bar(
            x=bars.values, y=bars.index, orientation="h",
            marker_color=[COLORS["churn"] if v > 0 else COLORS["primary"] for v in bars.values],
            text=[f"{v:+.2f}" for v in bars.values], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
        fig.add_vline(x=0, line_color=COLORS["ink"], line_width=1)
        fig.update_xaxes(range=[-limit, limit], title="← lowers risk     log-odds     raises risk →")
        ui.show(ui.style_fig(fig, height=420, legend=False,
                             title=f"From the average score ({to_probability(base_value):.0%}) to this customer "
                                   f"({probability:.0%})"))
        ui.note("SHAP contributions add up exactly: average log-odds plus all bars equals this customer's score. "
                "Orange raises risk, teal lowers it.")

    signature = tuple(customer.values())
    with tab_lime:
        st.markdown("LIME fits a simple local model around this customer by testing thousands of small variations.")
        if st.button("Run LIME explanation", icon=":material/play_arrow:"):
            with st.spinner("Sampling 4,000 variations"):
                st.session_state["lime_result"] = (signature, *lime_explain(customer))
        result = st.session_state.get("lime_result")
        if result and result[0] == signature:
            _, weights, local_pred = result
            w = weights.iloc[::-1]
            fig = go.Figure(go.Bar(x=w.values, y=w.index, orientation="h",
                                   marker_color=[COLORS["churn"] if v > 0 else COLORS["primary"] for v in w.values],
                                   hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
            fig.update_xaxes(title="Weight toward churn (probability scale)")
            ui.show(ui.style_fig(fig, height=400, legend=False))
            ui.note(f"Local surrogate prediction {local_pred:.0%} vs model {probability:.0%}. A large gap means the "
                    "linear approximation is rough for this customer.")
        else:
            ui.note("Run LIME to compare it with SHAP. It takes a couple of seconds.")

    with tab_compare:
        result = st.session_state.get("lime_result")
        if result and result[0] == signature:
            lime_rank = pd.Series({lime_feature(r): abs(v) for r, v in result[1].items()}).rank(ascending=False)
            shap_rank = contributions.abs().rank(ascending=False)
            top_features = shap_rank.nsmallest(6).index
            table = pd.DataFrame({
                "Feature": [FEATURE_LABELS[f] for f in top_features],
                "SHAP rank": shap_rank[top_features].astype(int).values,
                "LIME rank": [int(lime_rank[f]) if f in lime_rank else None for f in top_features],
                "Direction (SHAP)": ["raises risk" if contributions[f] > 0 else "lowers risk" for f in top_features],
            })
            st.dataframe(table, hide_index=True, width="stretch")
            ui.note("SHAP is exact for this model and additive; LIME is an approximation that is easier to read as "
                    "rules. Agreement on the top factors increases confidence in the explanation.")
        else:
            ui.note("Run LIME first in the previous tab.")
