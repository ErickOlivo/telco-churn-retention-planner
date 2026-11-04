import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_core import ui
from app_core.config import COLORS
from app_core.data import business_threshold, load_clean, load_metadata, scored_portfolio

meta = load_metadata()
clean = load_clean()
portfolio = scored_portfolio()
t_star = business_threshold()
test = meta["test_metrics"]
at_star = next(v for k, v in test.items() if k.startswith("test, t*"))
at_half = test["test, t = 0.50"]

ui.page_header(
    "Find the customers about to leave, and know why",
    "Retention Planner scores every customer's churn risk, explains each score in plain language and turns "
    "the scores into a campaign plan that respects your budget.",
    eyebrow="Telco churn · decision support")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers analyzed", f"{len(clean):,}", border=True)
k2.metric("Historical churn rate", ui.pct(clean["Churn"].mean()), border=True,
          help="Share of customers who left in the last month of the dataset.")
k3.metric("ROC-AUC on test", f"{at_star['roc_auc']:.3f}", border=True,
          help="Measured once on 1,409 held-out customers never used for training or tuning.")
k4.metric("Churners caught", ui.pct(at_star["recall"], 0), border=True,
          delta=f"+{100 * (at_star['recall'] - at_half['recall']):.0f} pp vs t = 0.50",
          help=f"Recall on held-out customers at the business threshold {t_star:.2f}.")

st.write("")
left, right = st.columns([1, 1.25], gap="large")
with left:
    st.markdown("#### How it works")
    steps = [
        ("Score", "A LightGBM model ranks customers by churn probability. Probabilities are calibrated, "
                  "so 30 % means roughly 30 of 100 similar customers leave.", "views/risk.py", "Score a customer"),
        ("Explain", "SHAP and LIME show which factors raise or lower each customer's risk, in business terms.",
         "views/risk.py", "See an explanation"),
        ("Act", "The planner contacts customers only when the expected value saved exceeds the cost, "
                "within your budget.", "views/planner.py", "Plan a campaign"),
    ]
    for i, (title, body, page, link) in enumerate(steps, start=1):
        with st.container(border=True):
            st.markdown(f'<div class="rp-step">STEP {i}</div><h5 style="margin:2px 0 4px 0">{title}</h5>'
                        f'<p class="rp-note" style="font-size:0.9rem">{body}</p>', unsafe_allow_html=True)
            st.page_link(page, label=link, icon=":material/arrow_forward:")

with right:
    st.markdown("#### Why ranking matters")
    order = np.argsort(-portfolio["churn_probability"].to_numpy())
    actual = portfolio["actual_churn"].to_numpy()[order]
    share = np.arange(1, len(order) + 1) / len(order)
    captured = np.cumsum(actual) / actual.sum()
    contacted = (portfolio["churn_probability"] >= t_star).mean()
    captured_at = captured[int(contacted * len(order)) - 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=share, y=share, name="Random targeting", line=dict(color=COLORS["muted"], dash="dash"),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=share, y=captured, name="Retention Planner", line=dict(color=COLORS["primary"], width=3),
                             fill="tonexty", fillcolor="rgba(15,118,110,0.08)",
                             hovertemplate="Contact %{x:.0%} → catch %{y:.0%} of churners<extra></extra>"))
    fig.add_trace(go.Scatter(x=[contacted], y=[captured_at], mode="markers+text", name="Business threshold",
                             marker=dict(size=12, color=COLORS["churn"]),
                             text=[f"  Contact {contacted:.0%} → catch {captured_at:.0%}"], textposition="middle right",
                             hoverinfo="skip"))
    fig.update_xaxes(tickformat=".0%", title="Share of customers contacted (highest risk first)")
    fig.update_yaxes(tickformat=".0%", title="Share of churners captured")
    ui.show(ui.style_fig(fig, height=400))
    ui.note("Held-out customers. Random targeting catches churners in proportion to the share contacted; "
            "the model concentrates them at the top of the list.")

st.write("")
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Model**  \nLightGBM selected by multi-objective optimization: same ranking quality as the "
                f"best model with far fewer leaves.")
    c2.markdown(f"**Evidence**  \nTest PR-AUC {at_star['pr_auc']:.3f}, consistent with cross-validation. "
                "No retraining happens in this app.")
    c3.markdown("**Caveat**  \nPredictions show association, not causation. Monetary figures rely on declared "
                "hypothetical assumptions.")
