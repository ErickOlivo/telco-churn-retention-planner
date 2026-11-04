import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_core import ui
from app_core.business import plan, sensitivity, summarize
from app_core.config import COLORS, DEFAULT_ECONOMICS
from app_core.data import business_threshold, portfolio_shap, scored_portfolio
from app_core.explain import main_driver, recommend

portfolio = scored_portfolio()
t_star = business_threshold()

ui.page_header(
    "Retention Planner",
    f"Decide whom to contact this month. The {len(portfolio):,} held-out customers act as the current portfolio: "
    "the model scores them, and the planner contacts a customer only if the expected value saved exceeds the "
    "contact cost.", eyebrow="Business application")
ui.hypothetical_notice()

with st.container(border=True):
    st.markdown("**Campaign assumptions**")
    a, b, c, d = st.columns(4)
    horizon = a.slider("Value horizon (months)", 3, 24, DEFAULT_ECONOMICS["horizon_months"],
                       help="Months of margin preserved when a churner is retained.")
    margin = b.slider("Gross margin", 0.2, 0.8, DEFAULT_ECONOMICS["gross_margin"], 0.05, format="%.2f")
    cost = c.slider("Cost per contact (USD)", 5, 120, int(DEFAULT_ECONOMICS["contact_cost"]), 5,
                    help="Agent time plus the average cost of the incentive offered.")
    success = d.slider("Retention success rate", 0.05, 0.7, DEFAULT_ECONOMICS["success_rate"], 0.05, format="%.2f",
                       help="Share of contacted churners who accept the offer and stay.")
    e, f, g = st.columns([1.4, 1, 1], vertical_alignment="bottom")
    policy_label = e.segmented_control(
        "Targeting policy", ["Value-based", "Risk threshold"], default="Value-based",
        help="Value-based: contact when probability × success × customer value > cost. "
             "Risk threshold: contact everyone above a probability cut-off.")
    threshold = f.slider("Risk threshold", 0.05, 0.95, round(t_star, 2), 0.01,
                         disabled=policy_label != "Risk threshold")
    use_budget = g.toggle("Limit budget")
    budget = g.number_input("Budget (USD)", 500, 100_000, 8_000, 500, disabled=not use_budget,
                            label_visibility="collapsed")

econ = {"horizon_months": horizon, "gross_margin": margin, "contact_cost": float(cost), "success_rate": success}
policy = "value" if policy_label != "Risk threshold" else "threshold"
result = plan(portfolio, econ, policy=policy, threshold=threshold, budget=budget if use_budget else None)
stats = summarize(result, econ)

k = st.columns(5)
k[0].metric("Contacts", f"{stats['contacts']:,}", f"{stats['contacts'] / len(result):.0%} of base",
            delta_color="off", border=True)
k[1].metric("Cost", ui.money(stats["cost"]), border=True)
k[2].metric("Retained (exp.)", f"{stats['expected_saved_customers']:.0f}", border=True,
            help="Sum over contacted customers of churn probability × success rate.")
k[3].metric("Net value (exp.)", ui.money(stats["expected_net"]), border=True,
            help="Estimated at decision time from model probabilities; no outcomes are used.")
k[4].metric("ROI (exp.)", f"{stats['expected_roi']:.0%}" if stats["cost"] else "n/a", border=True,
            help="Expected net value divided by campaign cost.")

tab_plan, tab_policies, tab_sensitivity, tab_list = st.tabs(
    ["Who to contact", "Policy comparison", "Scenario sensitivity", "Action list"])

with tab_plan:
    left, right = st.columns(2)
    with left:
        prob = np.linspace(0.01, 1, 200)
        boundary = econ["contact_cost"] / (prob * econ["success_rate"])
        sample = result.assign(Decision=np.where(result["contact"], "Contact", "Do not contact"))
        fig = px.scatter(sample, x="churn_probability", y="customer_value", color="Decision", opacity=0.7,
                         color_discrete_map={"Contact": COLORS["churn"], "Do not contact": "#B7C2CE"},
                         hover_data={"customerID": True, "churn_probability": ":.1%", "customer_value": ":,.0f",
                                     "expected_net": ":,.0f", "Decision": False})
        fig.update_traces(marker=dict(size=6))
        if policy == "value":
            fig.add_trace(go.Scatter(x=prob, y=boundary, mode="lines", name="Break-even",
                                     line=dict(color=COLORS["ink"], dash="dash"), hoverinfo="skip"))
        else:
            fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["ink"])
        fig.update_xaxes(title="Churn probability", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="Customer value (USD)",
                         range=[0, result["customer_value"].max() * 1.08])
        ui.show(ui.style_fig(fig, height=420, title="Risk × value: the decision boundary"))
        ui.note("Value-based targeting contacts high-value customers at lower risk and skips low-value customers "
                "even at high risk. A single threshold ignores value.")
    with right:
        ranked = result.sort_values("expected_net", ascending=False)
        cumulative = ranked["expected_net"].cumsum().to_numpy()
        n = np.arange(1, len(ranked) + 1)
        fig = go.Figure(go.Scatter(x=n, y=cumulative, line=dict(color=COLORS["primary"], width=3),
                                   hovertemplate="Top %{x} customers: %{y:$,.0f}<extra></extra>",
                                   name="Cumulative expected net value"))
        fig.add_vline(x=stats["contacts"], line_dash="dash", line_color=COLORS["churn"],
                      annotation_text=f"Plan: {stats['contacts']:,}", annotation_position="top left")
        fig.add_hline(y=0, line_color=COLORS["muted"])
        fig.update_xaxes(title="Customers contacted, best expected value first")
        fig.update_yaxes(title="Expected net value (USD)", tickprefix="$")
        ui.show(ui.style_fig(fig, height=420, legend=False, title="Diminishing returns of contacting more"))
        ui.note("The curve peaks where the next customer's expected gain no longer covers the contact cost.")

    contacted = result[result["contact"]]
    if len(contacted):
        mix = (contacted.groupby(["Contract", "InternetService"]).size().rename("customers").reset_index())
        fig = px.bar(mix, x="customers", y="Contract", color="InternetService", orientation="h",
                     color_discrete_sequence=[COLORS["stay"], COLORS["churn"], "#B7C2CE"])
        fig.update_xaxes(title="Customers in the plan")
        fig.update_yaxes(title=None)
        ui.show(ui.style_fig(fig, height=260, title="Who is in the plan"))

with tab_policies:
    rng = np.random.default_rng(7)
    rows = []
    candidates = {
        "Value-based": plan(portfolio, econ, "value"),
        f"Risk threshold {t_star:.2f}": plan(portfolio, econ, "threshold", t_star),
        "Contact everyone": plan(portfolio, econ, "threshold", 0.0),
    }
    for name, df in candidates.items():
        s = summarize(df, econ)
        rows.append({"Policy": name, "Contacts": s["contacts"], "Expected net value": s["expected_net"],
                     "Backtest net value": s["realized_net"], "Churners reached": s["actual_churners_reached"]})
    size = rows[0]["Contacts"]
    value_col = candidates["Value-based"]
    random_net = np.mean([value_col.loc[rng.choice(value_col.index, size, replace=False), "realized_net"].sum()
                          for _ in range(200)]) if size else 0.0
    rows.append({"Policy": "Random (same size)", "Contacts": size,
                 "Expected net value": np.nan, "Backtest net value": random_net,
                 "Churners reached": round(size * portfolio["actual_churn"].mean())})
    table = pd.DataFrame(rows)

    fig = go.Figure([
        go.Bar(name="Expected (decision time)", x=table["Policy"], y=table["Expected net value"],
               marker_color=COLORS["stay"], hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"),
        go.Bar(name="Backtest (actual outcomes)", x=table["Policy"], y=table["Backtest net value"],
               marker_color=COLORS["primary"], hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"),
    ])
    fig.add_hline(y=0, line_color=COLORS["ink"])
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Net value (USD)", tickprefix="$")
    ui.show(ui.style_fig(fig, height=400, title="Same assumptions, different targeting rules"))
    st.dataframe(table.style.format({"Expected net value": "${:,.0f}", "Backtest net value": "${:,.0f}"},
                                    na_rep="-"), hide_index=True, width="stretch")
    ui.note("Backtest uses the real outcome of each held-out customer, which is unknown when the campaign is "
            "planned. It checks that the expected value is realistic. Random targeting shows what the model adds.")

with tab_sensitivity:
    rates = np.round(np.arange(0.10, 0.55, 0.05), 2)
    costs = np.arange(10, 130, 10)
    grid = sensitivity(portfolio, rates, costs, econ)
    fig = go.Figure(go.Heatmap(
        z=grid.values / 1000, x=[f"{r:.0%}" for r in rates], y=[f"${c}" for c in costs],
        colorscale=[[0, "#F4F7F6"], [1, COLORS["primary"]]], colorbar=dict(title="USD k"),
        text=[[f"{v / 1000:.1f}" for v in row] for row in grid.values], texttemplate="%{text}",
        hovertemplate="Success %{x}, cost %{y}: %{z:.1f} k<extra></extra>"))
    fig.update_xaxes(title="Retention success rate")
    fig.update_yaxes(title="Cost per contact", autorange="reversed")
    ui.show(ui.style_fig(fig, height=440, legend=False,
                         title=f"Expected net value of the value-based plan (USD k, horizon {horizon} months, "
                               f"margin {margin:.0%})"))
    ui.note("Use this before negotiating: it shows how robust the campaign value is if the offer costs more or "
            "converts fewer customers than assumed. The value-based rule never contacts at a loss in expectation, "
            "so the plan shrinks instead of turning negative.")

with tab_list:
    contacted = result[result["contact"]].sort_values("priority_rank")
    if contacted.empty:
        st.warning("No customer is worth contacting under these assumptions.", icon=":material/block:")
    else:
        shap_values, _ = portfolio_shap()
        drivers, actions = [], []
        for idx, row in contacted.iterrows():
            drivers.append(main_driver(shap_values.loc[idx], row))
            actions.append("; ".join(recommend(shap_values.loc[idx], row)) or "General retention offer")
        action_list = pd.DataFrame({
            "Priority": contacted["priority_rank"].astype(int).values,
            "Customer": contacted["customerID"].values,
            "Churn risk": contacted["churn_probability"].values,
            "Customer value": contacted["customer_value"].values,
            "Expected net": contacted["expected_net"].values,
            "Contract": contacted["Contract"].values,
            "Tenure": contacted["tenure"].astype(int).values,
            "Main risk driver": drivers,
            "Recommended action": actions,
        })
        st.dataframe(action_list, hide_index=True, width="stretch", height=430, column_config={
            "Churn risk": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Customer value": st.column_config.NumberColumn(format="$%.0f"),
            "Expected net": st.column_config.NumberColumn(format="$%.0f"),
        })
        st.download_button("Download action list (CSV)", action_list.to_csv(index=False).encode("utf-8"),
                           file_name="retention_action_list.csv", mime="text/csv", icon=":material/download:")
        ui.note("Main driver: the factor with the largest SHAP contribution toward churn for that customer. "
                "Actions come from a hypothetical retention playbook mapped to those drivers.")
