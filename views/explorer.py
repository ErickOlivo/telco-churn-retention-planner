import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_core import ui
from app_core.config import COLORS, FEATURE_LABELS
from app_core.data import load_clean
from telco_churn import CATEGORICAL_INPUTS

data = load_clean()
overall_rate = data["Churn"].mean()

ui.page_header("Customer Explorer",
               "Slice the customer base and see where churn concentrates. All 7,043 customers, after the "
               "same cleaning rules used by the model.", eyebrow="Exploratory analysis")

FILTER_KEYS = {"f_contract": "Contract", "f_internet": "InternetService", "f_payment": "PaymentMethod"}


def reset_filters():
    for key, col in FILTER_KEYS.items():
        st.session_state[key] = sorted(data[col].unique())
    st.session_state["f_tenure"] = (0, 72)
    st.session_state["f_senior"] = "All"


if "f_contract" not in st.session_state:
    reset_filters()

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.multiselect("Contract", sorted(data["Contract"].unique()), key="f_contract")
    c2.multiselect("Internet service", sorted(data["InternetService"].unique()), key="f_internet")
    c3.multiselect("Payment method", sorted(data["PaymentMethod"].unique()), key="f_payment")
    c4, c5, c6 = st.columns([2, 1.3, 0.7], vertical_alignment="bottom")
    c4.slider("Tenure (months)", 0, 72, key="f_tenure")
    c5.segmented_control("Senior citizen", ["All", "Yes", "No"], key="f_senior")
    c6.button("Reset", on_click=reset_filters, icon=":material/restart_alt:", width="stretch")

mask = (data["Contract"].isin(st.session_state.f_contract)
        & data["InternetService"].isin(st.session_state.f_internet)
        & data["PaymentMethod"].isin(st.session_state.f_payment)
        & data["tenure"].between(*st.session_state.f_tenure))
if st.session_state.f_senior in ("Yes", "No"):
    mask &= data["SeniorCitizen"] == st.session_state.f_senior
segment = data[mask]

if segment.empty:
    st.warning("No customers match these filters. Widen the selection or press Reset.", icon=":material/filter_alt_off:")
    st.stop()

rate = segment["Churn"].mean()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers in segment", f"{len(segment):,}", f"{len(segment) / len(data):.0%} of base",
          delta_color="off", border=True)
k2.metric("Churn rate", ui.pct(rate), f"{100 * (rate - overall_rate):+.1f} pp vs overall", delta_color="inverse",
          border=True)
k3.metric("Avg. monthly charges", ui.money(segment["MonthlyCharges"].mean(), 2), border=True)
k4.metric("Revenue lost per month", ui.money(segment.loc[segment["Churn"] == 1, "MonthlyCharges"].sum()),
          border=True, help="Sum of the monthly charges of customers in the segment who churned.")

tab_drivers, tab_tenure, tab_matrix, tab_data = st.tabs(["Churn drivers", "Tenure and charges", "Segment matrix",
                                                         "Data"])

with tab_drivers:
    options = CATEGORICAL_INPUTS
    dimension = st.selectbox("Compare churn rate by", options, index=options.index("Contract"),
                             format_func=lambda c: FEATURE_LABELS[c])
    grouped = (segment.groupby(dimension)["Churn"].agg(["mean", "size"]).reset_index()
               .sort_values("mean", ascending=True))
    fig = go.Figure(go.Bar(
        y=grouped[dimension], x=grouped["mean"], orientation="h",
        marker_color=[COLORS["churn"] if r > overall_rate else COLORS["stay"] for r in grouped["mean"]],
        text=[f"{r:.0%} · n={n:,}" for r, n in zip(grouped["mean"], grouped["size"])], textposition="outside",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.add_vline(x=overall_rate, line_dash="dash", line_color=COLORS["muted"],
                  annotation_text=f"Overall {overall_rate:.0%}", annotation_position="top")
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.65, grouped["mean"].max() * 1.25)], title="Churn rate")
    ui.show(ui.style_fig(fig, height=120 + 46 * len(grouped), legend=False,
                         title=f"Churn rate by {FEATURE_LABELS[dimension].lower()}"))
    ui.note("Orange bars are above the overall churn rate. Small groups (low n) are less reliable.")

with tab_tenure:
    left, right = st.columns(2)
    with left:
        normalize = st.toggle("Show as share within each group", value=True)
        fig = px.histogram(segment.assign(Status=segment["Churn"].map({0: "Stays", 1: "Churns"})), x="tenure",
                           color="Status", nbins=36, barmode="overlay", opacity=0.65,
                           histnorm="probability" if normalize else None,
                           color_discrete_map={"Stays": COLORS["stay"], "Churns": COLORS["churn"]})
        fig.update_xaxes(title="Tenure (months)")
        fig.update_yaxes(title="Share" if normalize else "Customers", tickformat=".0%" if normalize else None)
        ui.show(ui.style_fig(fig, height=380, title="Churners are concentrated in the first months"))
    with right:
        sample = segment.sample(min(len(segment), 2500), random_state=7)
        fig = px.scatter(sample.assign(Status=sample["Churn"].map({0: "Stays", 1: "Churns"})), x="tenure",
                         y="MonthlyCharges", color="Status", opacity=0.55,
                         color_discrete_map={"Stays": COLORS["stay"], "Churns": COLORS["churn"]},
                         hover_data={"Contract": True, "InternetService": True, "tenure": True,
                                     "MonthlyCharges": ":.2f"})
        fig.update_traces(marker=dict(size=5))
        fig.update_xaxes(title="Tenure (months)")
        fig.update_yaxes(title="Monthly charges (USD)")
        ui.show(ui.style_fig(fig, height=380, title="Highest risk: short tenure with high charges"))

with tab_matrix:
    c1, c2 = st.columns(2)
    rows = c1.selectbox("Rows", CATEGORICAL_INPUTS, index=CATEGORICAL_INPUTS.index("Contract"),
                        format_func=lambda c: FEATURE_LABELS[c])
    cols = c2.selectbox("Columns", CATEGORICAL_INPUTS, index=CATEGORICAL_INPUTS.index("InternetService"),
                        format_func=lambda c: FEATURE_LABELS[c])
    if rows == cols:
        st.info("Choose two different features.", icon=":material/info:")
    else:
        rate_table = segment.pivot_table(index=rows, columns=cols, values="Churn", aggfunc="mean")
        size_table = segment.pivot_table(index=rows, columns=cols, values="Churn", aggfunc="size")
        text = [[f"{r:.0%}<br><span style='font-size:10px'>n={int(n):,}</span>" if n == n else ""
                 for r, n in zip(rr, nn)] for rr, nn in zip(rate_table.values, size_table.reindex_like(rate_table).values)]
        fig = go.Figure(go.Heatmap(z=rate_table.values, x=rate_table.columns, y=rate_table.index, text=text,
                                   texttemplate="%{text}", zmin=0, zmax=0.6,
                                   colorscale=[[0, "#F4F7F6"], [0.5, "#F2B49C"], [1, COLORS["churn"]]],
                                   colorbar=dict(title="Churn", tickformat=".0%"),
                                   hovertemplate="%{y} × %{x}: %{z:.1%}<extra></extra>"))
        fig.update_xaxes(title=FEATURE_LABELS[cols])
        fig.update_yaxes(title=FEATURE_LABELS[rows])
        ui.show(ui.style_fig(fig, height=420, legend=False,
                             title=f"Churn rate: {FEATURE_LABELS[rows]} × {FEATURE_LABELS[cols]}"))

with tab_data:
    st.dataframe(segment, width="stretch", height=380, hide_index=True)
    st.download_button("Download segment (CSV)", segment.to_csv(index=False).encode("utf-8"),
                       file_name="telco_segment.csv", mime="text/csv", icon=":material/download:")
