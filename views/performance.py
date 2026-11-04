import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

from app_core import ui
from app_core.config import COLORS, MODEL_COLORS, SEARCH_SPACE, STRATEGY_COLORS
from app_core.data import business_threshold, load_metadata, load_result, scored_portfolio

meta = load_metadata()
t_star = business_threshold()

ui.page_header("Model Performance",
               "How the model was chosen, tuned and validated. Every comparison uses the same pipeline and the "
               "same 5 cross-validation folds; the test set was used once, at the end.",
               eyebrow="Evidence")

tab_models, tab_hpo, tab_pareto, tab_test, tab_xai = st.tabs(
    ["Model comparison", "Hyperparameter optimization", "Multi-objective (Pareto)", "Test evaluation",
     "Global explainability"])

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
          "pr_auc": "PR-AUC"}

with tab_models:
    models = load_result("model_comparison").set_index("model")
    fig = go.Figure()
    for name, row in models.iterrows():
        fig.add_trace(go.Bar(name=name, x=[LABELS[m] for m in METRICS], y=row[METRICS],
                             error_y=dict(type="data", array=row[[f"{m}_std" for m in METRICS]], thickness=1),
                             marker_color=MODEL_COLORS.get(name, COLORS["muted"]),
                             hovertemplate=f"{name}<br>%{{x}}: %{{y:.3f}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(range=[0, 1], title="Mean over 5 folds (± std)")
    ui.show(ui.style_fig(fig, height=400, title="Cross-validated metrics at threshold 0.5"))

    table = models[METRICS + ["fit_time_s"]].rename(columns={**LABELS, "fit_time_s": "Fit time (s)"})
    st.dataframe(table.style.format("{:.3f}").highlight_max(subset=list(LABELS.values()), color="#D7EEEB"),
                 width="stretch")
    st.markdown(
        "**Reading.** Logistic Regression is the strongest untuned model (PR-AUC 0.661). LightGBM was chosen "
        "for optimization: it has the richest hyperparameter space, captures the contract × internet interaction "
        "and supports exact SHAP. After tuning it reaches 0.665-0.670; the margin is within one standard error, "
        "so a linear model remains a credible fallback. PR-AUC is the selection metric because it focuses on the "
        "churn class and does not depend on a threshold.")

with tab_hpo:
    trials = load_result("hpo_trials")
    summary = load_result("optimizer_summary").set_index("strategy")
    strategies = load_result("strategy_comparison").set_index("strategy")

    c1, c2, c3, c4 = st.columns(4)
    for col, strategy in zip([c1, c2, c3, c4], summary.index):
        row = summary.loc[strategy]
        spread = f"± {row['best_pr_auc_std']:.4f} · " if pd.notna(row["best_pr_auc_std"]) else ""
        col.metric(f"{strategy}", f"{row['best_pr_auc_mean']:.4f}",
                   f"{spread}{row['time_s_mean']:.0f} s", delta_color="off",
                   border=True, help="Best CV PR-AUC (mean ± std over seeds) · time per run · evaluations")

    left, right = st.columns(2)
    with left:
        curves = trials.groupby(["strategy", "evaluation"])["best_so_far"].agg(["mean", "min", "max"]).reset_index()
        fig = go.Figure()
        for strategy, g in curves.groupby("strategy"):
            color = STRATEGY_COLORS[strategy]
            fig.add_trace(go.Scatter(x=np.r_[g["evaluation"], g["evaluation"][::-1]], y=np.r_[g["max"], g["min"][::-1]],
                                     fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=g["evaluation"], y=g["mean"], name=strategy, line=dict(color=color, width=2.5),
                                     hovertemplate=f"{strategy}: %{{y:.4f}} after %{{x}} evaluations<extra></extra>"))
        fig.update_yaxes(range=[trials["value"].quantile(0.05), trials["value"].max() + 0.003],
                         title="Best CV PR-AUC so far")
        fig.update_xaxes(title="Evaluations")
        ui.show(ui.style_fig(fig, height=380, title="Anytime performance"))
    with right:
        fig = px.box(trials, x="strategy", y="value", color="strategy", points="all",
                     color_discrete_map=STRATEGY_COLORS)
        fig.update_traces(marker=dict(size=3, opacity=0.5), jitter=0.4)
        fig.update_yaxes(title="CV PR-AUC of each configuration")
        fig.update_xaxes(title=None)
        ui.show(ui.style_fig(fig, height=380, legend=False, title="Where each strategy spends its budget"))

    left, right = st.columns([1, 1.3])
    with left:
        fig = go.Figure(go.Bar(x=summary.index, y=summary["time_s_mean"],
                               error_y=dict(type="data", array=summary["time_s_std"].fillna(0)),
                               marker_color=[STRATEGY_COLORS[s] for s in summary.index],
                               hovertemplate="%{x}: %{y:.0f} s<extra></extra>"))
        fig.update_yaxes(title="Seconds per run")
        ui.show(ui.style_fig(fig, height=330, legend=False, title="Optimization time (48 evaluations)"))
    with right:
        st.markdown("**Search space**")
        st.dataframe(pd.DataFrame(SEARCH_SPACE, columns=["Hyperparameter", "Type", "Range (Random, TPE, NSGA-II)",
                                                         "Scale", "Grid values"]), hide_index=True, width="stretch")
    st.markdown("**Best configuration of each strategy, all metrics (5-fold CV)**")
    st.dataframe(strategies.drop(columns=["pr_auc_std"]).rename(columns={**LABELS, "time_s_mean": "Time (s)",
                                                                          "configs_evaluated": "Configs"})
                 .style.format(precision=3), width="stretch")
    st.markdown(
        "**Reading.** All strategies reach PR-AUC ≈ 0.667-0.670: differences are far below the fold-to-fold "
        "standard error, so they are statistically tied. TPE is selected: same quality, lowest variance across "
        "seeds and markedly faster than Random Search, because it learns to avoid expensive, unpromising "
        "configurations. Learning rate is the decisive hyperparameter.")

with tab_pareto:
    mo = load_result("pareto_trials")
    latency = load_result("pareto_latency")
    pareto = mo[mo["pareto"]].sort_values("total_leaves")
    info = meta["pareto"]
    selected = mo[mo["trial"] == info["selected_trial"]].iloc[0]

    hover = ("Trial %{customdata[0]}<br>PR-AUC %{y:.4f}<br>Leaves %{x:,.0f}<br>"
             "Trees %{customdata[1]}, leaves/tree %{customdata[2]}<extra></extra>")
    fig = go.Figure()
    dominated = mo[~mo["pareto"]]
    fig.add_trace(go.Scatter(x=dominated["total_leaves"], y=dominated["pr_auc"], mode="markers", name="Dominated",
                             marker=dict(color="#C9D2DC", size=8),
                             customdata=dominated[["trial", "n_estimators", "num_leaves"]], hovertemplate=hover))
    fig.add_trace(go.Scatter(x=pareto["total_leaves"], y=pareto["pr_auc"], mode="lines+markers", name="Pareto front",
                             line=dict(color=STRATEGY_COLORS["NSGA-II"], shape="hv", width=2),
                             marker=dict(size=10), customdata=pareto[["trial", "n_estimators", "num_leaves"]],
                             hovertemplate=hover))
    fig.add_trace(go.Scatter(x=[selected["total_leaves"]], y=[selected["pr_auc"]], mode="markers", name="Selected",
                             marker=dict(symbol="star", size=22, color=COLORS["churn"],
                                         line=dict(color="black", width=1)), hoverinfo="skip"))
    fig.add_hline(y=info["one_se_threshold"], line_dash="dash", line_color=COLORS["ink"],
                  annotation_text="1-SE threshold", annotation_position="bottom right")
    fig.add_hline(y=info["reference_pr_auc"], line_dash="dot", line_color=COLORS["muted"],
                  annotation_text="Best PR-AUC, all optimizers", annotation_position="top right")
    fig.update_xaxes(type="log", tickvals=[500, 1000, 2000, 5000, 10000, 20000, 50000],
                     ticktext=["500", "1k", "2k", "5k", "10k", "20k", "50k"],
                     title="Model complexity: total leaves (log scale, lower is better)")
    fig.update_yaxes(title="CV PR-AUC (higher is better)", range=[max(0.58, mo["pr_auc"].min() - 0.005),
                                                                   info["reference_pr_auc"] + 0.006])
    ui.show(ui.style_fig(fig, height=460, title="NSGA-II: ranking quality vs model complexity"))

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(r"""
**Formulation**

$\max_\theta f_1(\theta) = \overline{\text{PR-AUC}}_{CV}(\theta)$ and
$\min_\theta f_2(\theta) = \overline{\sum_t L_t(\theta)}$ (total leaves).

**Selection.** Any solution within one standard error of the best PR-AUC is statistically equivalent;
the simplest one on the front is chosen. It keeps the ranking quality with a fraction of the trees,
which lowers inference, maintenance and audit cost.
""")
    with right:
        st.dataframe(latency.rename(columns={"trial": "Trial", "total_leaves": "Leaves", "pr_auc": "PR-AUC",
                                             "ms_per_1000_customers": "ms / 1,000 customers"})
                     .style.format({"Leaves": "{:,.0f}", "PR-AUC": "{:.4f}", "ms / 1,000 customers": "{:.1f}"}),
                     hide_index=True, width="stretch")
        ui.note("Latency measured after the search, for Pareto solutions only. Complexity is the deterministic "
                "proxy used during optimization.")

with tab_test:
    portfolio = scored_portfolio()
    y, p = portfolio["actual_churn"].to_numpy(), portfolio["churn_probability"].to_numpy()
    threshold = st.slider("Decision threshold", 0.05, 0.95, value=round(t_star, 2), step=0.01,
                          help=f"Business threshold chosen on cross-validation: {t_star:.2f}")
    pred = (p >= threshold).astype(int)
    m = st.columns(5)
    m[0].metric("Accuracy", f"{accuracy_score(y, pred):.3f}", border=True)
    m[1].metric("Precision", f"{precision_score(y, pred, zero_division=0):.3f}", border=True)
    m[2].metric("Recall", f"{recall_score(y, pred):.3f}", border=True)
    m[3].metric("F1", f"{f1_score(y, pred):.3f}", border=True)
    m[4].metric("Customers flagged", f"{pred.sum():,}", f"of {len(pred):,}", delta_color="off", border=True)

    c1, c2, c3 = st.columns([1.15, 1, 1])
    with c1:
        cm = confusion_matrix(y, pred)
        labels = [["True negative", "False positive"], ["False negative", "True positive"]]
        text = [[f"<b>{cm[i, j]:,}</b><br>{labels[i][j]}" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=cm, x=["Stays", "Churns"], y=["Stays", "Churns"], text=text,
                                   texttemplate="%{text}", textfont=dict(size=13),
                                   colorscale=[[0, "#F4F7F6"], [1, COLORS["primary"]]], showscale=False,
                                   hoverinfo="skip", xgap=3, ygap=3))
        fig.update_yaxes(autorange="reversed", title="Actual")
        fig.update_xaxes(title="Predicted", side="bottom")
        ui.show(ui.style_fig(fig, height=330, legend=False, title=f"Confusion matrix (t = {threshold:.2f})"))
    with c2:
        fpr, tpr, roc_t = roc_curve(y, p)
        point = np.argmin(np.abs(roc_t - threshold))
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=COLORS["muted"], dash="dash"), showlegend=False),
                         go.Scatter(x=fpr, y=tpr, line=dict(color=COLORS["primary"], width=2.5),
                                    name=f"AUC {roc_auc_score(y, p):.3f}"),
                         go.Scatter(x=[fpr[point]], y=[tpr[point]], mode="markers", name="Current threshold",
                                    marker=dict(color=COLORS["churn"], size=11))])
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        ui.show(ui.style_fig(fig, height=330, title="ROC curve (test)"))
    with c3:
        prec, rec, pr_t = precision_recall_curve(y, p)
        point = np.argmin(np.abs(pr_t - threshold))
        fig = go.Figure([go.Scatter(x=rec, y=prec, line=dict(color=COLORS["primary"], width=2.5),
                                    name=f"AP {average_precision_score(y, p):.3f}"),
                         go.Scatter(x=[rec[point]], y=[prec[point]], mode="markers", name="Current threshold",
                                    marker=dict(color=COLORS["churn"], size=11))])
        fig.add_hline(y=y.mean(), line_dash="dash", line_color=COLORS["muted"], annotation_text="Base rate")
        fig.update_xaxes(title="Recall")
        fig.update_yaxes(title="Precision", range=[0, 1.02])
        ui.show(ui.style_fig(fig, height=330, title="Precision-recall curve (test)"))
    ui.note("1,409 held-out customers. Moving the threshold trades false alarms for missed churners; "
            "the business threshold maximizes expected campaign profit on cross-validation, not on test.")

with tab_xai:
    importance = load_result("global_importance").set_index("Unnamed: 0")
    importance.index.name = "feature"
    norm = importance.clip(lower=0) / importance.clip(lower=0).sum()
    top = norm.mean(axis=1).sort_values(ascending=False).head(10).index
    method_colors = [COLORS["primary"], "#5FA8A0", COLORS["stay"], COLORS["accent"]]
    fig = go.Figure([go.Bar(name=col, x=top, y=norm.loc[top, col], marker_color=color,
                            hovertemplate=f"{col}<br>%{{x}}: %{{y:.1%}}<extra></extra>")
                     for col, color in zip(norm.columns, method_colors)])
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", title="Share of total importance")
    ui.show(ui.style_fig(fig, height=420, title="Top 10 features by four importance methods"))
    ranks = importance.rank(ascending=False).astype(int).loc[top]
    st.dataframe(ranks, width="stretch")
    st.markdown(
        "**Reading.** SHAP, LIME and permutation importance agree: contract, tenure and internet service lead. "
        "Random Forest impurity importance is the outlier: it ranks total charges first and even places "
        "gender in its top 10, because impurity favors continuous features with many split points and is "
        "measured on training data. Importance describes how the model uses features, not what causes churn.")
