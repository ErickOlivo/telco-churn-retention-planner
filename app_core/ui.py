"""Visual system: CSS, page headers, formatting and Plotly styling."""
import streamlit as st

from app_core.config import COLORS

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric, button, input, textarea, select {{
    font-family: 'Inter', sans-serif;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ color: {COLORS['ink']}; letter-spacing: -0.01em; }}
div[data-testid="stMetric"] {{
    background: #FFFFFF; border: 1px solid {COLORS['grid']} !important; border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 1px 2px rgba(11, 31, 51, 0.04);
}}
div[data-testid="stMetricLabel"] p {{ color: {COLORS['muted']}; font-size: 0.82rem; font-weight: 500; }}
div[data-testid="stMetricValue"] {{ font-weight: 650; color: {COLORS['ink']}; font-size: 1.75rem; }}
.rp-hero {{
    background: linear-gradient(120deg, #F1F8F7 0%, #F7F9FB 60%, #FFFFFF 100%);
    border: 1px solid {COLORS['grid']}; border-radius: 16px; padding: 22px 26px; margin-bottom: 1.2rem;
}}
.rp-eyebrow {{ color: {COLORS['primary']}; font-weight: 600; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }}
.rp-hero .rp-title {{ font-size: 1.85rem !important; font-weight: 700; color: {COLORS['ink']}; margin: 0; line-height: 1.2; }}
.rp-subtitle {{ color: {COLORS['muted']}; font-size: 1.0rem; margin-top: 6px; max-width: 860px; }}
.rp-badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem;
    font-weight: 600; margin-right: 6px; }}
.rp-card {{ border: 1px solid {COLORS['grid']}; border-radius: 12px; padding: 16px 18px; background: #FFFFFF;
    height: 100%; }}
.rp-card h5 {{ margin: 0 0 6px 0; font-size: 1.0rem; color: {COLORS['ink']}; }}
.rp-card p {{ color: {COLORS['muted']}; font-size: 0.9rem; margin: 0; }}
.rp-step {{ color: {COLORS['primary']}; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.06em; }}
.rp-risk {{ border-radius: 14px; padding: 18px 20px; border: 1px solid {COLORS['grid']}; }}
.rp-risk .value {{ font-size: 2.6rem; font-weight: 700; line-height: 1.1; }}
.rp-risk .label {{ color: {COLORS['muted']}; font-size: 0.85rem; }}
.rp-note {{ color: {COLORS['muted']}; font-size: 0.82rem; }}
section[data-testid="stSidebar"] .rp-side {{ font-size: 0.8rem; color: {COLORS['muted']}; line-height: 1.5; }}
</style>
"""

TONES = {
    "high": (COLORS["churn_soft"], "#9A2F12"),
    "medium": ("#FBEFD3", "#8A6100"),
    "low": (COLORS["primary_soft"], "#0B5A54"),
    "neutral": ("#EEF2F6", COLORS["ink"]),
}


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title, subtitle, eyebrow=None):
    eyebrow_html = f'<div class="rp-eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(f'<div class="rp-hero">{eyebrow_html}<div class="rp-title" style="font-size:1.9rem;font-weight:700;line-height:1.2;color:{COLORS['ink']}">{title}</div>'
                f'<div class="rp-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)


def badge(text, tone="neutral"):
    bg, fg = TONES[tone]
    return f'<span class="rp-badge" style="background:{bg};color:{fg}">{text}</span>'


def card(title, body, step=None):
    step_html = f'<div class="rp-step">{step}</div>' if step else ""
    st.markdown(f'<div class="rp-card">{step_html}<h5>{title}</h5><p>{body}</p></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<p class="rp-note">{text}</p>', unsafe_allow_html=True)


def hypothetical_notice():
    st.info("Monetary figures use **hypothetical** campaign assumptions (editable). They illustrate the "
            "decision logic and are not results estimated from the dataset.", icon=":material/info:")


def md_money(x, decimals=0):
    """Money string safe for st.markdown (a bare $ starts LaTeX)."""
    return money(x, decimals).replace("$", "\\$")


def money(x, decimals=0):
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}${x / 1_000_000:,.2f} M"
    if x >= 10_000:
        return f"{sign}${x / 1_000:,.1f} k"
    return f"{sign}${x:,.{decimals}f}"


def pct(x, decimals=1):
    return f"{100 * x:.{decimals}f}%"


def style_fig(fig, height=360, title=None, legend=True):
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=COLORS["ink"])) if title else None,
        margin=dict(l=8, r=8, t=56 if title else 16, b=8),
        font=dict(family="Inter, sans-serif", color=COLORS["ink"], size=12.5),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title=None) if legend else None,
        showlegend=legend,
        hoverlabel=dict(bgcolor="white", font_size=12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False,
                                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
