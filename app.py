"""Retention Planner: churn risk intelligence for telecom retention teams."""
import streamlit as st

from app_core import ui
from app_core.data import load_metadata

st.set_page_config(page_title="Retention Planner | Telco Churn", page_icon=":material/radar:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

pages = {
    "Start": [
        st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    ],
    "Understand": [
        st.Page("views/explorer.py", url_path="explorer", title="Customer Explorer", icon=":material/query_stats:"),
        st.Page("views/performance.py", url_path="performance", title="Model Performance", icon=":material/monitoring:"),
    ],
    "Act": [
        st.Page("views/risk.py", url_path="customer-risk", title="Customer Risk", icon=":material/person_search:"),
        st.Page("views/planner.py", url_path="planner", title="Retention Planner", icon=":material/target:"),
    ],
    "Reference": [
        st.Page("views/methodology.py", url_path="methodology", title="Methodology", icon=":material/menu_book:"),
    ],
}
navigation = st.navigation(pages)

meta = load_metadata()
with st.sidebar:
    st.markdown("**Retention Planner**")
    st.markdown(
        f'<div class="rp-side">Model: LightGBM, {meta["final_params"]["n_estimators"]} trees of '
        f'{meta["final_params"]["num_leaves"]} leaves<br>Decision threshold: {meta["business_threshold"]:.2f}'
        f'<br>Trained: {meta["created_utc"][:10]}<br>Inference only, no retraining in the app.</div>',
        unsafe_allow_html=True)

navigation.run()
