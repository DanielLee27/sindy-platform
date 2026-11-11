"""
SINDy Platform: Streamlit front end.

Run with:  streamlit run app.py

This file only sets up the pages. Each page lives in views/:
  explorer.py    fit ONE model to one dataset and inspect it closely
  comparison.py  run the comparison engine (models x datasets) as a job
  history.py     browse every run ever logged to the SQLite store
  common.py      tables and charts shared by comparison and history

The science stays outside the UI: pipeline/ (single fits) and engine/
(harness, candidates, benchmarks, store). The pages only call into them.
"""
import matplotlib

matplotlib.use("Agg")  # render figures off-screen; Streamlit displays them

import streamlit as st  # noqa: E402

st.set_page_config(page_title="SINDy Platform", layout="wide")

page = st.navigation([
    st.Page("views/explorer.py", title="Explorer", icon=":material/science:", default=True),
    st.Page("views/comparison.py", title="Model comparison", icon=":material/leaderboard:"),
    st.Page("views/history.py", title="Run history", icon=":material/history:"),
])
page.run()
