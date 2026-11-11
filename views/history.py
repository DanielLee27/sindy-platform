"""
Run history page: browse every run logged to the SQLite store.

Runs come from the CLI (python -m engine.run_benchmark) and from the
Model comparison page alike. Each batch is one invocation, stamped with the
git commit of the code that produced it, so results stay traceable even
after the code changes.
"""
import pandas as pd
import streamlit as st

from engine.store import RunStore
from views import common

st.title("Run history")

store = RunStore(common.store_path())
runs = store.load_runs()
if runs.empty:
    st.info("No runs logged yet. Start one on the **Model comparison** page or with "
            "`python -m engine.run_benchmark`.")
    st.stop()

df = common.prepare(runs)

# One row per batch, newest first: what was run, when, and on which code.
batches = (df.groupby("batch_id")
           .agg(started=("created_at", "min"), code=("git_commit", "first"),
                runs=("run_id", "count"),
                systems=("system", lambda s: ", ".join(sorted(s.unique()))),
                models=("model_name", lambda s: ", ".join(common.model_order(s.unique()))))
           .sort_values("started", ascending=False))
batches["started"] = pd.to_datetime(batches["started"]).dt.strftime("%Y-%m-%d %H:%M")


def batch_label(b):
    r = batches.loc[b]
    return f"{r['started']} · {r['runs']} runs · {b}"


# --------------------------------------------------------------- filters ---
# All filters in one row above the results.
c1, c2, c3 = st.columns([2, 1.3, 1.3])
chosen = c1.multiselect("Batches", list(batches.index), default=[batches.index[0]],
                        format_func=batch_label,
                        help="Defaults to the most recent batch. Mixing batches from different "
                             "code versions is allowed but the code column tells them apart.")
systems = c2.multiselect("Systems", sorted(df["system"].unique()))
models = c3.multiselect("Models", common.model_order(df["model_name"].unique()))

sel = df[df["batch_id"].isin(chosen or batches.index)]
if systems:
    sel = sel[sel["system"].isin(systems)]
if models:
    sel = sel[sel["model_name"].isin(models)]

st.caption(f"{len(sel)} of {len(df)} stored runs selected · store: `{common.store_path().name}`")
if sel.empty:
    st.warning("No runs match these filters.")
    st.stop()

tab_scores, tab_all, tab_eqs, tab_batches, tab_raw = st.tabs(
    ["Scores", "All metrics", "Equations", "Batches", "Raw runs"])
with tab_scores:
    metric = common.metric_picker(key="hist_metric")
    common.show_metric_chart(sel, metric)
    common.show_metric_table(sel, metric)
with tab_all:
    common.show_full_table(sel)
with tab_eqs:
    common.show_equations(sel, key="hist_eq")
with tab_batches:
    st.dataframe(batches, use_container_width=True)
with tab_raw:
    cols = ["created_at", "batch_id", "git_commit", "system", "noise", "seed", "dt",
            "n_samples", "model_name", "status", "stability", *[m for m in common.METRICS
                                                                 if m != "stable"], "error"]
    st.dataframe(sel[[c for c in cols if c in sel]], use_container_width=True, hide_index=True)
    st.download_button("Download selected runs (CSV)", runs[runs["run_id"].isin(sel["run_id"])]
                       .to_csv(index=False), file_name="runs.csv", mime="text/csv")
