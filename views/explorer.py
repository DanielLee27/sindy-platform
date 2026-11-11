"""
Explorer page: fit ONE model to one dataset and inspect it in detail.

This file is UI only. The science lives in pipeline/:
  systems.py   demo systems        -> DEMO_SYSTEMS
  registry.py  models and plots    -> MODELS, PLOTS
  runner.py    the pipeline itself -> run_pipeline()
"""
import io

import matplotlib

matplotlib.use("Agg")  # render figures off-screen; Streamlit displays them

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline.loader import load_csv
from pipeline.registry import MODELS, PLOTS
from pipeline.runner import run_pipeline
from pipeline.systems import DEMO_SYSTEMS


# ------------------------------------------------------------ data loading ---
# @st.cache_data remembers results per unique set of arguments, so Streamlit's
# rerun-on-every-interaction doesn't re-solve the ODE or re-parse the CSV.

@st.cache_data
def load_demo(system_name, params, t_end, dt, noise_level, seed):
    s = DEMO_SYSTEMS[system_name]
    return s.generate(t_end, dt, noise_level, seed, **params)


@st.cache_data
def read_csv_columns(file_bytes):
    return list(pd.read_csv(io.BytesIO(file_bytes), nrows=0).columns)


@st.cache_data
def load_uploaded_csv(file_bytes, time_column):
    # load_csv passes its argument to pd.read_csv, which accepts file objects
    return load_csv(io.BytesIO(file_bytes), time_column=time_column)


def number_widget(label, param, key, default=None):
    """Render a sidebar widget from a registry.Param description."""
    value = param.default if default is None else default
    if param.kind == "int":
        return st.number_input(label, int(param.min), int(param.max), int(value),
                               step=int(param.step), key=key, help=param.help)
    return st.number_input(label, float(param.min), float(param.max), float(value),
                           step=float(param.step), key=key, help=param.help,
                           format="%.3f")


# ----------------------------------------------------------------- sidebar ---
# Every widget below returns its current value. Streamlit reruns this whole
# script whenever one changes, so these variables always hold the latest input.

with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["Demo system", "Upload CSV"], horizontal=True)

    t = X = var_names = None
    true_rhs = None
    suggested = {}
    data_key = None  # identifies the dataset, used to key widget defaults
    data_sig = None  # fingerprint of every data setting, to detect stale results

    if source == "Demo system":
        system_name = st.selectbox("System", list(DEMO_SYSTEMS))
        system = DEMO_SYSTEMS[system_name]
        st.caption(system.description)

        with st.expander("System parameters"):
            sys_params = {
                p: st.number_input(p, value=float(v), key=f"{system_name}:{p}", format="%.3f")
                for p, v in system.params.items()
            }
        noise = st.slider("Noise level (std)", 0.0, 1.0, 0.0, 0.01,
                          help="Gaussian noise added to the simulated data.")
        t_end = st.number_input("Duration", 1.0, 100.0, float(system.t_end), 1.0,
                                key=f"{system_name}:t_end")
        seed = st.number_input("Noise seed", 0, 10_000, 0, 1)

        t, X = load_demo(system_name, sys_params, t_end, system.dt, noise, int(seed))
        var_names = system.var_names
        true_rhs = system.true_equations(**sys_params)
        suggested = system.suggested
        data_key = system_name
        data_sig = (system_name, tuple(sys_params.items()), noise, t_end, int(seed))

    else:
        upload = st.file_uploader("CSV file", type="csv",
                                  help="One column per variable, optional time column.")
        if upload is not None:
            file_bytes = upload.getvalue()
            columns = read_csv_columns(file_bytes)
            time_col = st.selectbox("Time column", ["(none)"] + columns)
            time_col = None if time_col == "(none)" else time_col
            try:
                t, X = load_uploaded_csv(file_bytes, time_col)
                X = np.asarray(X, dtype=float)
            except ValueError as exc:
                st.error(f"Could not load CSV: {exc}")
                X = None
            if X is not None:
                var_names = [c for c in columns if c != time_col]
                if t is None:
                    dt = st.number_input("Sampling interval dt", 1e-6, 1e6, 0.01,
                                         format="%.4f",
                                         help="No time column, so time is rebuilt as 0, dt, 2dt, ...")
                    t = np.arange(len(X)) * dt
                data_key = upload.name
                data_sig = (upload.name, len(file_bytes), time_col,
                            dt if time_col is None else None)

    st.header("2. Preprocessing")
    scale = st.checkbox(
        "Z-score scale variables", value=False,
        help="Off by default. Scaling each variable independently rescales the "
             "nonlinear library terms inconsistently, which distorts STLSQ's "
             "magnitude-based pruning. Recovered equations would also be in "
             "scaled units.",
    )
    deriv_method = st.selectbox(
        "Derivative method", ["finite_difference", "smoothed_fd"],
        help="finite_difference: best for clean data. "
             "smoothed_fd: Savitzky-Golay smoothing first; use it for noisy data.",
    )

    st.header("3. Model")
    # Default to the model the demo system suggests (e.g. poly + trig for the
    # pendulum); keyed by dataset so switching systems resets it.
    default_model = system.suggested_model if source == "Demo system" else None
    model_name = st.selectbox(
        "Model", list(MODELS), key=f"model:{data_key}",
        index=list(MODELS).index(default_model) if default_model in MODELS else 0)
    spec = MODELS[model_name]
    st.caption(spec.description)
    # Keying widgets by dataset means switching systems resets them to that
    # system's suggested values (e.g. degree 3 for Van der Pol).
    model_params = {
        name: number_widget(name, p, key=f"{model_name}:{data_key}:{name}",
                            default=suggested.get(name))
        for name, p in spec.params.items()
    }

    run_clicked = st.button("Run SINDy", type="primary", use_container_width=True,
                            disabled=X is None)


# ------------------------------------------------------------- run & store ---
# st.session_state survives reruns. Without it, results would vanish the next
# time any widget changed. We refit only when the button is pressed.

config = dict(data=data_sig, model=model_name, params=model_params,
              scale=scale, deriv=deriv_method)

if run_clicked and X is not None:
    with st.spinner("Fitting..."):
        try:
            result = run_pipeline(t, X, var_names, model_name, model_params,
                                  scale=scale, deriv_method=deriv_method)
            st.session_state["result"] = result
            st.session_state["true_rhs"] = true_rhs
            st.session_state["config"] = config
        except Exception as exc:
            st.session_state.pop("result", None)
            st.error(f"Pipeline failed: {exc}")


# -------------------------------------------------------------- main area ---

st.title("SINDy Explorer")
st.caption("Sparse Identification of Nonlinear Dynamics: recover governing "
           "equations from time-series data.")

result = st.session_state.get("result")

if result is None:
    if X is None:
        st.info("Pick a demo system or upload a CSV in the sidebar.")
    else:
        st.info(f"Data ready: {X.shape[0]} timesteps x {X.shape[1]} variables. "
                "Press **Run SINDy**.")
    st.stop()

if st.session_state.get("config") != config:
    st.warning("Settings changed since the last run. Press **Run SINDy** to update.")

for note in result["notes"]:
    st.warning(note)

# Equations
st.subheader("Recovered equations")
true_rhs = st.session_state.get("true_rhs")
names = result["var_names"]

cols = st.columns(2 if true_rhs else 1)
with cols[0]:
    st.markdown("**SINDy**")
    st.code("\n".join(f"d{v}/dt = {eq}" for v, eq in zip(names, result["equations"])),
            language=None)
if true_rhs:
    with cols[1]:
        st.markdown("**True system**")
        st.code("\n".join(f"d{v}/dt = {eq}" for v, eq in zip(names, true_rhs)),
                language=None)

coef = np.asarray(result["coefficients"])
m1, m2, m3 = st.columns(3)
m1.metric("R² (derivative fit)", f"{result['score']:.4f}",
          help="How well the recovered equations predict the estimated derivatives.")
m2.metric("Active terms", f"{int(np.count_nonzero(coef))} / {coef.size}",
          help="Nonzero coefficients out of all library terms. Fewer means sparser.")
m3.metric("Library size", len(result["feature_library_names"]))

with st.expander("Coefficient matrix"):
    st.dataframe(pd.DataFrame(coef, index=[f"d{v}/dt" for v in names],
                              columns=result["feature_library_names"]).round(4))

# Plots: one tab per entry in registry.PLOTS
st.subheader("Plots")
tabs = st.tabs(list(PLOTS))
for tab, (plot_name, plot) in zip(tabs, PLOTS.items()):
    with tab:
        st.caption(plot.description)
        missing = [k for k in plot.requires if result.get(k) is None]
        if missing:
            st.info(f"Not available for this run (missing: {', '.join(missing)}).")
            continue
        fig = plot.fn(result)
        st.pyplot(fig)
        plt.close(fig)  # free memory; reruns would otherwise accumulate figures
