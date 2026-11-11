"""
Model comparison page: run the comparison engine on a grid of datasets.

Why the run happens in a separate process: Streamlit reruns this script
whenever a widget changes, which would abort a long in-page loop (a PySR
grid takes many minutes), and PySR's Julia runtime does not tolerate
Streamlit switching threads between reruns. So the page starts the
existing CLI (python -m engine.run_benchmark) in the background with a
batch id it chose, and polls the SQLite store for that batch's runs. The
page never re-implements the engine; it is just another client of it.
"""
import subprocess
import sys
import time
import uuid

import streamlit as st

from engine.benchmarks import grid_settings
from engine.candidates import CANDIDATES
from engine.store import RunStore
from pipeline.systems import DEMO_SYSTEMS
from views import common

ROOT = common.ROOT
JOBS_DIR = ROOT / "results" / "jobs"
NOISE_CHOICES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]


def estimate_seconds(settings, models):
    """Rough duration: PySR searches 60 s per state variable, SINDy ~1 s per run."""
    total = 0.0
    for sys_name, *_ in settings:
        n_vars = len(DEMO_SYSTEMS[sys_name].var_names)
        for m in models:
            cfg = CANDIDATES[m].config
            total += cfg["timeout_per_equation"] * n_vars + 10 if "timeout_per_equation" in cfg else 1
    return total


def start_job(args, total):
    batch_id = uuid.uuid4().hex[:12]
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOBS_DIR / f"{batch_id}.log"
    cmd = [sys.executable, "-m", "engine.run_benchmark", *args,
           "--db", str(common.store_path()), "--batch-id", batch_id, "--quiet"]
    log = open(log_path, "w", encoding="utf-8")  # noqa: SIM115  (the child process owns it)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    st.session_state["cmp_job"] = dict(batch_id=batch_id, proc=proc, log=log_path, total=total,
                                       started=time.time(), finished=False, cmd=cmd)


# ----------------------------------------------------------------- sidebar ---

with st.sidebar:
    st.header("Datasets")
    systems = st.multiselect("Systems", list(DEMO_SYSTEMS), default=["Van der Pol", "Pendulum"])
    noises = st.multiselect("Noise levels", NOISE_CHOICES, default=[0.0, 0.01, 0.05],
                            format_func=lambda v: f"{v * 100:g}%",
                            help="Relative noise: std = this fraction of each variable's std.")
    n_seeds = st.number_input("Noise seeds per level", 1, 10, 1,
                              help="Repeats with different noise draws. Noise-free data is "
                                   "identical across seeds, so it runs once.")
    with st.expander("Sampling"):
        dts = st.multiselect("Sampling interval dt", [0.005, 0.01, 0.02, 0.05], default=[0.01])
        length = st.number_input("Samples per dataset", 500, 20000, 5000, 500,
                                 help="80% is used for fitting, the last 20% for the forecast.")

    st.header("Models")
    sindy_models = [m for m in CANDIDATES if CANDIDATES[m].family != "PySR"]
    models = st.multiselect("Candidates", list(CANDIDATES), default=sindy_models)
    if "PySR" not in CANDIDATES:
        st.caption("PySR is not installed (see requirements-pysr.txt).")

    settings = grid_settings(systems or None, noises, dts, [int(length)],
                             list(range(int(n_seeds))))
    ready = bool(systems and noises and dts and models)
    n_runs = len(settings) * len(models) if ready else 0
    eta = estimate_seconds(settings, models) if ready else 0
    st.caption(f"**{len(settings) if ready else 0} datasets × {len(models)} models = "
               f"{n_runs} runs**, about {eta / 60:.0f} min" if eta >= 90 else
               f"**{n_runs} runs**, about {eta:.0f} s")

    job = st.session_state.get("cmp_job")
    running = job is not None and not job["finished"]
    if st.button("Run comparison", type="primary", use_container_width=True,
                 disabled=not ready or running):
        start_job(["--systems", *systems, "--models", *models,
                   "--noise", *map(str, noises), "--dt", *map(str, dts),
                   "--length", str(int(length)),
                   "--seeds", *map(str, range(int(n_seeds)))], n_runs)
        st.rerun()


# --------------------------------------------------------------- main area ---

st.title("Model comparison")
st.caption("Every selected model is fitted to every dataset and scored on the same axes by "
           "the evaluation harness. All runs are logged to the run store "
           f"(`{common.store_path().name}`), so they also appear under **Run history**.")

job = st.session_state.get("cmp_job")
store = RunStore(common.store_path())

if job is None:
    st.info("Choose systems, noise levels and models in the sidebar, then press "
            "**Run comparison**. SINDy runs take about a second each; PySR runs take minutes.")
    st.stop()


@st.fragment(run_every="2s")
def show_progress():
    """Re-runs on its own every 2 s (only this function, not the page)."""
    job = st.session_state["cmp_job"]
    done = store.count(batch_id=job["batch_id"])
    code = job["proc"].poll()
    elapsed = time.time() - job["started"]
    st.progress(min(done / max(job["total"], 1), 1.0),
                text=f"{done} / {job['total']} runs finished · {elapsed:.0f} s elapsed")
    if st.button("Cancel", key="cancel_job"):
        job["proc"].terminate()
        code = -1
    if code is not None:  # process ended: show results with a full rerun
        job["finished"], job["returncode"] = True, code
        st.rerun()


if not job["finished"]:
    show_progress()
    st.stop()

# ----------------------------------------------------------------- results ---

runs = store.load_runs(where="batch_id = ?", params=(job["batch_id"],))
if job.get("returncode") not in (0, None):
    tail = job["log"].read_text(encoding="utf-8", errors="replace")[-3000:]
    st.error(f"The job stopped early (exit code {job['returncode']}). "
             f"{len(runs)} of {job['total']} runs were logged.")
    with st.expander("Job log"):
        st.code(tail, language=None)
if runs.empty:
    st.stop()

df = common.prepare(runs)
st.success(f"{len(df)} runs in batch `{job['batch_id']}` "
           f"(code version `{df['git_commit'].iloc[0]}`).")

tab_scores, tab_all, tab_eqs = st.tabs(["Scores", "All metrics", "Equations"])
with tab_scores:
    metric = common.metric_picker(key="cmp_metric")
    common.show_metric_chart(df, metric)
    common.show_metric_table(df, metric)
with tab_all:
    common.show_full_table(df)
with tab_eqs:
    common.show_equations(df, key="cmp_eq")

failed = df[df["status"] != "ok"]
if len(failed):
    with st.expander(f"{len(failed)} failed run(s)"):
        st.dataframe(failed[["system", "noise", "seed", "model_name", "error"]],
                     use_container_width=True, hide_index=True)

st.download_button("Download these runs (CSV)", runs.to_csv(index=False),
                   file_name=f"comparison_{job['batch_id']}.csv", mime="text/csv")
