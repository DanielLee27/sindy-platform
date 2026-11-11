"""
Tables and charts shared by the comparison and history pages.

Both pages show runs loaded from the SQLite store (RunStore.load_runs), so
everything here takes that DataFrame format. The aggregation rules (median
over seeds, "k/n stable") come from engine.harness.comparison_table, so the
CLI and the app always agree.
"""
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from engine.candidates import CANDIDATES
from engine.harness import SETTING_COLUMNS, comparison_table, with_settings
from engine.store import DEFAULT_PATH
from pipeline.systems import DEMO_SYSTEMS

ROOT = Path(__file__).resolve().parent.parent


def store_path():
    """The run store the app reads and writes. SINDY_RUNS_DB overrides it
    (the tests use this so they never touch your real results)."""
    return Path(os.environ.get("SINDY_RUNS_DB") or ROOT / DEFAULT_PATH)


# Score axes: label, which direction is better, log scale?, explanation.
METRICS = {
    "reconstruction_error": ("Reconstruction error", "lower", True,
                             "Relative error of the learned vector field dx/dt against the true "
                             "one, at clean training states. Immune to chaotic divergence."),
    "term_f1": ("Term F1 (structure recovery)", "higher", False,
                "Overlap of recovered and true terms: 1 = exactly the right terms. "
                "Blank for models without named terms."),
    "forecast_nrmse": ("Forecast NRMSE", "lower", True,
                       "Normalised RMS error of a simulation over the held-out window. "
                       "Blank if the simulation diverged."),
    "valid_time": ("Valid prediction time (s)", "higher", False,
                   "How long the forecast stays within 0.5 normalised error of the truth."),
    "n_terms": ("Number of terms", "lower", False, "Total nonzero terms over all equations."),
    "stable": ("Stable runs (fraction)", "higher", False,
               "Share of runs whose model stayed bounded over a long simulation."),
    "coef_error": ("Coefficient error", "lower", True,
                   "Relative error of the coefficients, over the union of true and recovered terms."),
    "fit_seconds": ("Fit time (s)", "lower", True, "Wall-clock fitting time."),
}

# Fixed colour per model, assigned in registry order so a model keeps its
# colour no matter which models a filter shows. Validated categorical palette
# (light and dark steps of the same hues); shapes are a second encoding so
# identity never depends on colour alone.
_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
_SHAPES = ["circle", "square", "triangle-up", "diamond", "cross", "triangle-down"]


def model_order(names):
    """Registry order first, then any other (e.g. removed) models alphabetically."""
    known = [m for m in CANDIDATES if m in set(names)]
    return known + sorted(set(names) - set(known))


def _palette():
    try:
        dark = st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001
        dark = False
    return _DARK if dark else _LIGHT


def prepare(runs):
    """Stored runs -> flat settings columns plus a numeric 'stable' column."""
    df = with_settings(runs)
    df["stable"] = (df["stability"] == "stable").astype(float)
    return df


# ------------------------------------------------------------------ tables ---

def metric_table(df, metric):
    """Rows = settings, columns = models, values = median over seeds."""
    agg = "mean" if metric == "stable" else "median"
    tab = df.pivot_table(index=SETTING_COLUMNS, columns="model_name", values=metric,
                         aggfunc=agg, dropna=False)
    tab = tab[[m for m in model_order(tab.columns)]]
    runs = df.groupby(SETTING_COLUMNS)["seed"].nunique().rename("seeds")
    return tab.join(runs)


def show_metric_table(df, metric):
    label, direction, _, _ = METRICS[metric]
    tab = metric_table(df, metric)
    models = [c for c in tab.columns if c != "seeds"]

    def best(row):
        vals = row[models]
        if vals.notna().sum() < 2:
            return [""] * len(row)
        target = vals.min() if direction == "lower" else vals.max()
        return ["background-color: rgba(42, 120, 214, 0.18)"
                if c in models and row[c] == target else "" for c in row.index]

    styled = (tab.style.apply(best, axis=1)
              .format("{:.3g}", subset=models, na_rep="–")
              .format("{:d}", subset=["seeds"]))
    st.dataframe(styled, use_container_width=True)
    st.caption(f"{label}: median over seeds. Highlight = best in the row "
               f"({direction} is better). '–' = no value (e.g. diverged or no named terms).")


def show_full_table(df):
    """Every metric side by side, exactly as the CLI prints it."""
    tab = comparison_table(df)
    tab.columns = [f"{m} · {model}" for m, model in tab.columns]
    st.dataframe(tab, use_container_width=True)


# ------------------------------------------------------------------ charts ---

def metric_chart(df, metric):
    """
    Median score vs noise level, one small multiple per system, one line per
    model. Faint dots are the individual seeds, so spread is visible.
    """
    label, _, log, _ = METRICS[metric]
    d = df[["system", "model_name", "noise", "seed", metric]].dropna(subset=[metric]).copy()
    if log:
        d = d[d[metric] > 0]  # log scale cannot show 0
    if d.empty:
        return None
    d["noise_pct"] = (d["noise"] * 100).map(lambda v: f"{v:g}%")
    noise_order = [f"{v * 100:g}%" for v in sorted(d["noise"].unique())]
    agg = "mean" if metric == "stable" else "median"
    med = (d.groupby(["system", "model_name", "noise", "noise_pct"])[metric]
           .agg([agg, "count"]).reset_index().rename(columns={agg: "value", "count": "runs"}))

    models = model_order(d["model_name"].unique())
    pal, all_models = _palette(), list(CANDIDATES)
    colors = [pal[all_models.index(m) % len(pal)] if m in all_models else "#898781"
              for m in models]
    shapes = [_SHAPES[all_models.index(m) % len(_SHAPES)] if m in all_models else "circle"
              for m in models]

    x = alt.X("noise_pct:O", sort=noise_order, title="Noise (fraction of each variable's std)",
              axis=alt.Axis(labelAngle=0))
    y_scale = alt.Scale(type="log") if log else alt.Scale(zero=False)
    color = alt.Color("model_name:N", scale=alt.Scale(domain=models, range=colors),
                      legend=alt.Legend(title="Model", orient="top"))
    shape = alt.Shape("model_name:N", scale=alt.Scale(domain=models, range=shapes), legend=None)

    # A faceted layer chart needs ONE data source, so seeds and medians go in
    # one long table, told apart by `kind`; each layer filters its own rows.
    # (seed is a string so median rows can carry "all" rather than NaN,
    # which is not valid JSON)
    seed_rows = d.rename(columns={metric: "value"}).assign(
        kind="seed", runs=1, seed=d["seed"].map(lambda v: str(int(v))))
    data = pd.concat([seed_rows, med.assign(kind="median", seed="all")], ignore_index=True)
    # Few ticks (decades on a log axis) keep the grid recessive.
    y_axis = alt.Axis(format="~g", tickCount=5)
    base = alt.Chart(data).encode(x=x, y=alt.Y("value:Q", scale=y_scale, title=label, axis=y_axis),
                                  color=color)
    is_median = alt.datum.kind == "median"

    lines = base.transform_filter(is_median).mark_line(strokeWidth=2)
    points = base.transform_filter(is_median).mark_point(size=80, filled=True, opacity=1).encode(
        shape=shape,
        tooltip=[alt.Tooltip("system:N", title="System"),
                 alt.Tooltip("model_name:N", title="Model"),
                 alt.Tooltip("noise_pct:N", title="Noise"),
                 alt.Tooltip("value:Q", title=f"{agg.capitalize()} {label}", format=".3g"),
                 alt.Tooltip("runs:Q", title="Runs")])
    seeds = base.transform_filter(alt.datum.kind == "seed").mark_point(
        size=30, opacity=0.35, filled=True).encode(
        shape=shape,
        tooltip=[alt.Tooltip("model_name:N", title="Model"),
                 alt.Tooltip("seed:N", title="Seed"),
                 alt.Tooltip("value:Q", title=label, format=".3g")])

    n_sys = d["system"].nunique()
    return (alt.layer(seeds, lines, points, data=data)
            .properties(width=260, height=200)
            .facet(facet=alt.Facet("system:N", title=None), columns=min(n_sys, 3))
            .resolve_scale(y="independent"))


def show_metric_chart(df, metric):
    chart = metric_chart(df, metric)
    if chart is None:
        st.info("No values to plot for this metric.")
        return
    st.altair_chart(chart, theme="streamlit")
    st.caption("Lines = median over seeds; faint dots = individual seeds. Each system has its "
               "own y-axis, because error scales differ by orders of magnitude between systems.")


def metric_picker(key):
    return st.selectbox("Score", list(METRICS), key=key, format_func=lambda m: METRICS[m][0],
                        help="\n\n".join(f"**{v[0]}**: {v[3]}" for v in METRICS.values()))


# --------------------------------------------------------------- equations ---

def _true_equations(row):
    system = DEMO_SYSTEMS.get(row["system"])
    if system is None:
        return None
    params = {}
    for p, default in system.params.items():
        v = row.get(f"spec.params.{p}")
        params[p] = default if v is None or pd.isna(v) else float(v)
    return system.true_equations(**params), system.var_names


def show_equations(df, key):
    """Pick one dataset and compare every model's equations to the truth."""
    df = df.copy()
    df["setting"] = df.apply(
        lambda r: f"{r['system']} · noise {r['noise'] * 100:g}% · seed {int(r['seed'])}"
                  + (f" · dt {r['dt']:g} · n {int(r['n_samples'])}"
                     if df["dt"].nunique() > 1 or df["n_samples"].nunique() > 1 else ""),
        axis=1)
    choice = st.selectbox("Dataset", sorted(df["setting"].unique()), key=key)
    sub = df[df["setting"] == choice]
    rows = {r["model_name"]: r for _, r in sub.iterrows()}
    first = sub.iloc[0]
    truth = _true_equations(first)

    names = None
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**True system**")
        if truth:
            true_eqs, names = truth
            st.code("\n".join(f"d{v}/dt = {eq}" for v, eq in zip(names, true_eqs)),
                    language=None)
    for i, model in enumerate(model_order(rows)):
        r = rows[model]
        with cols[(i + 1) % 2]:
            st.markdown(f"**{model}** · {r['stability']}, {int(r['n_terms']) if pd.notna(r['n_terms']) else '?'} terms")
            if r["status"] != "ok":
                st.error(r["error"] or "failed")
            elif r["equations"]:
                vs = names or [f"x{k}" for k in range(len(r["equations"]))]
                st.code("\n".join(f"d{v}/dt = {eq}" for v, eq in zip(vs, r["equations"])),
                        language=None)
