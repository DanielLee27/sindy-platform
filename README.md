# sindy-platform

Recover governing equations from time-series data with SINDy, and compare
model families (SINDy with different libraries, PySR) on benchmark systems
with known ground truth.

## Setup (Windows)

Use the python.org Python 3.11 (not the Microsoft Store build: PySR's Julia
backend cannot load the Store Python's DLL).

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pip install -r requirements-pysr.txt   # optional: PySR + Julia
```

## Run the app

```
.venv\Scripts\python -m streamlit run app.py
```

Then open http://localhost:8501. Three pages:

- **Explorer**: fit one SINDy model to a demo system or an uploaded CSV and
  inspect the equations, phase portrait and trajectories.
- **Model comparison**: pick systems, noise levels, seeds and models; the run
  happens as a background job (the CLI below) and every run is logged.
- **Run history**: browse, filter and export every logged run.

## Run benchmarks from the command line

```
.venv\Scripts\python -m engine.run_benchmark --noise 0 0.01 0.05 --seeds 0 1 2
.venv\Scripts\python -m engine.run_benchmark --systems Pendulum --models SINDy-poly SINDy-poly+trig
```

Runs are appended to `results/runs.sqlite` (git-ignored).

## Tests

```
.venv\Scripts\python -m pytest            # fast suite
.venv\Scripts\python -m pytest -m slow    # PySR end-to-end (needs Julia, ~1-2 min)
```

## Layout

| Folder | What it holds |
|---|---|
| `pipeline/` | single-fit SINDy pipeline: data, preprocessing, models, plots, demo systems |
| `engine/` | comparison engine: candidates, metrics, harness, benchmark generator, run store, CLI |
| `views/` | Streamlit pages (`app.py` is only the page router) |
| `tests/` | pytest suite, including headless tests of every page |
