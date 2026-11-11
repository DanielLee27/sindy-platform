"""
Recording layer: every evaluation is logged to a SQLite file.

Why SQLite: it ships with Python (no install), it's a single file, each
insert is atomic (a crash can't leave a half-written row the way it can
with CSV), and it can be queried with SQL. load_runs() turns it into a
pandas DataFrame and export_csv() gives a spreadsheet-friendly copy.

Table `runs`, one row per (model, dataset) evaluation:
  identity     run_id, batch_id, created_at, git_commit, schema_version
  keys         dataset_id, system, model_name, family, status, error
  scores       one real column per score (see SCORE_COLUMNS)
  JSON blobs   spec_json      generation settings (synthetic data only)
               features_json  measured features (computable for any data)
               model_config_json, equations_json, extra_json

Keeping spec (how the data was made) separate from features (what can be
measured from the data alone) matters for the future recommender: it may
only learn from features, because a real CSV has no spec.
"""
import json
import math
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
DEFAULT_PATH = Path("results/runs.sqlite")

KEY_COLUMNS = {
    "dataset_id": "TEXT", "system": "TEXT", "model_name": "TEXT", "family": "TEXT",
    "status": "TEXT", "error": "TEXT",
}
SCORE_COLUMNS = {
    "fit_seconds": "REAL", "n_terms": "REAL", "interpretability": "TEXT",
    "reconstruction_error": "REAL", "forecast_nrmse": "REAL",
    "valid_time": "REAL", "valid_time_frac": "REAL",
    "stability": "TEXT", "stability_frac": "REAL",
    "term_precision": "REAL", "term_recall": "REAL", "term_f1": "REAL",
    "coef_error": "REAL",
}
JSON_COLUMNS = {"spec": "spec_json", "features": "features_json",
                "model_config": "model_config_json", "equations": "equations_json"}


def _to_py(v):
    """numpy scalars -> Python; NaN -> None (stored as SQL NULL)."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _json(obj):
    return json.dumps(obj, default=lambda o: o.item() if isinstance(o, np.generic)
                      else o.tolist() if isinstance(o, np.ndarray) else str(o))


def _git_commit():
    """Short commit hash, with '+dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


class RunStore:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.git_commit = _git_commit()
        self.batch_id = uuid.uuid4().hex[:12]  # groups the runs of one session
        self._init_schema()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init_schema(self):
        cols = {
            "run_id": "TEXT PRIMARY KEY", "batch_id": "TEXT", "created_at": "TEXT",
            "git_commit": "TEXT", "schema_version": "INTEGER",
            **KEY_COLUMNS, **SCORE_COLUMNS,
            **{c: "TEXT" for c in JSON_COLUMNS.values()}, "extra_json": "TEXT",
        }
        with self._connect() as con:
            con.execute(f"CREATE TABLE IF NOT EXISTS runs "
                        f"({', '.join(f'{k} {t}' for k, t in cols.items())})")
            # Lightweight migration: a score axis added later becomes a new
            # column; old rows just have NULL there.
            existing = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
            for name, typ in cols.items():
                if name not in existing:
                    con.execute(f"ALTER TABLE runs ADD COLUMN {name} {typ}")
            for col in ("system", "model_name", "dataset_id", "batch_id"):
                con.execute(f"CREATE INDEX IF NOT EXISTS idx_runs_{col} ON runs({col})")

    def log(self, record):
        """Insert one harness record. Returns its run_id."""
        known = set(KEY_COLUMNS) | set(SCORE_COLUMNS) | set(JSON_COLUMNS)
        row = {
            "run_id": uuid.uuid4().hex,
            "batch_id": self.batch_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": self.git_commit,
            "schema_version": SCHEMA_VERSION,
            **{k: _to_py(record.get(k)) for k in (*KEY_COLUMNS, *SCORE_COLUMNS)},
            **{col: _json(record.get(k)) for k, col in JSON_COLUMNS.items()},
            # anything a future harness adds that has no column yet is kept here
            "extra_json": _json({k: v for k, v in record.items() if k not in known}),
        }
        with self._connect() as con:
            con.execute(f"INSERT INTO runs ({', '.join(row)}) "
                        f"VALUES ({', '.join('?' * len(row))})", list(row.values()))
        return row["run_id"]

    def load_runs(self, where=None, params=()):
        """
        All runs as one flat DataFrame. JSON blobs are expanded into columns
        prefixed spec. / feat. / config. (e.g. 'feat.noise_rel_mean').
        Optional SQL filter, e.g. where="system = ?", params=("Lorenz",).
        """
        sql = "SELECT * FROM runs" + (f" WHERE {where}" if where else "") + " ORDER BY created_at"
        with self._connect() as con:
            df = pd.read_sql_query(sql, con, params=params)
        if df.empty:
            return df
        # An all-NULL column comes back as object/None; force numeric scores
        # to float so missing values are NaN, not None.
        for col, typ in SCORE_COLUMNS.items():
            if typ == "REAL" and col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        parts = [df.drop(columns=["spec_json", "features_json", "model_config_json"])]
        for col, prefix in (("spec_json", "spec"), ("features_json", "feat"),
                            ("model_config_json", "config")):
            parts.append(pd.json_normalize(df[col].map(json.loads).tolist())
                         .add_prefix(f"{prefix}."))
        out = pd.concat(parts, axis=1)
        out["equations"] = out.pop("equations_json").map(json.loads)
        # e.g. {"model_extra": {"library_size": 20}}
        out["extra"] = out.pop("extra_json").map(lambda s: json.loads(s) if s else {})
        return out

    def export_csv(self, path):
        df = self.load_runs()
        df.to_csv(path, index=False)
        return path

    def __len__(self):
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
