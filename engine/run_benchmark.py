"""
Run a benchmark grid, log every run, and print the comparison table.

Examples
  python -m engine.run_benchmark
  python -m engine.run_benchmark --systems Lorenz Rossler --noise 0 0.02 0.1 --seeds 0 1 2
  python -m engine.run_benchmark --dt 0.005 0.02 --length 2000 5000 --csv results/runs.csv
  python -m engine.run_benchmark --no-store        # dry run, nothing logged

Every run is appended to the SQLite store (default results/runs.sqlite), so
repeated invocations accumulate data rather than overwrite it.
"""
import argparse
import sys
import time

import pandas as pd

from engine.benchmarks import make_grid
from engine.candidates import CANDIDATES
from engine.harness import comparison_table, compare
from engine.store import DEFAULT_PATH, RunStore
from pipeline.systems import DEMO_SYSTEMS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--systems", nargs="+", default=list(DEMO_SYSTEMS), choices=list(DEMO_SYSTEMS))
    p.add_argument("--models", nargs="+", default=list(CANDIDATES), choices=list(CANDIDATES))
    p.add_argument("--noise", nargs="+", type=float, default=[0.0, 0.01, 0.05],
                   help="relative noise levels (fraction of each variable's std)")
    p.add_argument("--dt", nargs="+", type=float, default=[0.01], help="sampling intervals")
    p.add_argument("--length", nargs="+", type=int, default=[5000], help="samples per dataset")
    p.add_argument("--seeds", nargs="+", type=int, default=[0], help="noise seeds")
    p.add_argument("--db", default=str(DEFAULT_PATH), help="SQLite file to append runs to")
    p.add_argument("--no-store", action="store_true", help="don't log runs")
    p.add_argument("--csv", help="also export the whole store to this CSV")
    p.add_argument("--quiet", action="store_true", help="no per-run progress lines")
    p.add_argument("--batch-id", help="batch id to log runs under (default: random)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    candidates = {k: CANDIDATES[k] for k in args.models}

    t0 = time.perf_counter()
    datasets = make_grid(systems=args.systems, noises=args.noise, dts=args.dt,
                         lengths=args.length, seeds=args.seeds)
    print(f"{len(datasets)} datasets x {len(candidates)} models = "
          f"{len(datasets) * len(candidates)} runs\n")

    store = None if args.no_store else RunStore(args.db, batch_id=args.batch_id)
    df = compare(candidates, datasets, store=store, verbose=not args.quiet)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n" + comparison_table(df).round(4).to_string())

    failed = df[df["status"] == "failed"]
    if len(failed):
        print(f"\n{len(failed)} failed run(s):")
        for _, r in failed.iterrows():
            print(f"  {r['system']} / {r['model_name']}: {r['error']}")

    print(f"\nDone in {time.perf_counter() - t0:.1f}s.")
    if store is not None:
        print(f"Logged {len(df)} runs to {store.path} (batch {store.batch_id}, "
              f"commit {store.git_commit}); store now holds {len(store)} runs.")
        if args.csv:
            print(f"Exported store to {store.export_csv(args.csv)}")
    return df


if __name__ == "__main__":
    main()
    sys.exit(0)
