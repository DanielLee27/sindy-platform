from engine.run_benchmark import main
from engine.store import RunStore


def test_cli_runs_grid_logs_and_exports(tmp_path, capsys):
    db, csv = tmp_path / "r.sqlite", tmp_path / "r.csv"
    df = main(["--systems", "Van der Pol", "--noise", "0", "0.05", "--length", "1500",
               "--db", str(db), "--csv", str(csv), "--quiet"])
    assert len(df) == 4 and len(RunStore(db)) == 4 and csv.exists()
    assert "reconstruction_error" in capsys.readouterr().out


def test_cli_no_store_logs_nothing(tmp_path):
    db = tmp_path / "r.sqlite"
    main(["--systems", "Van der Pol", "--length", "800", "--models", "SINDy-poly",
          "--db", str(db), "--no-store", "--quiet"])
    assert not db.exists()
