"""
PySR candidate tests.

The conversion tests are fast and need no Julia. The end-to-end test is
marked slow (Julia startup + a timed search); run it with: pytest -m slow
"""
import pytest
import sympy

from engine.candidates import pysr_available

pytest.importorskip("sympy")
from engine.candidates_pysr import count_terms, format_equation, to_coefficients  # noqa: E402

V = ["x", "y", "z"]


def test_count_terms_uses_expanded_form():
    assert count_terms("2*x*y - y") == 2
    assert count_terms("x*(28 - z) - y") == 3      # -> 28x - xz - y
    assert count_terms("(x + y)**2") == 3           # x² + 2xy + y²
    assert count_terms("0") == 0


def test_to_coefficients_uses_pysindy_term_names():
    c = to_coefficients("x*(28.0 - z) - y + 0.5", V)
    assert c == {"x": 28.0, "x z": -1.0, "y": -1.0, "1": 0.5}
    vdp = to_coefficients("-x + y - x**2*y", ["x", "y"])
    assert set(vdp) == {"x", "y", "x^2 y"}


def test_to_coefficients_drops_float_dust_and_rejects_non_polynomials():
    assert to_coefficients("x + 1e-17*y", V) == {"x": 1.0}
    assert to_coefficients("sin(x) + y", V) is None
    assert to_coefficients("x / y", V) is None


def test_format_equation_is_expanded_and_rounded():
    s = format_equation("x*(28.0000001 - z)")
    assert sympy.sympify(s) == sympy.sympify("28.0*x - x*z")


def test_registered_only_when_installed():
    from engine.candidates import CANDIDATES
    assert ("PySR" in CANDIDATES) == pysr_available()


@pytest.mark.slow
@pytest.mark.skipif(not pysr_available(), reason="pysr not installed")
def test_pysr_end_to_end_on_van_der_pol():
    import faulthandler

    from engine.benchmarks import make_dataset
    from engine.candidates_pysr import PySRCandidate
    from engine.harness import evaluate

    # Julia's runtime uses deliberate memory faults internally (GC safepoints).
    # pytest's faulthandler reports each one as "Windows fatal exception:
    # access violation" (megabytes of false alarms), so switch it off here only.
    faulthandler.disable()
    try:
        d = make_dataset("Van der Pol", n_samples=2000)
        r = evaluate(PySRCandidate("PySR-test", timeout_per_equation=30), d)
    finally:
        faulthandler.enable()
    assert r["status"] == "ok", r["error"]
    assert r["family"] == "PySR" and r["interpretability"] in ("parsimonious", "complex")
    assert r["reconstruction_error"] < 0.05
    assert r["model_config"]["reproducible"] is False and r["model_config"]["pysr_version"]
