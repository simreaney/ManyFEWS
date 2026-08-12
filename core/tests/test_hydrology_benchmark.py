"""
Parity with the MATLAB reference implementation.

This is the anchor test for the whole port. The benchmark CSVs in ``Data/`` are
outputs of the original MATLAB model, and they let the hydrology be checked
without going anywhere near weather ingestion or FAO56: feed the recorded
rainfall and PET straight into ``ModelFun`` alongside the reference initial
conditions, and compare flow and end state.

If this fails, the numerics were changed. Nothing else in the port matters until
it passes again.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from manyfews_core.hydrology import ModelFun, PDMmodel, RoutingFun

# The benchmark files store three decimal places, so this tolerance is set by the
# reference data's own rounding, not by any looseness in the code. Measured
# agreement is 5.0e-4 for flow and 7.9e-5 for state. Do not loosen it.
BENCHMARK_ATOL = 1e-3

CATCHMENT_AREA_KM2 = 212.2640
DT = 0.25


@pytest.fixture(scope="module")
def benchmark(data_dir):
    return {
        "qp": np.loadtxt(data_dir / "qp_Benchmark.csv"),
        "Ep": np.loadtxt(data_dir / "Eq_Benchmark.csv"),
        "X": np.loadtxt(
            data_dir / "RainfallRunoffModelParameters.csv",
            delimiter=",",
            usecols=range(4),
        ),
        # This file is the benchmark's *input* state, which is what makes the
        # comparison possible at all.
        "F0_in": np.loadtxt(
            data_dir / "RainfallRunoffModelInitialConditions.csv", delimiter=","
        ),
        "Q_ref": np.loadtxt(data_dir / "Q_Benchmark.csv", delimiter=","),
        # NOTE: whitespace-delimited, unlike every other benchmark file.
        "F0_ref": np.loadtxt(data_dir / "F0_Benchmark.csv"),
    }


def test_flow_matches_matlab_reference(benchmark):
    Q, _ = ModelFun(
        benchmark["qp"],
        benchmark["Ep"],
        DT,
        CATCHMENT_AREA_KM2,
        benchmark["X"],
        benchmark["F0_in"].copy(),
    )
    assert Q.shape == benchmark["Q_ref"].shape
    assert_allclose(Q, benchmark["Q_ref"], atol=BENCHMARK_ATOL)


def test_end_state_matches_matlab_reference(benchmark):
    _, state = ModelFun(
        benchmark["qp"],
        benchmark["Ep"],
        DT,
        CATCHMENT_AREA_KM2,
        benchmark["X"],
        benchmark["F0_in"].copy(),
    )
    assert_allclose(state, benchmark["F0_ref"], atol=BENCHMARK_ATOL)


def test_modelfun_mutates_state_in_place(benchmark):
    """
    ``ModelFun`` writes end-of-run state back into the array it was handed.

    This is load-bearing rather than incidental: ``run_ensemble`` relies on it
    being true, and passes each member a private copy precisely because of it. If
    this ever stops holding, revisit that copy.
    """
    state = benchmark["F0_in"].copy()
    before = state.copy()
    _, returned = ModelFun(
        benchmark["qp"], benchmark["Ep"], DT, CATCHMENT_AREA_KM2, benchmark["X"], state
    )
    assert returned is state
    assert not np.allclose(state, before)


def test_pdm_storage_stays_within_bounds():
    """Storage is clamped to [0, Smax] regardless of how hard it is forced."""
    n = 200
    rng = np.random.default_rng(0)
    qp = rng.uniform(0, 300, n)
    Ep = rng.uniform(0, 8, n)
    smax = 120.0

    _, _, _, S = PDMmodel(qp, Ep, smax, 1, 40.0, DT, smax / 2)
    assert S.min() >= 0.0
    assert S.max() <= smax + 1e-9


def test_linear_routing_store_decays_without_inflow():
    """With no inflow, the linear store empties monotonically."""
    q = RoutingFun(np.zeros(50), 10.0, 1, DT, 5.0)
    assert np.all(np.diff(q) <= 1e-12)
    assert q[-1] < q[0]


def test_routing_returns_same_length_as_input():
    for b, x in ((1, 10.0), (5 / 3, 150.0)):
        assert RoutingFun(np.ones(37), x, b, DT, 1.0).shape == (37,)
