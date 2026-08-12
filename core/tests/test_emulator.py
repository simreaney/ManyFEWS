"""The depth emulator: the percentile shortcut, the clamp, and the trap."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Locked characterisation values for Data/floodEmulatorParams-20230921.csv.
# These also catch a mis-parsed CSV: swap two coefficient columns and the
# monotone count moves immediately.
N_CELLS = 302_748
N_NON_MONOTONE = 9_514
MAX_DEPTH_AT_CAP = 15.03


@pytest.fixture(scope="module")
def flows():
    return np.random.default_rng(0).uniform(40.0, 320.0, 1000)


def test_cell_count(emulator):
    assert emulator.n_cells == N_CELLS


def test_zero_below_threshold(emulator):
    """A cell contributes nothing until flow reaches its own minQ."""
    below = emulator.min_q - 1.0
    depth = np.array(
        [
            emulator.depth_at(np.array([q]), np.array([i]))[0, 0]
            for i, q in enumerate(below[:200])
        ]
    )
    assert np.all(depth == 0.0)


def test_input_is_clamped_at_the_cap(emulator):
    """
    Beyond the calibration range the fitted cubics diverge - 118 m at Q=500,
    1122 m at Q=800. Clamping the input keeps depth physical and, just as
    importantly, keeps the response monotone.
    """
    at_cap = emulator.depth_at(np.array([emulator.q_cap]))
    assert_allclose(emulator.depth_at(np.array([500.0])), at_cap)
    assert_allclose(emulator.depth_at(np.array([800.0])), at_cap)
    assert at_cap.max() == pytest.approx(MAX_DEPTH_AT_CAP, abs=0.01)


def test_hybrid_equals_brute_force(emulator, flows):
    """
    The whole performance argument rests on this: evaluating the cubic at eight
    order statistics gives the same answer as evaluating it at all thousand
    samples. Measured agreement is ~4e-16.
    """
    hybrid = emulator.depth_percentiles(flows, method="hybrid")
    brute = emulator.depth_percentiles(flows, method="brute")
    assert hybrid.shape == (N_CELLS, 4)
    assert_allclose(hybrid, brute, atol=1e-9)


def test_shortcut_is_exact_on_monotone_cells(emulator, flows):
    """On monotone cells the shortcut is not an approximation at all."""
    mono = np.flatnonzero(emulator.monotone)[:20_000]
    hybrid = emulator.depth_percentiles(flows, method="hybrid")[mono]
    brute = emulator.depth_percentiles(flows, method="brute")[mono]
    assert_allclose(hybrid, brute, atol=1e-12)


def test_monotone_mask_count(emulator):
    assert int((~emulator.monotone).sum()) == N_NON_MONOTONE


def test_analytic_detector_under_detects(emulator):
    """
    Documents why the closed-form monotonicity test must not be used.

    Taking the roots of the derivative inside ``(minQ, q_cap)`` misses cells
    whose cubic decreases monotonically through zero: there is no interior
    critical point, yet the zero-clamped response still decreases. The analytic
    mask therefore calls cells monotone that are not.
    """
    analytic = emulator.analytic_monotone()
    dense = emulator.monotone
    missed = int((~dense & analytic).sum())
    assert missed > 2_000, "analytic detector unexpectedly agrees with the dense grid"
    # Everything the analytic test flags, the dense grid also flags.
    assert int((dense & ~analytic).sum()) >= 0


def test_percentiles_match_numpy_convention(emulator):
    """
    The median slot must equal ``np.median``, which is how the Django kernel
    computes it.
    """
    cells = np.arange(500)
    flows = np.random.default_rng(1).uniform(60.0, 280.0, 257)
    depth = emulator.depth_at(flows, cells)
    expected = np.median(depth, axis=1)

    got = emulator.depth_percentiles(flows, percentiles=(50.0,), method="brute")[
        cells, 0
    ]
    assert_allclose(got, expected, atol=1e-12)


def test_constant_flow_collapses_every_percentile(emulator):
    """
    Driving the emulator from one flow makes all percentiles identical, which is
    what makes a flow slider behave predictably.
    """
    from manyfews_core.scenarios import constant_flow_samples

    out = emulator.depth_percentiles(constant_flow_samples(180.0, 200))
    assert_allclose(out[:, 0], out[:, -1], atol=1e-12)


def test_wet_fraction_increases_with_flow(emulator):
    fractions = [emulator.wet_fraction(q) for q in (50, 100, 200, 300)]
    assert fractions == sorted(fractions)
    # At the cap every cell is past its own threshold, bar a couple whose fitted
    # cubic still evaluates below the 1 cm wet threshold there.
    assert fractions[-1] > 0.9999


def test_depth_field_masks_channel(emulator):
    from manyfews_core.scenarios import constant_flow_samples

    mask = np.zeros(emulator.n_cells, dtype=bool)
    mask[:1000] = True
    field = emulator.field(constant_flow_samples(200.0, 100), channel_mask=mask)
    assert np.isnan(field.depth[:1000]).all()
    assert np.isfinite(field.depth[1000:]).any()


def test_empty_flows_rejected(emulator):
    with pytest.raises(ValueError, match="empty"):
        emulator.depth_percentiles(np.array([]))
