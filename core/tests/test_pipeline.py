"""End-to-end runs on synthetic weather. No network, no database."""

import numpy as np
import pytest

from manyfews_core.config import CatchmentConfig, StormConfig
from manyfews_core.riverflow import (
    default_initial_state,
    generate_river_flows,
    run_ensemble,
    spin_up,
)
from manyfews_core.scenarios import inject_storm_ensemble

FLOOD_THRESHOLD = 50.0


def test_timestep_other_than_a_quarter_day_is_rejected():
    """
    The reshape-based day grouping only works at four buckets per day. Upstream
    exposes MODEL_TIMESTEP as a setting anyway, where a wrong value corrupts the
    daily temperature extremes silently.
    """
    with pytest.raises(ValueError, match="timestep_days must be 0.25"):
        CatchmentConfig(timestep_days=0.5)


def test_single_run_shapes(synthetic_weather, params):
    result = generate_river_flows(
        synthetic_weather, default_initial_state(100), params
    )
    n = len(synthetic_weather)
    assert result.flow_m3s.shape == (n, 100)
    assert result.state.shape == (100, 3)
    assert np.isfinite(result.flow_m3s).all()
    assert (result.flow_m3s >= 0).all()


def test_caller_state_is_never_mutated(synthetic_weather, params):
    state = default_initial_state(100)
    before = state.copy()
    generate_river_flows(synthetic_weather, state, params)
    np.testing.assert_allclose(state, before)


def test_ensemble_members_do_not_contaminate_each_other(synthetic_weather, params):
    """
    ``ModelFun`` writes its end state back into the array it is given. If members
    shared one array, member 2 would start where member 1 finished.
    """
    state = spin_up(synthetic_weather, params)
    ens = run_ensemble([synthetic_weather] * 3, state, params)
    np.testing.assert_allclose(ens.flow_m3s[0], ens.flow_m3s[1])
    np.testing.assert_allclose(ens.flow_m3s[1], ens.flow_m3s[2])


def test_ensemble_summary_helpers(synthetic_weather, params):
    state = spin_up(synthetic_weather, params)
    ens = run_ensemble([synthetic_weather] * 2, state, params)

    assert ens.flow_m3s.shape == (2, len(synthetic_weather), 100)
    assert ens.percentiles().shape == (len(synthetic_weather), 4)
    assert 0 <= ens.peak_step() < ens.n_steps
    assert ens.pooled(0).shape == (200,)


@pytest.mark.slow
def test_benign_weather_produces_no_flooding(synthetic_weather, params, emulator):
    """
    The expected outcome on an ordinary day, and the reason both notebooks say so
    explicitly rather than rendering an empty map.
    """
    state = spin_up(synthetic_weather, params)
    ens = run_ensemble([synthetic_weather], state, params)
    pooled = ens.pooled(ens.peak_step())

    assert pooled.max() < FLOOD_THRESHOLD
    assert emulator.field(pooled).wet_cells(50) == 0


@pytest.mark.slow
def test_large_storm_floods_and_stays_physical(synthetic_weather, params, emulator):
    """
    200 mm is roughly where flooding appears. The upper clamp must keep depths
    physical - without it the cubics reach 118 m at Q=500.
    """
    storm = StormConfig(enabled=True, total_mm=200.0, days_ahead=2)
    state = spin_up(synthetic_weather, params)
    forecast = inject_storm_ensemble([synthetic_weather], synthetic_weather.start, storm)

    ens = run_ensemble(forecast, state, params)
    field = emulator.field(ens.pooled(ens.peak_step()))

    assert field.wet_cells(90) > 100_000
    assert field.max_depth(90) < 16.0


@pytest.mark.slow
def test_storm_response_is_monotonic_in_storm_size(synthetic_weather, params):
    """More rain gives more flow. A failure here means the storm is misplaced."""
    state = spin_up(synthetic_weather, params)
    peaks = []
    for mm in (50.0, 100.0, 200.0, 300.0):
        storm = StormConfig(enabled=True, total_mm=mm, days_ahead=2)
        forecast = inject_storm_ensemble(
            [synthetic_weather], synthetic_weather.start, storm
        )
        ens = run_ensemble(forecast, state, params)
        peaks.append(np.percentile(ens.pooled(ens.peak_step()), 90))

    assert peaks == sorted(peaks)
