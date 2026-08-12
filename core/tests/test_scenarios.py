"""Synthetic storms and flow overrides."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from manyfews_core.config import StormConfig
from manyfews_core.scenarios import (
    constant_flow_samples,
    inject_storm,
    inject_storm_ensemble,
    scale_precip,
)
from manyfews_core.weather import WeatherSeries


def test_storm_is_a_noop_when_disabled(synthetic_weather, issue_time):
    out = inject_storm(synthetic_weather, issue_time, StormConfig(enabled=False))
    assert out is synthetic_weather


def test_storm_total_lands_on_the_target_day(synthetic_weather, issue_time):
    storm = StormConfig(enabled=True, total_mm=200.0, days_ahead=2)
    out = inject_storm(synthetic_weather, issue_time, storm)

    # Buckets 8..11 are day 2 (four 6-hour buckets per day).
    day = out.data[8:12, WeatherSeries.PRECIP]
    assert day.sum() == pytest.approx(200.0)
    assert_allclose(day, 50.0)


def test_storm_replaces_rather_than_adds(synthetic_weather, issue_time):
    """Matches ``_testStormOverrides``, which overwrites the bucket value."""
    storm = StormConfig(enabled=True, total_mm=100.0, days_ahead=2)
    out = inject_storm(synthetic_weather, issue_time, storm)
    assert out.data[8, WeatherSeries.PRECIP] == pytest.approx(25.0)


def test_storm_leaves_other_days_untouched(synthetic_weather, issue_time):
    storm = StormConfig(enabled=True, total_mm=200.0, days_ahead=2)
    out = inject_storm(synthetic_weather, issue_time, storm)

    before = np.r_[0:8, 12 : len(synthetic_weather)]
    assert_allclose(
        out.data[before, WeatherSeries.PRECIP],
        synthetic_weather.data[before, WeatherSeries.PRECIP],
    )


def test_storm_returns_a_copy(synthetic_weather, issue_time):
    original = synthetic_weather.data.copy()
    inject_storm(
        synthetic_weather, issue_time, StormConfig(enabled=True, total_mm=200.0)
    )
    assert_allclose(synthetic_weather.data, original)


def test_storm_outside_the_forecast_range_is_a_noop(synthetic_weather, issue_time):
    storm = StormConfig(enabled=True, total_mm=200.0, days_ahead=99)
    out = inject_storm(synthetic_weather, issue_time, storm)
    assert_allclose(out.data, synthetic_weather.data)


def test_storm_applies_to_every_member(synthetic_weather, issue_time):
    storm = StormConfig(enabled=True, total_mm=120.0, days_ahead=1)
    members = inject_storm_ensemble([synthetic_weather] * 3, issue_time, storm)
    assert len(members) == 3
    for member in members:
        assert member.data[4:8, WeatherSeries.PRECIP].sum() == pytest.approx(120.0)


def test_scale_precip(synthetic_weather):
    doubled = scale_precip(synthetic_weather, 2.0)
    assert_allclose(
        doubled.data[:, WeatherSeries.PRECIP],
        synthetic_weather.data[:, WeatherSeries.PRECIP] * 2,
    )


def test_constant_flow_samples():
    samples = constant_flow_samples(160.0, 500)
    assert samples.shape == (500,)
    assert np.ptp(samples) == 0.0
