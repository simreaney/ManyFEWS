"""Weather ingestion, against recorded Open-Meteo responses. No network."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from manyfews_core.weather import (
    WeatherSeries,
    bucket_to_6h,
    buckets_to_series,
    member_suffixes,
    suffix_label,
    wind_components,
    align_series,
)


def test_buckets_sum_precipitation_and_take_temperature_extremes(archive_json):
    hourly = archive_json["hourly"]
    buckets = bucket_to_6h(hourly)

    assert len(buckets) == 8  # 48 hours / 6
    first = buckets[0]
    assert first["precipitation"] == pytest.approx(sum(hourly["precipitation"][:6]))
    assert first["max_temperature"] == pytest.approx(
        max(hourly["temperature_2m"][:6]) + 273.15
    )
    assert first["min_temperature"] == pytest.approx(
        min(hourly["temperature_2m"][:6]) + 273.15
    )
    assert first["relative_humidity"] == pytest.approx(
        sum(hourly["relativehumidity_2m"][:6]) / 6
    )


def test_column_order_puts_max_temperature_before_min(archive_json):
    """
    The single easiest thing in this pipeline to get backwards.

    The Django docstring lists the columns min-then-max while the code that
    builds the array zips them max-then-min. The code is authoritative, and the
    hydrology reads column 1 as Tmax.
    """
    series = buckets_to_series(bucket_to_6h(archive_json["hourly"]))
    assert WeatherSeries.TMAX == 1
    assert WeatherSeries.TMIN == 2
    assert np.all(
        series.data[:, WeatherSeries.TMAX] >= series.data[:, WeatherSeries.TMIN]
    )


def test_member_suffixes_put_control_first(ensemble_json):
    suffixes = member_suffixes(ensemble_json["hourly"])
    assert suffixes[0] == ""
    assert suffix_label(suffixes[0]) == "control"
    assert suffix_label("_member01") == "member01"
    assert all(s.startswith("_member") for s in suffixes[1:])


def test_each_member_parses_independently(ensemble_json):
    hourly = ensemble_json["hourly"]
    control = buckets_to_series(bucket_to_6h(hourly, ""), "control")
    member = buckets_to_series(bucket_to_6h(hourly, "_member01"), "member01")

    assert len(control) == len(member)
    assert_allclose(control.times.astype(np.int64), member.times.astype(np.int64))
    # Different members must not be the same numbers.
    assert not np.allclose(control.data, member.data)


@pytest.mark.parametrize(
    "direction,expected_u,expected_v",
    [
        (0.0, 0.0, -1.0),  # from the north: blows southward
        (90.0, -1.0, 0.0),  # from the east: blows westward
        (180.0, 0.0, 1.0),  # from the south: blows northward
        (270.0, 1.0, 0.0),  # from the west: blows eastward
    ],
)
def test_wind_decomposition(direction, expected_u, expected_v):
    u, v = wind_components(1.0, direction)
    assert u == pytest.approx(expected_u, abs=1e-12)
    assert v == pytest.approx(expected_v, abs=1e-12)


def test_temperatures_are_kelvin(archive_json):
    series = buckets_to_series(bucket_to_6h(archive_json["hourly"]))
    assert series.data[:, WeatherSeries.TMAX].min() > 200.0


def test_incomplete_bucket_is_dropped(archive_json):
    """
    Nulls in the response - which the archive returns for recent dates - make a
    bucket incomplete, and it is dropped rather than guessed at.
    """
    hourly = {k: list(v) for k, v in archive_json["hourly"].items()}
    hourly["precipitation"][6:12] = [None] * 6
    buckets = bucket_to_6h(hourly)
    assert len(buckets) == 7  # the second bucket vanished


def test_validate_rejects_a_gap(archive_json):
    """
    A dropped bucket leaves a hole that shifts the day grouping. The Django
    pipeline never checks for this; validate() must.
    """
    hourly = {k: list(v) for k, v in archive_json["hourly"].items()}
    hourly["precipitation"][6:12] = [None] * 6
    series = buckets_to_series(bucket_to_6h(hourly))
    with pytest.raises(ValueError, match="not 6-hourly contiguous"):
        series.validate()


def test_validate_rejects_a_partial_day(synthetic_weather):
    from dataclasses import replace

    truncated = replace(
        synthetic_weather,
        times=synthetic_weather.times[:6],
        data=synthetic_weather.data[:6],
    )
    with pytest.raises(ValueError, match="whole number of days"):
        truncated.validate()


def test_truncate_to_whole_days(synthetic_weather):
    from dataclasses import replace

    odd = replace(
        synthetic_weather,
        times=synthetic_weather.times[:10],
        data=synthetic_weather.data[:10],
    )
    assert len(odd.truncate_to_whole_days()) == 8


def test_with_precip_returns_a_copy(synthetic_weather):
    original = synthetic_weather.data[:, WeatherSeries.PRECIP].copy()
    modified = synthetic_weather.with_precip(np.full(len(synthetic_weather), 99.0))
    assert modified.data[0, WeatherSeries.PRECIP] == 99.0
    assert_allclose(synthetic_weather.data[:, WeatherSeries.PRECIP], original)


def test_align_series_trims_to_common_times(synthetic_weather):
    from dataclasses import replace

    short = replace(
        synthetic_weather,
        times=synthetic_weather.times[:32],
        data=synthetic_weather.data[:32],
        member="member01",
    )
    aligned = align_series([synthetic_weather, short])
    assert {len(s) for s in aligned} == {32}
