"""
FAO56 characterisation tests.

There is no MATLAB reference for evapotranspiration-from-weather, so these are
golden values captured from the **Django implementation** at
``manyfews/calculations/generate_river_flows.py`` by stubbing out its django and
celery imports and calling ``FAO56`` directly. The port reproduces them exactly
(measured difference 0.0), so any drift here is a real change in behaviour.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from manyfews_core.hydrology import FAO56

DT = 0.25
ALT = 1157.0
LAT = -7.125

# Captured from the upstream Django implementation - see module docstring.
GOLDEN_ETO_HEAD = np.array(
    [
        4.3642717952,
        4.4067126252,
        4.6181448789,
        4.4073555842,
        4.3655365304,
        4.4080128129,
        4.6194904774,
        4.4086839542,
    ]
)
GOLDEN_E0_HEAD = np.array(
    [
        5.9843363098,
        6.0289608110,
        6.2569163434,
        6.0298324349,
        5.9860570415,
        6.0307234036,
        6.2587310277,
        6.0316332327,
    ]
)
GOLDEN_ETO_SUM = 285.4709432511
GOLDEN_E0_SUM = 389.7860935077


@pytest.fixture
def inputs(synthetic_weather):
    """Reduce the shared synthetic series to FAO56's argument list."""
    from manyfews_core.weather import WeatherSeries

    n = 64
    data = synthetic_weather.data[:n]
    temp_max = data[:, WeatherSeries.TMAX] - 273.15
    temp_min = data[:, WeatherSeries.TMIN] - 273.15
    return {
        "dt": DT,
        "predictionDate": synthetic_weather.start,
        "Tmin": np.repeat(temp_min.reshape(n // 4, 4).min(axis=1), 4),
        "Tmax": np.repeat(temp_max.reshape(n // 4, 4).max(axis=1), 4),
        "alt": ALT,
        "lat": LAT,
        "T": (temp_min + temp_max) / 2,
        "u2": np.full(n, 1.2),
        "RH": data[:, WeatherSeries.RH],
    }


def test_matches_upstream_golden_values(inputs):
    eto, e0 = FAO56(**inputs)
    assert_allclose(eto[:8], GOLDEN_ETO_HEAD, rtol=0, atol=1e-9)
    assert_allclose(e0[:8], GOLDEN_E0_HEAD, rtol=0, atol=1e-9)
    assert eto.sum() == pytest.approx(GOLDEN_ETO_SUM, abs=1e-7)
    assert e0.sum() == pytest.approx(GOLDEN_E0_SUM, abs=1e-7)


def test_rs_assignment_is_live(inputs):
    """
    Regression guard for the trap described in ``hydrology`` module docstring.

    Upstream, the Hargreaves solar radiation assignment sits inside
    ``try: Rs / except NameError:``. Five other blocks in that file share the
    shape but are dead - their name is always a parameter. This one is not:
    ``Rs`` is never bound, so the ``except`` body is the only live path. Anyone
    tidying "dead code" by pattern-matching will delete the wrong half and make
    the function raise ``NameError`` at ``Rso``.

    A wrong-but-not-crashing edit would also be caught, because ETo depends on
    ``Rs`` throughout.
    """
    eto, _ = FAO56(**inputs)
    assert np.isfinite(eto).all()
    assert_allclose(eto[:8], GOLDEN_ETO_HEAD, rtol=0, atol=1e-9)


def test_temperature_swap_is_a_noop(inputs):
    """
    Upstream reassigns ``Tmax`` and then computes ``Tmin`` from the *new* value,
    so the intended swap does nothing. Preserved deliberately; pinned here so
    nobody "fixes" it without realising parity breaks.
    """
    swapped = dict(inputs)
    swapped["Tmin"], swapped["Tmax"] = inputs["Tmax"].copy(), inputs["Tmin"].copy()

    normal_eto, _ = FAO56(**inputs)
    swapped_eto, _ = FAO56(**swapped)

    # A real swap-guard would make these identical. Because the guard is a no-op,
    # feeding min and max the other way round changes the answer.
    assert not np.allclose(normal_eto, swapped_eto)


def test_open_water_evaporation_exceeds_reference_et(inputs):
    """Lower albedo means more net radiation, so E0 > ETo."""
    eto, e0 = FAO56(**inputs)
    assert np.all(e0 > eto)


def test_rs_over_rso_is_clipped_at_one(inputs):
    """
    A large diurnal range would push Rs/Rso above 1 without the in-place clip,
    driving net longwave - and therefore ETo - to implausible values.
    """
    extreme = dict(inputs)
    extreme["Tmax"] = inputs["Tmax"] + 25.0
    eto, _ = FAO56(**extreme)
    assert np.isfinite(eto).all()
    assert eto.max() < 40.0
