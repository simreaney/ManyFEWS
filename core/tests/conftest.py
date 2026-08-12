"""Shared fixtures. Nothing here touches the network or a database."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from manyfews_core import data as data_mod
from manyfews_core.emulator import FloodEmulator
from manyfews_core.riverflow import load_parameters
from manyfews_core.weather import WeatherSeries

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return data_mod.data_dir()


@pytest.fixture(scope="session")
def params() -> np.ndarray:
    return load_parameters()


@pytest.fixture(scope="session")
def emulator() -> FloodEmulator:
    """The real 302,748-cell emulator. Cached, so this costs ~20 ms after the first run."""
    return FloodEmulator.from_csv()


@pytest.fixture(scope="session")
def ensemble_json() -> dict:
    return json.loads((FIXTURES / "ensemble_sample.json").read_text())


@pytest.fixture(scope="session")
def archive_json() -> dict:
    return json.loads((FIXTURES / "archive_sample.json").read_text())


@pytest.fixture
def synthetic_weather() -> WeatherSeries:
    """
    Sixteen benign days of 6-hourly weather - warm, humid, almost no rain.

    Deterministic, so it doubles as the input for golden-value characterisation.
    """
    n_days = 16
    n = n_days * 4
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = np.array(
        [(start + timedelta(hours=6 * i)).replace(tzinfo=None) for i in range(n)],
        dtype="datetime64[s]",
    )

    phase = np.arange(n) % 4  # 0, 6, 12, 18 UTC
    data = np.empty((n, 6))
    data[:, WeatherSeries.RH] = 80.0 - 10.0 * (phase == 2)
    data[:, WeatherSeries.TMAX] = 273.15 + 28.0 + 3.0 * (phase == 2)
    data[:, WeatherSeries.TMIN] = 273.15 + 21.0 - 2.0 * (phase == 0)
    data[:, WeatherSeries.U] = 1.5
    data[:, WeatherSeries.V] = -0.8
    data[:, WeatherSeries.PRECIP] = 0.4

    return WeatherSeries(times=times, data=data, member="control")


@pytest.fixture
def issue_time(synthetic_weather) -> datetime:
    return synthetic_weather.start
