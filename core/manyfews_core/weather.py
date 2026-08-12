"""
Weather ingestion from Open-Meteo, and the array contract the hydrology expects.

Ported from ``manyfews/calculations/open_meteo.py`` with the Django ORM writes
removed. Both endpoints are free, keyless, and (verified) send
``access-control-allow-origin: *``, so the same requests work from a browser.

The hydrology has no notion of dates: it treats its input as a flat, strictly
ordered sequence and reshapes every four consecutive rows into one calendar day.
Everything here therefore emits chronological, 6-hour-aligned buckets.

Units follow the original schema: temperatures in Kelvin, wind decomposed into
components in m/s using the meteorological convention that direction is the one
the wind blows *from*::

    U = -speed * sin(direction)
    V = -speed * cos(direction)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, ClassVar, Iterable

import numpy as np
import requests

from .config import CatchmentConfig, ForecastConfig, DEFAULT_CATCHMENT

logger = logging.getLogger(__name__)

__all__ = [
    "WeatherSeries",
    "OpenMeteoClient",
    "offset_time",
    "wind_components",
    "member_suffixes",
    "bucket_to_6h",
    "buckets_to_series",
    "fetch_forecast",
    "fetch_history",
]

_KELVIN_OFFSET = 273.15
_BUCKET_HOURS = 6
_BUCKET_SECONDS = _BUCKET_HOURS * 3600


@dataclass
class WeatherSeries:
    """
    One realisation of weather - a single ensemble member, or an observed
    history - as the ``(N, 6)`` array the hydrology consumes.

    .. warning::
       Column 1 is **max** temperature and column 2 is **min**. This is the
       easiest thing in the whole pipeline to get backwards: the Django
       docstring lists them min-then-max while the code zips them max-then-min.
       The code is authoritative. ``TMAX``/``TMIN`` below exist so nothing has
       to remember.
    """

    times: np.ndarray  # (N,) datetime64[s], UTC, strictly increasing, 6h apart
    data: np.ndarray  # (N, 6) float64
    member: str = "control"

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "relative_humidity_pct",
        "max_temperature_k",
        "min_temperature_k",
        "wind_u_ms",
        "wind_v_ms",
        "precip_mm",
    )
    RH: ClassVar[int] = 0
    TMAX: ClassVar[int] = 1
    TMIN: ClassVar[int] = 2
    U: ClassVar[int] = 3
    V: ClassVar[int] = 4
    PRECIP: ClassVar[int] = 5

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype="datetime64[s]")
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 2 or self.data.shape[1] != 6:
            raise ValueError(f"data must be (N, 6), got {self.data.shape}")
        if self.times.shape[0] != self.data.shape[0]:
            raise ValueError(
                f"times has {self.times.shape[0]} entries but data has "
                f"{self.data.shape[0]} rows"
            )

    def validate(self) -> None:
        """
        Assert the contract the hydrology silently assumes: strictly increasing
        times exactly 6 hours apart, a whole number of days, and no NaNs.

        The Django pipeline checks none of this. When Open-Meteo returns nulls -
        which the archive endpoint does for recent dates - buckets get dropped
        and the day grouping shifts without any error.
        """
        if len(self) == 0:
            raise ValueError("weather series is empty")

        # Contiguity first: a dropped bucket is usually also what makes the
        # length wrong, and the gap is the more useful thing to report.
        gaps = np.diff(self.times).astype("timedelta64[s]").astype(np.int64)
        if len(gaps) and not np.all(gaps == _BUCKET_SECONDS):
            bad = np.flatnonzero(gaps != _BUCKET_SECONDS)
            raise ValueError(
                f"weather series is not 6-hourly contiguous: {len(bad)} gap(s), "
                f"first at index {bad[0]} ({gaps[bad[0]] / 3600:g} h)"
            )
        if len(self) % 4 != 0:
            raise ValueError(
                f"weather series has {len(self)} buckets, which is not a whole "
                "number of days. The hydrology reshapes N/4 x 4, so a partial "
                "day silently corrupts the daily min/max temperature grouping."
            )
        if not np.isfinite(self.data).all():
            n = int((~np.isfinite(self.data)).sum())
            raise ValueError(f"weather series contains {n} non-finite value(s)")

    def truncate_to_whole_days(self) -> "WeatherSeries":
        """Drop trailing buckets so the length is a multiple of 4."""
        n = (len(self) // 4) * 4
        if n == len(self):
            return self
        logger.info("Truncating %d buckets to %d (whole days)", len(self), n)
        return replace(self, times=self.times[:n], data=self.data[:n])

    def with_precip(self, precip: np.ndarray) -> "WeatherSeries":
        """Return a copy with the precipitation column replaced."""
        precip = np.asarray(precip, dtype=np.float64)
        if precip.shape != (len(self),):
            raise ValueError(
                f"precip must have shape ({len(self)},), got {precip.shape}"
            )
        data = self.data.copy()
        data[:, self.PRECIP] = precip
        return replace(self, data=data)

    def slice_days(self, start: datetime, end: datetime) -> "WeatherSeries":
        """Restrict to ``[start, end)``, keeping whole days."""
        lo = np.datetime64(start.astimezone(timezone.utc).replace(tzinfo=None), "s")
        hi = np.datetime64(end.astimezone(timezone.utc).replace(tzinfo=None), "s")
        keep = (self.times >= lo) & (self.times < hi)
        return replace(
            self, times=self.times[keep], data=self.data[keep]
        ).truncate_to_whole_days()

    @property
    def rainfall_mm_day(self) -> np.ndarray:
        """Precipitation converted from mm per bucket to mm/day."""
        return self.data[:, self.PRECIP] / 0.25

    @property
    def start(self) -> datetime:
        return self.times[0].astype(datetime).replace(tzinfo=timezone.utc)

    def summary(self) -> str:
        return (
            f"{self.member}: {len(self)} buckets, "
            f"{self.times[0]} to {self.times[-1]}, "
            f"total precip {self.data[:, self.PRECIP].sum():.1f} mm"
        )


def offset_time(backDays: int) -> tuple[datetime, datetime]:
    """
    UTC ``(start, end)`` for the calendar day ``backDays`` days ago:
    00:00:00 to 23:55:00. Ported from ``open_meteo.offsetTime``.
    """
    startDate = datetime.now(tz=timezone.utc) - timedelta(days=backDays)
    startTime = datetime(
        startDate.year, startDate.month, startDate.day, 0, 0, 0, 0, timezone.utc
    )
    endTime = datetime(
        startDate.year, startDate.month, startDate.day, 23, 55, 0, 0, timezone.utc
    )
    return startTime, endTime


def _kmh_to_ms(speed_kmh: float) -> float:
    return speed_kmh / 3.6


def wind_components(speed_ms: float, direction_deg: float) -> tuple[float, float]:
    """Wind speed and meteorological direction to zonal/meridional components."""
    direction_rad = math.radians(direction_deg)
    return -speed_ms * math.sin(direction_rad), -speed_ms * math.cos(direction_rad)


def member_suffixes(hourly: dict) -> list[str]:
    """
    Ensemble member suffixes present in a response, control run first.

    Keys look like ``precipitation`` (control) and ``precipitation_member01``.
    """
    suffixes = []
    for key in hourly:
        if key == "precipitation":
            suffixes.append("")
        elif key.startswith("precipitation_member"):
            suffixes.append(key[len("precipitation") :])
    return sorted(suffixes, key=lambda s: (s != "", s))


def suffix_label(suffix: str) -> str:
    """``""`` -> ``"control"``, ``"_member01"`` -> ``"member01"``."""
    return f"member{suffix[len('_member'):]}" if suffix else "control"


def bucket_to_6h(hourly: dict, suffix: str = "") -> list[dict]:
    """
    Aggregate one member's hourly data into 6-hour buckets aligned to the
    response's own start time.

    Precipitation sums; temperature takes the min and max within the bucket and
    converts to Kelvin; humidity and the decomposed wind components average.
    Incomplete buckets are dropped rather than guessed at - which is why
    :meth:`WeatherSeries.validate` exists to catch the resulting gap.
    """
    times = hourly.get("time", [])
    precip = hourly.get(f"precipitation{suffix}", [])
    temp = hourly.get(f"temperature_2m{suffix}", [])
    wspd = hourly.get(f"windspeed_10m{suffix}", [])
    wdir = hourly.get(f"winddirection_10m{suffix}", [])
    rh = hourly.get(f"relativehumidity_2m{suffix}", [])

    buckets: list[dict] = []
    full_buckets_end = len(times) - (len(times) % _BUCKET_HOURS)

    for start in range(0, full_buckets_end, _BUCKET_HOURS):
        idxs = range(start, start + _BUCKET_HOURS)

        bucket_precip = [
            precip[i] for i in idxs if i < len(precip) and precip[i] is not None
        ]
        bucket_temp = [temp[i] for i in idxs if i < len(temp) and temp[i] is not None]
        bucket_rh = [rh[i] for i in idxs if i < len(rh) and rh[i] is not None]
        bucket_uv = [
            wind_components(_kmh_to_ms(wspd[i]), wdir[i])
            for i in idxs
            if i < len(wspd)
            and i < len(wdir)
            and wspd[i] is not None
            and wdir[i] is not None
        ]

        if not (bucket_precip and bucket_temp and bucket_rh and bucket_uv):
            logger.warning(
                "Dropping incomplete 6h bucket at index %d (member %r) - this "
                "will leave a gap that WeatherSeries.validate() will reject",
                start,
                suffix_label(suffix),
            )
            continue

        buckets.append(
            {
                "time": datetime.fromtimestamp(times[start], tz=timezone.utc),
                "precipitation": sum(bucket_precip),
                "min_temperature": min(bucket_temp) + _KELVIN_OFFSET,
                "max_temperature": max(bucket_temp) + _KELVIN_OFFSET,
                "wind_u": sum(u for u, _ in bucket_uv) / len(bucket_uv),
                "wind_v": sum(v for _, v in bucket_uv) / len(bucket_uv),
                "relative_humidity": sum(bucket_rh) / len(bucket_rh),
            }
        )

    return buckets


def buckets_to_series(buckets: list[dict], member: str = "control") -> WeatherSeries:
    """Assemble bucket dicts into the ``(N, 6)`` array contract."""
    if not buckets:
        raise ValueError(f"no complete 6-hour buckets for member {member!r}")
    times = np.array(
        [b["time"].replace(tzinfo=None) for b in buckets], dtype="datetime64[s]"
    )
    data = np.array(
        [
            [
                b["relative_humidity"],
                b["max_temperature"],  # column 1 is MAX
                b["min_temperature"],  # column 2 is MIN
                b["wind_u"],
                b["wind_v"],
                b["precipitation"],
            ]
            for b in buckets
        ],
        dtype=np.float64,
    )
    return WeatherSeries(times=times, data=data, member=member)


class OpenMeteoClient:
    """
    Thin Open-Meteo wrapper with retries.

    Replaces the ``tenacity`` decorator in the Django module with a short loop,
    keeping this package's dependencies to numpy and requests.
    """

    ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    HOURLY_VARIABLES = (
        "precipitation",
        "temperature_2m",
        "windspeed_10m",
        "winddirection_10m",
        "relativehumidity_2m",
    )

    def __init__(
        self,
        timeout_s: float = 60.0,
        retries: int = 3,
        backoff_s: float = 5.0,
        session: requests.Session | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.session = session or requests.Session()

    def _get(self, url: str, params: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_s)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                if attempt == self.retries - 1:
                    break
                wait = self.backoff_s * (2**attempt)
                logger.warning(
                    "Open-Meteo request failed (attempt %d/%d): %s - retrying in %gs",
                    attempt + 1,
                    self.retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"Open-Meteo request to {url} failed: {last}") from last

    def fetch_ensemble(
        self, lat: float, lon: float, model: str, start_date: str, end_date: str
    ) -> dict:
        return self._get(
            self.ENSEMBLE_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(self.HOURLY_VARIABLES),
                "models": model,
                "start_date": start_date,
                "end_date": end_date,
                "timeformat": "unixtime",
                "timezone": "UTC",
            },
        )

    def fetch_archive(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> dict:
        return self._get(
            self.ARCHIVE_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(self.HOURLY_VARIABLES),
                "start_date": start_date,
                "end_date": end_date,
                "timeformat": "unixtime",
                "timezone": "UTC",
            },
        )


def fetch_forecast(
    catchment: CatchmentConfig = DEFAULT_CATCHMENT,
    cfg: ForecastConfig = ForecastConfig(),
    client: OpenMeteoClient | None = None,
    start: date | None = None,
) -> list[WeatherSeries]:
    """
    Fetch the ensemble forecast, one :class:`WeatherSeries` per member.

    :return: control run first, then members in ascending order, truncated to
        ``cfg.max_members``.
    """
    client = client or OpenMeteoClient(cfg.timeout_s, cfg.retries, cfg.backoff_s)
    issue_date = (
        datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        if start
        else offset_time(0)[0]
    )
    end_date = issue_date + timedelta(days=cfg.forecast_days - 1)

    logger.info(
        "Fetching Open-Meteo ensemble: model=%s lat=%s lon=%s %s..%s",
        cfg.model,
        catchment.weather_lat,
        catchment.weather_lon,
        issue_date.date(),
        end_date.date(),
    )
    raw = client.fetch_ensemble(
        catchment.weather_lat,
        catchment.weather_lon,
        cfg.model,
        issue_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )
    hourly = raw.get("hourly", {})
    suffixes = member_suffixes(hourly)
    if not suffixes:
        raise ValueError("Open-Meteo response contained no precipitation columns")
    if cfg.max_members:
        suffixes = suffixes[: cfg.max_members]

    series = [
        buckets_to_series(
            bucket_to_6h(hourly, s), suffix_label(s)
        ).truncate_to_whole_days()
        for s in suffixes
    ]
    logger.info(
        "Parsed %d ensemble member(s), %d buckets each", len(series), len(series[0])
    )
    return series


def fetch_history(
    catchment: CatchmentConfig = DEFAULT_CATCHMENT,
    cfg: ForecastConfig = ForecastConfig(),
    client: OpenMeteoClient | None = None,
    days: int | None = None,
    end: date | None = None,
) -> WeatherSeries:
    """
    Fetch observed history for the spin-up, ending far enough back to avoid the
    archive's ingestion lag.

    Django's ``initialModelSetUp`` requests data up to *yesterday*. Open-Meteo's
    ERA5T archive lags real time by roughly five days, so those recent hours come
    back null, get dropped by :func:`bucket_to_6h`, and shift the day grouping
    with no error raised. Ending the window at ``today - archive_lag_days``
    avoids that entirely; the resulting state is a day or two stale, which
    matters far less than a corrupted spin-up.
    """
    client = client or OpenMeteoClient(cfg.timeout_s, cfg.retries, cfg.backoff_s)
    days = days or cfg.spinup_days

    end_dt = (
        datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)
        if end
        else offset_time(cfg.archive_lag_days)[0]
    )
    start_dt = end_dt - timedelta(days=days - 1)

    logger.info(
        "Fetching Open-Meteo archive: lat=%s lon=%s %s..%s (%d days)",
        catchment.weather_lat,
        catchment.weather_lon,
        start_dt.date(),
        end_dt.date(),
        days,
    )
    raw = client.fetch_archive(
        catchment.weather_lat,
        catchment.weather_lon,
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )
    series = buckets_to_series(bucket_to_6h(raw.get("hourly", {})), "historical")

    expected = days * (24 // _BUCKET_HOURS)
    if len(series) < expected:
        raise RuntimeError(
            f"Open-Meteo archive incomplete for {start_dt.date()}..{end_dt.date()}: "
            f"expected {expected} 6-hour buckets, got {len(series)}. Increase "
            f"ForecastConfig.archive_lag_days (currently {cfg.archive_lag_days})."
        )
    series = series.truncate_to_whole_days()
    series.validate()
    return series


def align_series(series: Iterable[WeatherSeries]) -> list[WeatherSeries]:
    """
    Restrict a set of ensemble members to their common time axis.

    Members can end up with different lengths because :func:`bucket_to_6h` drops
    incomplete buckets independently per member. The Django pipeline does not
    check this, so one short member silently misaligns the ``reshape(N/4, 4)``
    day grouping for that member only.
    """
    series = list(series)
    if not series:
        return []
    common = series[0].times
    for s in series[1:]:
        common = np.intersect1d(common, s.times)
    if len(common) == 0:
        raise ValueError("ensemble members share no common time steps")

    gaps = np.diff(common).astype("timedelta64[s]").astype(np.int64)
    if len(gaps) and not np.all(gaps == _BUCKET_SECONDS):
        raise ValueError("common time axis across ensemble members is not contiguous")

    aligned = []
    for s in series:
        keep = np.isin(s.times, common)
        aligned.append(
            replace(s, times=s.times[keep], data=s.data[keep]).truncate_to_whole_days()
        )
    return aligned
