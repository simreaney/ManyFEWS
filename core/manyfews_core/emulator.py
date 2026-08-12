"""
The statistical flood-inundation emulator.

A 2D hydraulic model was run offline over a range of river flows, and a cubic in
flow rate was fitted per ground cell. Only those fitted coefficients ship. For
each of the 302,748 cells::

    depth(Q) = max(0, P0 + Q*(P1 + Q*(P2 + Q*P3)))   if Q >= minQ
             = 0                                      otherwise

There is no spatial coupling, no time dependence and no state. The entire
inundation surface is a pure function of one scalar, which is what makes an
interactive slider - and a fully client-side static site - possible at all.

Two things differ from ``manyfews/calculations/flood_risk.py``:

**An upper clamp.** The Django app bounds only negative depths. The fitted cubics
diverge above their calibration range: measured maxima are 15 m at Q=300, 118 m
at Q=500 and 1122 m at Q=800, and the wet-cell count actually *falls* above 300
as cubics turn over and go negative. Clamping the input at ``q_cap`` keeps the
output physical.

**A percentile shortcut.** The Django app evaluates the cubic at every one of
~1,000 pooled flow samples per cell, then takes percentiles - a numba loop that
the source itself flags as taking "several hours for 1 time". Because the clamped
response is monotone in Q for ~97% of cells, the percentile of the depths equals
the depth at the percentile of the flows, so 8 evaluations suffice for those
cells. See :meth:`FloodEmulator.depth_percentiles`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np

from .config import EmulatorConfig
from .data import EMULATOR_CSV, cached_array, data_path

logger = logging.getLogger(__name__)

__all__ = ["FloodEmulator", "DepthField"]

_MONOTONE_GRID = 801
_MONOTONE_TOL = -1e-9


@dataclass
class FloodEmulator:
    """
    Per-cell depth polynomials loaded from the emulator parameter CSV.

    All arrays have shape ``(n_cells,)`` and are index-aligned: cell ``i`` sits at
    ``(lng[i], lat[i])`` with side ``size[i]`` degrees.
    """

    lng: np.ndarray
    lat: np.ndarray
    size: np.ndarray
    beta0: np.ndarray
    beta1: np.ndarray
    beta2: np.ndarray
    beta3: np.ndarray
    min_q: np.ndarray
    q_cap: float = 300.0
    source: Path | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_csv(
        cls,
        path: str | Path | None = None,
        q_cap: float = 300.0,
        cache: bool = True,
    ) -> "FloodEmulator":
        """
        Load ``floodEmulatorParams-*.csv`` (22.9 MB, 302,748 rows).

        Parsing takes several seconds, so the result is memoised as a sibling
        ``.npy``; subsequent loads take about 20 ms. Columns are
        ``lng, lat, size, P0, P1, P2, P3, minQ``.
        """
        path = Path(path) if path else data_path(EMULATOR_CSV)

        def build() -> np.ndarray:
            logger.info("Parsing %s (this takes a few seconds)", path.name)
            return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)

        table = np.asarray(cached_array(path, "cells", build, cache))
        if table.ndim != 2 or table.shape[1] != 8:
            raise ValueError(
                f"expected 8 columns (lng,lat,size,P0..P3,minQ) in {path}, "
                f"got shape {table.shape}"
            )

        return cls(
            lng=table[:, 0],
            lat=table[:, 1],
            size=table[:, 2],
            beta0=table[:, 3],
            beta1=table[:, 4],
            beta2=table[:, 5],
            beta3=table[:, 6],
            min_q=table[:, 7],
            q_cap=q_cap,
            source=path,
        )

    # ------------------------------------------------------------- properties

    @property
    def n_cells(self) -> int:
        return int(self.lng.shape[0])

    @property
    def cell_size(self) -> float:
        """The single grid resolution in degrees (the file uses a uniform one)."""
        return float(self.size[0])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(lat_min, lng_min, lat_max, lng_max)`` - Leaflet's ordering."""
        return (
            float(self.lat.min()),
            float(self.lng.min()),
            float(self.lat.max()),
            float(self.lng.max()),
        )

    # ------------------------------------------------------------ core kernel

    def depth_at(
        self, flows: np.ndarray, cells: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Depth for every cell at every supplied flow: ``(n_cells, n_flows)``.

        This is the vectorised replacement for the per-pixel numba loop. Memory
        is 8 bytes per cell per flow, so keep ``n_flows`` small or pass ``cells``
        to restrict to a subset.
        """
        flows = np.atleast_1d(np.asarray(flows, dtype=np.float64))
        b0, b1, b2, b3, mq = self._coeffs(cells)

        q = np.minimum(flows, self.q_cap)[None, :]
        depth = b0[:, None] + q * (b1[:, None] + q * (b2[:, None] + q * b3[:, None]))
        # Below the cell's threshold there is no flooding at all. Compare against
        # the *unclamped* flow, matching the original.
        depth = np.where(flows[None, :] < mq[:, None], 0.0, depth)
        np.maximum(depth, 0.0, out=depth)
        return depth

    def _coeffs(self, cells: np.ndarray | None):
        if cells is None:
            return self.beta0, self.beta1, self.beta2, self.beta3, self.min_q
        return (
            self.beta0[cells],
            self.beta1[cells],
            self.beta2[cells],
            self.beta3[cells],
            self.min_q[cells],
        )

    # -------------------------------------------------------- monotonicity

    @cached_property
    def monotone(self) -> np.ndarray:
        """
        Boolean mask of cells whose clamped depth is non-decreasing in Q.

        Determined by evaluating on a dense grid spanning ``[minQ, q_cap]``, not
        by the closed form.

        .. warning::
           Do **not** "optimise" this into the analytic test (real roots of
           ``3*P3*Q**2 + 2*P2*Q + P1`` inside the interval). That misses 2,596 of
           the 9,514 non-monotone cells, because a cubic that decreases
           monotonically through zero has no interior critical point yet its
           zero-clamped form still decreases. ``tests/test_emulator.py`` pins
           this with a test asserting the analytic mask is a strict subset.
        """

        def build() -> np.ndarray:
            logger.info("Computing monotonicity mask over %d cells", self.n_cells)
            return self._monotone_mask()

        if self.source is not None:
            mask = cached_array(
                self.source, f"monotone-q{self.q_cap:g}", lambda: self._monotone_mask()
            )
            return np.asarray(mask).astype(bool)
        return build().astype(bool)

    def _monotone_mask(self, n_grid: int = _MONOTONE_GRID, chunk: int = 40_000):
        out = np.empty(self.n_cells, dtype=bool)
        frac = np.linspace(0.0, 1.0, n_grid)[None, :]

        for start in range(0, self.n_cells, chunk):
            sl = slice(start, min(start + chunk, self.n_cells))
            mq = self.min_q[sl][:, None]
            # Per-cell grid from that cell's own threshold up to the cap.
            q = mq + frac * (self.q_cap - mq)
            depth = (
                self.beta0[sl][:, None]
                + q
                * (
                    self.beta1[sl][:, None]
                    + q * (self.beta2[sl][:, None] + q * self.beta3[sl][:, None])
                )
            )
            np.maximum(depth, 0.0, out=depth)
            out[sl] = (np.diff(depth, axis=1) >= _MONOTONE_TOL).all(axis=1)

        return out

    def analytic_monotone(self) -> np.ndarray:
        """
        The tempting-but-wrong closed-form monotonicity test, kept only so the
        test suite can demonstrate that it under-detects. Not used in anger.
        """
        a, b, c = 3 * self.beta3, 2 * self.beta2, self.beta1
        disc = b * b - 4 * a * c
        ok = np.ones(self.n_cells, dtype=bool)
        real = disc >= 0
        sqrt_disc = np.sqrt(np.where(real, disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            for root in ((-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)):
                inside = real & np.isfinite(root) & (root > self.min_q) & (root < self.q_cap)
                ok &= ~inside
        return ok

    # ------------------------------------------------------------ percentiles

    def depth_percentiles(
        self,
        flows: np.ndarray,
        percentiles: tuple[float, ...] = (10.0, 30.0, 50.0, 90.0),
        method: Literal["auto", "hybrid", "brute"] = "auto",
        chunk: int = 20_000,
    ) -> np.ndarray:
        """
        Depth percentiles across pooled flow samples: ``(n_cells, n_pct)``.

        ``flows`` is the pooled sample population for one forecast time - every
        weather ensemble member times every rainfall-runoff parameter set.

        The ``hybrid`` path exploits the fact that sorting depths is sorting
        flows whenever the response is monotone. NumPy's default linear
        percentile interpolates between the two order statistics bracketing
        ``(M-1)*p/100``, so for a monotone cell the answer is exactly the same
        interpolation applied to the depths at those two flows. That is 8 cubic
        evaluations instead of ~1,000. The ~3% of cells that are not monotone are
        brute-forced. Measured against the full brute-force path the agreement is
        2.2e-16, and the whole call drops from 3.8 s to 0.16 s.
        """
        flows = np.atleast_1d(np.asarray(flows, dtype=np.float64))
        if flows.size == 0:
            raise ValueError("flows is empty")
        pcts = tuple(float(p) for p in percentiles)

        if method == "brute" or (method == "auto" and flows.size <= 16):
            return self._percentiles_brute(flows, pcts, chunk)
        return self._percentiles_hybrid(flows, pcts, chunk)

    def _percentiles_brute(self, flows, pcts, chunk) -> np.ndarray:
        """Reference implementation: evaluate everywhere, then reduce."""
        out = np.empty((self.n_cells, len(pcts)), dtype=np.float64)
        for start in range(0, self.n_cells, chunk):
            sl = slice(start, min(start + chunk, self.n_cells))
            cells = np.arange(sl.start, sl.stop)
            depth = self.depth_at(flows, cells)
            out[sl] = np.percentile(depth, list(pcts), axis=1).T
        return out

    def _percentiles_hybrid(self, flows, pcts, chunk) -> np.ndarray:
        xs = np.sort(flows)
        M = xs.size

        # The two order statistics bracketing each percentile, and the weight
        # between them - NumPy's default 'linear' method.
        lo_idx, hi_idx, frac = [], [], []
        for p in pcts:
            pos = (M - 1) * p / 100.0
            k = int(np.floor(pos))
            lo_idx.append(k)
            hi_idx.append(min(k + 1, M - 1))
            frac.append(pos - k)
        probe = xs[np.array(lo_idx + hi_idx)]
        frac = np.array(frac)[None, :]

        depth = self.depth_at(probe)  # (n_cells, 2 * n_pct)
        n = len(pcts)
        out = depth[:, :n] + frac * (depth[:, n:] - depth[:, :n])

        bad = np.flatnonzero(~self.monotone)
        if bad.size:
            logger.debug("Brute-forcing %d non-monotone cell(s)", bad.size)
            for start in range(0, bad.size, chunk):
                cells = bad[start : start + chunk]
                out[cells] = np.percentile(
                    self.depth_at(xs, cells), list(pcts), axis=1
                ).T
        return out

    # -------------------------------------------------------------- summaries

    def wet_fraction(self, flow: float, threshold_m: float = 0.01) -> float:
        """Fraction of cells with depth above ``threshold_m`` at a single flow."""
        return float((self.depth_at(np.array([flow]))[:, 0] > threshold_m).mean())

    def field(
        self,
        flows: np.ndarray,
        cfg: EmulatorConfig = EmulatorConfig(),
        channel_mask: np.ndarray | None = None,
    ) -> "DepthField":
        """Compute a :class:`DepthField` for one pooled flow population."""
        depth = self.depth_percentiles(
            flows, cfg.percentiles, method=cfg.method
        )
        if channel_mask is not None and cfg.mask_channel:
            depth = depth.copy()
            depth[channel_mask] = np.nan
        flows = np.asarray(flows, dtype=np.float64)
        return DepthField(
            emulator=self,
            percentiles=tuple(cfg.percentiles),
            depth=depth,
            flow_summary={
                "n_samples": float(flows.size),
                "min": float(flows.min()),
                "median": float(np.median(flows)),
                "max": float(flows.max()),
            },
        )


@dataclass
class DepthField:
    """Depth percentiles for every cell at one forecast time."""

    emulator: FloodEmulator
    percentiles: tuple[float, ...]
    depth: np.ndarray  # (n_cells, n_pct); NaN where channel-masked
    flow_summary: dict[str, float]

    def _index(self, pct: float) -> int:
        try:
            return self.percentiles.index(pct)
        except ValueError:
            raise ValueError(
                f"percentile {pct} not computed; available: {self.percentiles}"
            ) from None

    def layer(self, pct: float = 50.0) -> np.ndarray:
        """Depths at one percentile: ``(n_cells,)``."""
        return self.depth[:, self._index(pct)]

    def wet_cells(self, pct: float = 50.0, threshold_m: float = 0.01) -> int:
        layer = self.layer(pct)
        return int(np.count_nonzero(np.nan_to_num(layer) > threshold_m))

    def max_depth(self, pct: float = 50.0) -> float:
        layer = self.layer(pct)
        return float(np.nanmax(layer)) if np.isfinite(layer).any() else 0.0

    def mean_wet_depth(self, pct: float = 50.0, threshold_m: float = 0.01) -> float:
        layer = self.layer(pct)
        wet = layer[np.isfinite(layer) & (layer > threshold_m)]
        return float(wet.mean()) if wet.size else 0.0

    def to_csv(self, path: str | Path) -> None:
        """Write wet cells only, as ``lng,lat,size,<percentile columns>``."""
        keep = np.nan_to_num(self.depth).max(axis=1) > 0.01
        cols = ",".join(f"depth_p{p:g}" for p in self.percentiles)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"lng,lat,size,{cols}\n")
            for i in np.flatnonzero(keep):
                values = ",".join(f"{v:.4f}" for v in self.depth[i])
                handle.write(
                    f"{self.emulator.lng[i]:.7f},{self.emulator.lat[i]:.8f},"
                    f"{self.emulator.size[i]:.8g},{values}\n"
                )

    def to_geojson(
        self, path: str | Path, pct: float = 50.0, min_depth: float = 0.05
    ) -> None:
        """Write cells deeper than ``min_depth`` as square polygons."""
        import json

        layer = np.nan_to_num(self.layer(pct))
        features = []
        for i in np.flatnonzero(layer > min_depth):
            half = self.emulator.size[i] / 2
            x, y = self.emulator.lng[i], self.emulator.lat[i]
            features.append(
                {
                    "type": "Feature",
                    "properties": {"depth_m": round(float(layer[i]), 3)},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [x - half, y - half],
                                [x + half, y - half],
                                [x + half, y + half],
                                [x - half, y + half],
                                [x - half, y - half],
                            ]
                        ],
                    },
                }
            )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"type": "FeatureCollection", "features": features}, handle)
