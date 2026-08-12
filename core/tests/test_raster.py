"""Rasterising scattered cells onto a north-up grid."""

import numpy as np
import pytest

from manyfews_core.raster import rasterise
from manyfews_core.risk import risk_fraction, wet_cell_count
from manyfews_core.config import RiskConfig

EXPECTED_SHAPE = (1682, 770)


def test_grid_shape_and_bounds(emulator):
    raster = rasterise(emulator, np.zeros(emulator.n_cells))
    assert raster.shape == EXPECTED_SHAPE

    lat_min, lng_min, lat_max, lng_max = raster.bounds
    assert lat_min < lat_max and lng_min < lng_max
    # Bounds are cell edges, so they sit half a cell outside the centroids.
    assert lat_min == pytest.approx(emulator.lat.min() - emulator.cell_size / 2)


def test_north_is_row_zero(emulator):
    """Image convention: row 0 is the top of the picture, i.e. the highest latitude."""
    values = np.where(emulator.lat > emulator.lat.mean(), 1.0, 0.0)
    raster = rasterise(emulator, values)
    top = np.nanmean(raster.values[: EXPECTED_SHAPE[0] // 4])
    bottom = np.nanmean(raster.values[-EXPECTED_SHAPE[0] // 4 :])
    assert top > bottom


def test_absent_positions_are_nan(emulator):
    """The grid is only ~23% occupied; the rest must stay NaN, not zero."""
    raster = rasterise(emulator, np.ones(emulator.n_cells))
    assert np.isnan(raster.values).any()
    assert np.isfinite(raster.values).mean() < 0.30


def test_collisions_keep_the_deeper_value(emulator):
    """
    About 421 source cells collide when the rotated lattice is binned onto an
    axis-aligned grid. Under-reporting depth would be the worse failure, so the
    deeper value wins.
    """
    values = np.linspace(0.0, 5.0, emulator.n_cells)
    raster = rasterise(emulator, values)
    assert np.nanmax(raster.values) == pytest.approx(values.max(), abs=1e-5)


def test_mask_excludes_cells(emulator):
    mask = np.zeros(emulator.n_cells, dtype=bool)
    mask[:] = True
    raster = rasterise(emulator, np.ones(emulator.n_cells), mask=mask)
    assert not np.isfinite(raster.values).any()


def test_rgba_is_transparent_where_dry(emulator):
    """
    Dry ground must be fully transparent, not the palest ramp step - otherwise
    the whole rectangle sits over the basemap as a haze.
    """
    raster = rasterise(emulator, np.zeros(emulator.n_cells))
    rgba = raster.to_rgba(vmax=3.0)
    assert rgba.shape == EXPECTED_SHAPE + (4,)
    assert (rgba[..., 3] == 0).all()


def test_rgba_is_opaque_where_wet(emulator):
    raster = rasterise(emulator, np.full(emulator.n_cells, 1.5))
    rgba = raster.to_rgba(vmax=3.0)
    assert (rgba[..., 3] == 255).sum() > 200_000


def test_wet_cell_count_ignores_nan():
    values = np.array([0.0, 0.5, np.nan, 2.0])
    assert wet_cell_count(values) == 2


def test_legacy_risk_formula_does_not_subtract_the_baseline():
    """
    Pinning the upstream oddity: at exactly the channel cell count, the legacy
    formula reports ~7% risk rather than zero.
    """
    cfg = RiskConfig(legacy_formula=True)
    assert risk_fraction(cfg.channel_cell_count, cfg) == pytest.approx(0.0696, abs=1e-3)

    corrected = RiskConfig(legacy_formula=False)
    assert risk_fraction(corrected.channel_cell_count, corrected) == 0.0


def test_risk_is_clipped_to_unit_interval():
    assert risk_fraction(10**9) == 1.0
    assert risk_fraction(0) == 0.0
