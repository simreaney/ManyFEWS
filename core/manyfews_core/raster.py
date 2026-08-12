"""
Turning the scattered emulator cells into a north-up raster for display.

The emulator's cells form a regular lattice that is **rotated 0.3387 degrees**
from north - it was almost certainly built in a projected CRS and reprojected to
WGS-84. Over the 3.4 km domain that rotation amounts to about 20 m, or ten cell
widths, so reconstructing an axis-aligned image from lattice indices would put
the flood in visibly the wrong place.

The fix is simply to bin the actual longitudes and latitudes into an axis-aligned
grid, which is what :func:`rasterise` does. The result is 1682 x 770 at the
native 1.81e-5 degree resolution, about 23.5% occupied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DepthRaster", "rasterise", "DEPTH_RAMP"]

# Single-hue sequential ramp. Depth is a continuous magnitude, so it gets one hue
# varying in lightness rather than a rainbow: lightness ordering is the only
# channel that survives greyscale printing and the common colour deficiencies.
DEPTH_RAMP = (
    (0xCD, 0xE2, 0xFB),
    (0x9E, 0xC5, 0xF4),
    (0x55, 0x98, 0xE7),
    (0x2A, 0x78, 0xD6),
    (0x1C, 0x5C, 0xAB),
    (0x10, 0x42, 0x81),
)


@dataclass
class DepthRaster:
    """A north-up depth grid. ``NaN`` marks positions with no emulator cell."""

    values: np.ndarray  # (n_row, n_col) float32, row 0 is the northernmost
    bounds: tuple[float, float, float, float]  # (lat_min, lng_min, lat_max, lng_max)
    cell_size_deg: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def leaflet_bounds(self) -> list[list[float]]:
        """``[[south, west], [north, east]]`` as Leaflet wants it."""
        lat_min, lng_min, lat_max, lng_max = self.bounds
        return [[lat_min, lng_min], [lat_max, lng_max]]

    def to_rgba(self, vmax: float = 3.0, gamma: float = 0.7) -> np.ndarray:
        """
        Render to ``(h, w, 4)`` uint8.

        Dry ground and absent cells get **alpha 0**, not the palest ramp colour,
        so the basemap shows through instead of a uniform haze over the whole
        rectangle. ``gamma`` below 1 lifts shallow water, which is where the
        interesting detail is.
        """
        values = self.values
        wet = np.isfinite(values) & (values > 0.01)

        norm = np.zeros(values.shape, dtype=np.float64)
        np.divide(np.nan_to_num(values), vmax, out=norm, where=wet)
        np.clip(norm, 0.0, 1.0, out=norm)
        norm **= gamma

        ramp = np.array(DEPTH_RAMP, dtype=np.float64)
        position = norm * (len(ramp) - 1)
        lower = np.clip(np.floor(position).astype(int), 0, len(ramp) - 1)
        upper = np.clip(lower + 1, 0, len(ramp) - 1)
        weight = (position - lower)[..., None]

        rgb = ramp[lower] * (1 - weight) + ramp[upper] * weight

        rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = rgb.astype(np.uint8)
        rgba[..., 3] = np.where(wet, 255, 0).astype(np.uint8)
        return rgba

    def to_png_bytes(self, vmax: float = 3.0, gamma: float = 0.7) -> bytes:
        """PNG encoding of :meth:`to_rgba`. Requires Pillow."""
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "to_png_bytes needs Pillow: pip install 'manyfews-core[map]'"
            ) from exc
        import io

        buffer = io.BytesIO()
        Image.fromarray(self.to_rgba(vmax, gamma), mode="RGBA").save(
            buffer, format="PNG"
        )
        return buffer.getvalue()


def rasterise(
    emulator,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> DepthRaster:
    """
    Bin per-cell values onto an axis-aligned north-up grid.

    :param values: ``(n_cells,)``, typically ``DepthField.layer(50)``
    :param mask: optional boolean of cells to exclude (e.g. the river channel)

    Where two source cells land in the same output pixel - about 421 of them do,
    because the rotated lattice does not map perfectly onto an axis-aligned one -
    the deeper value wins. Under-reporting a flood depth would be the worse
    error.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (emulator.n_cells,):
        raise ValueError(
            f"values must have shape ({emulator.n_cells},), got {values.shape}"
        )

    keep = np.isfinite(values)
    if mask is not None:
        keep &= ~mask

    size = emulator.cell_size
    lat_min, lng_min, lat_max, lng_max = emulator.bounds

    n_col = int(round((lng_max - lng_min) / size)) + 1
    n_row = int(round((lat_max - lat_min) / size)) + 1

    col = np.round((emulator.lng - lng_min) / size).astype(np.int64)
    row = np.round((lat_max - emulator.lat) / size).astype(np.int64)  # row 0 = north
    np.clip(col, 0, n_col - 1, out=col)
    np.clip(row, 0, n_row - 1, out=row)

    grid = np.full((n_row, n_col), np.nan, dtype=np.float32)
    idx = np.flatnonzero(keep)
    # Seed occupied positions at zero so dry-but-present cells stay distinct from
    # positions with no cell at all, then take the deepest value per position.
    grid[row[idx], col[idx]] = 0.0
    np.maximum.at(grid, (row[idx], col[idx]), values[idx].astype(np.float32))

    logger.info(
        "Rasterised %d cells onto a %d x %d grid (%.1f%% occupied)",
        idx.size,
        n_row,
        n_col,
        100 * np.isfinite(grid).mean(),
    )
    return DepthRaster(
        values=grid,
        bounds=(
            lat_min - size / 2,
            lng_min - size / 2,
            lat_max + size / 2,
            lng_max + size / 2,
        ),
        cell_size_deg=size,
    )
