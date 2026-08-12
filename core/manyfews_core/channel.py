"""
River-channel masking without GEOS.

The emulator happily reports metres of "flood" inside the river itself, which is
just the river. The Django app removes those cells with
``channel.channel_location.intersects(param.bounding_box)`` - a GEOS call per
cell per forecast time. Here the same job is a vectorised even-odd ray cast over
all 302,748 cell centroids at once, so the whole mask takes a few seconds and is
then cached.

Two properties of ``Data/channel.geojson`` are worth knowing before touching this:

* **It is not valid JSON.** There are five stray characters after the closing
  brace, so ``json.load`` raises ``Extra data: line 874``. GEOS tolerated it;
  Python's parser does not. :func:`load_channel_polygons` reads the first
  complete value and warns about the remainder.
* **It is one MultiPolygon with 869 parts**, and seven of those have interior
  rings. A bounding-box test would be badly wrong.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .data import CHANNEL_GEOJSON, cached_array, data_path

logger = logging.getLogger(__name__)

__all__ = ["load_channel_polygons", "channel_mask", "cached_channel_mask"]

# A part is a list of rings; a ring is an (n, 2) array of lng/lat vertices. The
# first ring is the exterior, any others are holes.
Part = list[np.ndarray]


def load_channel_polygons(path: str | Path | None = None) -> list[Part]:
    """Read the channel geometry, tolerating the trailing junk in the file."""
    path = Path(path) if path else data_path(CHANNEL_GEOJSON)
    text = path.read_text(encoding="utf-8").strip()

    obj, end = json.JSONDecoder().raw_decode(text)
    if end < len(text):
        logger.warning(
            "%s has %d trailing character(s) after the JSON value (%r); ignoring them. "
            "The file is malformed - json.load() cannot read it at all.",
            path.name,
            len(text) - end,
            text[end : end + 10],
        )

    geometry = obj.get("geometry", obj)
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if kind == "MultiPolygon":
        raw_parts = coords
    elif kind == "Polygon":
        raw_parts = [coords]
    else:
        raise ValueError(f"expected Polygon or MultiPolygon in {path.name}, got {kind}")

    parts = [
        [np.asarray(ring, dtype=np.float64) for ring in part] for part in raw_parts
    ]
    logger.info(
        "Loaded %d channel part(s), %d with holes",
        len(parts),
        sum(1 for p in parts if len(p) > 1),
    )
    return parts


def _points_in_ring(lng: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd ray cast of many points against one ring, vectorised."""
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]

    # Edges the horizontal ray from each point crosses in y.
    straddles = (y1[None, :] > lat[:, None]) != (y2[None, :] > lat[:, None])

    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_lat = (x2 - x1)[None, :] * (lat[:, None] - y1[None, :]) / (y2 - y1)[
            None, :
        ] + x1[None, :]

    crosses = straddles & (lng[:, None] < x_at_lat)
    return (crosses.sum(axis=1) % 2).astype(bool)


def channel_mask(
    lng: np.ndarray,
    lat: np.ndarray,
    polygons: list[Part],
    chunk: int = 50_000,
) -> np.ndarray:
    """
    Boolean mask of points whose centroid falls inside the channel.

    Each part is bbox-prefiltered before the ray cast, which is what keeps this
    tractable: most of the 869 parts are small rectangles covering a handful of
    the 302,748 cells.

    Uses centroid containment. Django's polygon-versus-square ``intersects``
    masks slightly more - every cell the channel merely clips - but centroid
    containment is the standard raster convention and avoids inflating the mask.
    """
    lng = np.asarray(lng, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    mask = np.zeros(lng.shape[0], dtype=bool)

    for part in polygons:
        exterior = part[0]
        x0, y0 = exterior[:, 0].min(), exterior[:, 1].min()
        x1, y1 = exterior[:, 0].max(), exterior[:, 1].max()

        candidates = np.flatnonzero(
            (lng >= x0) & (lng <= x1) & (lat >= y0) & (lat <= y1) & ~mask
        )
        if candidates.size == 0:
            continue

        for start in range(0, candidates.size, chunk):
            idx = candidates[start : start + chunk]
            inside = _points_in_ring(lng[idx], lat[idx], exterior)
            # XOR each hole back out.
            for hole in part[1:]:
                inside ^= inside & _points_in_ring(lng[idx], lat[idx], hole)
            mask[idx[inside]] = True

    logger.info(
        "Channel mask covers %d of %d cells (%.2f%%)",
        mask.sum(),
        mask.size,
        100 * mask.mean(),
    )
    return mask


def cached_channel_mask(
    emulator,
    path: str | Path | None = None,
    cache: bool = True,
) -> np.ndarray:
    """
    Channel mask for an emulator's cells, memoised next to the parameter CSV.

    The geometry never changes, so this is a genuine one-off cost.
    """
    polygons = load_channel_polygons(path)

    def build() -> np.ndarray:
        return channel_mask(emulator.lng, emulator.lat, polygons)

    if emulator.source is None:
        return build()
    return np.asarray(cached_array(emulator.source, "channel", build, cache)).astype(
        bool
    )
