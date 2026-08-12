"""River-channel geometry, including the file's malformed JSON."""

import json

import numpy as np
import pytest

from manyfews_core.channel import (
    cached_channel_mask,
    channel_mask,
    load_channel_polygons,
)
from manyfews_core.data import CHANNEL_GEOJSON, data_path

N_PARTS = 869
N_MASKED_CELLS = 20_959


def test_file_is_not_valid_json():
    """
    ``Data/channel.geojson`` has stray characters after the closing brace, so the
    standard parser cannot read it. GEOS tolerated this, which is why the Django
    app never noticed. Documented here so the tolerant loader is not "simplified"
    back to ``json.load``.
    """
    with pytest.raises(json.JSONDecodeError):
        json.loads(data_path(CHANNEL_GEOJSON).read_text())


def test_loader_tolerates_the_trailing_junk():
    parts = load_channel_polygons()
    assert len(parts) == N_PARTS


def test_some_parts_have_holes():
    """A bounding-box containment test would be wrong for these."""
    parts = load_channel_polygons()
    assert sum(1 for p in parts if len(p) > 1) > 0


def test_mask_covers_expected_cell_count(emulator):
    mask = cached_channel_mask(emulator)
    assert mask.dtype == bool
    assert int(mask.sum()) == N_MASKED_CELLS


def test_masked_cells_are_the_deepest(emulator):
    """
    Sanity check on what the mask is for: the channel is where the emulator
    reports the deepest "flooding", because that is simply the river.
    """
    from manyfews_core.scenarios import constant_flow_samples

    mask = cached_channel_mask(emulator)
    depth = emulator.depth_percentiles(constant_flow_samples(200.0, 50))[:, 2]
    assert depth[mask].mean() > depth[~mask].mean()


def test_ray_cast_against_a_known_square():
    """Point-in-polygon on a unit square, including a hole."""
    square = [
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]),
        np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6], [0.4, 0.4]]),
    ]
    lng = np.array([0.5, 0.2, 1.5, 0.5])
    lat = np.array([0.2, 0.5, 0.5, 0.5])  # inside, inside, outside, in the hole

    mask = channel_mask(lng, lat, [square])
    assert mask.tolist() == [True, True, False, False]
