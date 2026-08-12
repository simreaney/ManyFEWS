#!/usr/bin/env python3
"""
Pack the model data for the static site.

The browser needs three things that never change: the emulator's per-cell
polynomial coefficients, the river-channel mask, and the 100 rainfall-runoff
parameter sets. Everything else - the forecast itself - it fetches live from
Open-Meteo, which permits cross-origin requests.

Run this only when ``Data/`` changes, not on a schedule::

    python core/scripts/build_static_data.py

Output (into ``site/data/``):

``grid.bin.gz``
    The emulator. Per cell: a uint32 position on the axis-aligned display grid,
    four float32 coefficients, and a uint8 index into the nine distinct minQ
    values. Arrays are stored column-wise so each lands contiguously in a typed
    array on the other side. About 6.4 MB raw, 3.8 MB gzipped.

``channel.bin.gz``
    One bit per cell, set where the cell centre falls inside the river channel.

``meta.json``
    Grid geometry, the minQ lookup table, and the catchment configuration.

``params.json``
    The 100 PDM parameter sets and the seed model state.

Explicitly pre-compressed rather than relying on the host to negotiate it:
GitHub Pages will not gzip ``application/octet-stream``, and 6.4 MB versus 3.8 MB
is worth a ``DecompressionStream`` call.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from manyfews_core.channel import cached_channel_mask  # noqa: E402
from manyfews_core.config import (  # noqa: E402
    DEFAULT_CATCHMENT,
    DEFAULT_INITIAL_STATE,
    EmulatorConfig,
)
from manyfews_core.emulator import FloodEmulator  # noqa: E402
from manyfews_core.riverflow import load_parameters  # noqa: E402

FORMAT_VERSION = 1


def build(out_dir: Path, q_cap: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    emulator = FloodEmulator.from_csv(q_cap=q_cap)
    n = emulator.n_cells

    # --- axis-aligned display grid ----------------------------------------
    # The source lattice is rotated 0.34 degrees, which over this domain is
    # about 20 m - ten cell widths. Binning the real coordinates rather than
    # reconstructing lattice indices keeps the flood where it actually is.
    size = emulator.cell_size
    lat_min, lng_min, lat_max, lng_max = emulator.bounds
    n_col = int(round((lng_max - lng_min) / size)) + 1
    n_row = int(round((lat_max - lat_min) / size)) + 1

    col = np.clip(np.round((emulator.lng - lng_min) / size).astype(np.int64), 0, n_col - 1)
    row = np.clip(np.round((lat_max - emulator.lat) / size).astype(np.int64), 0, n_row - 1)
    position = (row * n_col + col).astype(np.uint32)

    collisions = n - np.unique(position).size
    print(f"grid {n_row} x {n_col} = {n_row * n_col:,} positions")
    print(f"cells {n:,} ({100 * n / (n_row * n_col):.1f}% fill), {collisions} collision(s)")

    # --- minQ as a small lookup -------------------------------------------
    levels = np.unique(emulator.min_q)
    if levels.size > 256:
        raise ValueError(f"minQ has {levels.size} distinct values; uint8 coding assumes <= 256")
    min_q_code = np.searchsorted(levels, emulator.min_q).astype(np.uint8)
    print(f"minQ levels: {[float(v) for v in levels]}")

    # --- pack --------------------------------------------------------------
    payload = b"".join(
        [
            position.astype("<u4").tobytes(),
            emulator.beta0.astype("<f4").tobytes(),
            emulator.beta1.astype("<f4").tobytes(),
            emulator.beta2.astype("<f4").tobytes(),
            emulator.beta3.astype("<f4").tobytes(),
            min_q_code.tobytes(),
        ]
    )
    _write_gz(out_dir / "grid.bin.gz", payload)

    mask = cached_channel_mask(emulator)
    _write_gz(out_dir / "channel.bin.gz", np.packbits(mask, bitorder="little").tobytes())
    print(f"channel mask: {int(mask.sum()):,} cells ({100 * mask.mean():.2f}%)")

    # Ship the monotonicity mask rather than recomputing it in the browser: the
    # dense-grid detector is ~240M evaluations, and the closed form is wrong
    # (see FloodEmulator.monotone). With this the browser can run the same exact
    # percentile shortcut the Python side does.
    monotone = emulator.monotone
    _write_gz(
        out_dir / "monotone.bin.gz", np.packbits(monotone, bitorder="little").tobytes()
    )
    print(f"monotone cells: {int(monotone.sum()):,} ({100 * monotone.mean():.2f}%)")

    # --- metadata ----------------------------------------------------------
    catchment = DEFAULT_CATCHMENT
    meta = {
        "format_version": FORMAT_VERSION,
        "n_cells": int(n),
        "grid": {
            "n_row": n_row,
            "n_col": n_col,
            "cell_size_deg": float(size),
            # Cell edges, matching DepthRaster.bounds, so Leaflet's overlay lines
            # up with the pixels rather than their centres.
            "lat_min": float(lat_min - size / 2),
            "lng_min": float(lng_min - size / 2),
            "lat_max": float(lat_max + size / 2),
            "lng_max": float(lng_max + size / 2),
        },
        "min_q_levels": [float(v) for v in levels],
        "q_cap_m3s": float(q_cap),
        "flood_threshold_m3s": float(levels.min()),
        "catchment": {
            "name": catchment.name,
            "latitude_deg": catchment.latitude_deg,
            "altitude_m": catchment.altitude_m,
            "area_km2": catchment.area_km2,
            "weather_lat": catchment.weather_lat,
            "weather_lon": catchment.weather_lon,
            "timestep_days": catchment.timestep_days,
        },
    }
    _write_json(out_dir / "meta.json", meta)

    params = load_parameters()
    _write_json(
        out_dir / "params.json",
        {
            "columns": ["Smax_mm", "qmax_mm_day", "k_mm_day", "Tr_days"],
            "sets": [[round(float(v), 6) for v in row] for row in params],
            "initial_state": list(DEFAULT_INITIAL_STATE),
        },
    )

    total = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    print(f"\ntotal payload: {total / 1e6:.2f} MB")


def _write_gz(path: Path, payload: bytes) -> None:
    # mtime=0 keeps the output byte-stable, so rebuilding unchanged data does not
    # churn the repository.
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    path.write_bytes(compressed)
    print(f"{path.name:<18} {len(payload) / 1e6:6.2f} MB raw -> {len(compressed) / 1e6:5.2f} MB gzip")


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")
    print(f"{path.name:<18} {path.stat().st_size / 1e3:6.1f} kB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "site" / "data",
        help="output directory (default: site/data)",
    )
    parser.add_argument("--q-cap", type=float, default=EmulatorConfig().q_cap_m3s)
    args = parser.parse_args()
    build(args.out, args.q_cap)
