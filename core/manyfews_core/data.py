"""
Locating the shipped data files, and caching the expensive ones as ``.npy``.

The repository's ``Data/`` directory holds everything the pipeline needs:

===============================================  =========================================
``RainfallRunoffModelParameters.csv``            100 calibrated PDM parameter sets
``RainfallRunoffModelInitialConditions.csv``     reference initial state (benchmark input)
``floodEmulatorParams-20230921.csv``             302,748 emulator cells, 22.9 MB
``channel.geojson``                              river channel MultiPolygon
``{Q,qp,Eq,F0,t}_Benchmark.csv``                 MATLAB parity fixtures
===============================================  =========================================

Parsing the 22.9 MB emulator CSV takes several seconds, which is intolerable in a
notebook that reloads it on every re-run. ``cached_array`` transparently keeps a
``.npy`` sibling keyed on the source file's size and mtime.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "data_dir",
    "data_path",
    "cached_array",
    "PARAMETERS_CSV",
    "INITIAL_CONDITIONS_CSV",
    "EMULATOR_CSV",
    "CHANNEL_GEOJSON",
]

PARAMETERS_CSV = "RainfallRunoffModelParameters.csv"
INITIAL_CONDITIONS_CSV = "RainfallRunoffModelInitialConditions.csv"
EMULATOR_CSV = "floodEmulatorParams-20230921.csv"
CHANNEL_GEOJSON = "channel.geojson"

_ENV_VAR = "MANYFEWS_DATA_DIR"


def data_dir() -> Path:
    """
    Resolve the repository's ``Data/`` directory.

    Checks ``$MANYFEWS_DATA_DIR`` first, then walks up from this file looking for
    a ``Data`` directory containing the rainfall-runoff parameters. Walking up
    means the package works both from a source checkout and from a
    ``git clone`` inside a Colab session, without configuration.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        candidate = Path(override).expanduser().resolve()
        if not (candidate / PARAMETERS_CSV).is_file():
            raise FileNotFoundError(
                f"${_ENV_VAR} is set to {candidate}, but {PARAMETERS_CSV} is not there."
            )
        return candidate

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "Data"
        if (candidate / PARAMETERS_CSV).is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate the ManyFEWS Data/ directory. Set "
        f"${_ENV_VAR} to point at it, or run from inside a repository checkout."
    )


def data_path(name: str) -> Path:
    """Absolute path to a file in ``Data/``, raising if it is missing."""
    path = data_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found in {data_dir()}")
    return path


def _cache_stamp(source: Path) -> np.ndarray:
    stat = source.stat()
    return np.array([stat.st_size, int(stat.st_mtime)], dtype=np.int64)


def cached_array(
    source: Path,
    suffix: str,
    build: Callable[[], np.ndarray],
    enabled: bool = True,
) -> np.ndarray:
    """
    Return ``build()``, memoised to ``<source>.<suffix>.npy`` on disk.

    The cache is invalidated whenever the source file's size or mtime changes, so
    swapping in a new emulator parameter file is picked up automatically. A cache
    that cannot be read or written is a warning, never an error - a read-only
    checkout still works, just slowly.
    """
    if not enabled:
        return build()

    cache = source.with_suffix(source.suffix + f".{suffix}.npy")
    stamp_file = source.with_suffix(source.suffix + f".{suffix}.stamp.npy")
    want = _cache_stamp(source)

    if cache.is_file() and stamp_file.is_file():
        try:
            if np.array_equal(np.load(stamp_file), want):
                return np.load(cache, mmap_mode="r")
            logger.info("Cache for %s is stale, rebuilding", source.name)
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable cache %s: %s", cache.name, exc)

    array = build()
    try:
        np.save(cache, array)
        np.save(stamp_file, want)
    except OSError as exc:
        logger.warning("Could not write cache %s: %s", cache.name, exc)
    return array
