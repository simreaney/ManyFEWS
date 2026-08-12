# manyfews-core

The ManyFEWS flood forecasting pipeline without Django.

The science in the main application is about 700 lines of numpy wrapped in
PostGIS, GeoDjango, GDAL, celery, RabbitMQ and numba. This package is that
science on its own, so it can run in a Colab notebook, in CI, or in a build
script, with two dependencies:

```
numpy, requests
```

`tests/test_no_heavy_deps.py` enforces that — it imports the package in a
subprocess and fails if django, celery, numba, GDAL, shapely or psycopg2 appear
in `sys.modules`.

## The notebooks

Two Google Colab notebooks in [`notebooks/`](notebooks/), runnable with no setup:

| Notebook | What it is |
|---|---|
| `01_walkthrough.ipynb` | Guided tour of the whole chain, plotting each stage: weather → evapotranspiration → soil store → routing → flow ensemble → flood depth → map. Includes a synthetic storm and a flow slider. |
| `02_operational_forecast.ipynb` | Runtime → Run all. Parameters at the top, forecast and map at the bottom. |

Both clone this repository (~25 MB of model data) and run in about 90 seconds.

> **Expect an empty flood map.** Typical flows in this catchment are 10–30 m³/s
> and nothing floods below 50, so an ordinary forecast produces no inundation at
> all. That is the correct answer, and both notebooks say so explicitly. Use the
> storm scenario or the flow slider to see the flood model work.

## Quick start

```python
import manyfews_core as mf

params = mf.load_parameters()
state  = mf.spin_up(mf.fetch_history(), params)      # ~29 days of reanalysis
ens    = mf.run_ensemble(mf.fetch_forecast(), state, params)

emulator = mf.FloodEmulator.from_csv()
field    = emulator.field(ens.pooled(ens.peak_step()))
raster   = mf.rasterise(emulator, field.layer(50))

print(f"{field.wet_cells(90):,} cells flooded, deepest {field.max_depth(90):.2f} m")
```

Install with the extras you want:

```bash
pip install -e './core[all]'      # + matplotlib, folium, pytest
pytest -q                          # 74 tests, no network, no database
```

## Layout

| Module | Role |
|---|---|
| `config.py` | Frozen dataclasses. Every magic number lands here. |
| `hydrology.py` | FAO56, PDM, routing — **verbatim numerics, do not refactor** |
| `riverflow.py` | Spin-up, single runs, ensemble runs |
| `weather.py` | Open-Meteo ingestion and the `(N, 6)` array contract |
| `scenarios.py` | Synthetic storms, direct flow override |
| `emulator.py` | Per-cell depth cubics and the percentile shortcut |
| `channel.py` | River-channel masking without GEOS |
| `raster.py` / `risk.py` | Display grid, RGBA, headline risk |
| `plotting.py` / `mapping.py` | Optional extras: matplotlib, folium |

## Things that will surprise you

Documented at each site, and pinned by tests, because they are all load-bearing:

- **`hydrology.py` preserves upstream quirks deliberately.** The temperature
  swap-guard in FAO56 is a no-op; the PDM's `gamma` is pinned to 1 despite its
  Pareto docstring; `ModelFun` mutates its state argument. Changing any of these
  breaks parity with the MATLAB reference in `Data/*_Benchmark.csv`.
- **One `try/except NameError` in FAO56 is live, not dead.** Five others in the
  same file are unreachable and were removed; the `Rs` one is the opposite case
  and deleting it makes the function raise. See the comment there.
- **`Data/channel.geojson` is not valid JSON.** Five stray characters after the
  closing brace. GEOS tolerated it; `json.load` does not.
- **The emulator's cubics diverge above ~300 m³/s** — 118 m at Q=500, 1122 m at
  Q=800. This package clamps the input. The Django application does not.
- **`timestep_days` must be 0.25.** Four buckets per day is baked into two
  reshapes. `CatchmentConfig` raises rather than letting it corrupt silently.

## Relationship to the Django app

Additive. Nothing in `manyfews/` is modified, so there is no regression risk to
the running system. Refactoring `calculations/` to import from here would remove
a duplicate copy of the numerics and is a sensible follow-up, but is out of scope.

The same package backs the static site — see
[`docs/STATIC_SITE_FEASIBILITY.md`](../docs/STATIC_SITE_FEASIBILITY.md).
