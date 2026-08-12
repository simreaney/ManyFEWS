"""
manyfews_core - the ManyFEWS flood forecasting pipeline without Django.

Everything the science needs, on numpy and requests alone: no PostGIS, no GDAL,
no GeoDjango, no celery, no numba. The full chain is

    Open-Meteo ensemble forecast
      -> 6-hour buckets            (weather)
      -> FAO56 evapotranspiration  (hydrology)
      -> PDM soil store            (hydrology)
      -> two routing stores        (hydrology)
      -> river flow ensemble       (riverflow)
      -> per-cell depth cubics     (emulator)
      -> raster / map              (raster, mapping)

A minimal forecast::

    from manyfews_core import (
        fetch_history, fetch_forecast, spin_up, run_ensemble,
        load_parameters, FloodEmulator, rasterise,
    )

    params  = load_parameters()
    state   = spin_up(fetch_history(), params)
    ens     = run_ensemble(fetch_forecast(), state, params)

    emulator = FloodEmulator.from_csv()
    field    = emulator.field(ens.pooled(ens.peak_step()))
    raster   = rasterise(emulator, field.layer(50))

``plotting`` and ``mapping`` are optional extras and are deliberately not
imported here, so the package stays importable without matplotlib or folium.
"""

__version__ = "0.1.0"

from .channel import cached_channel_mask, channel_mask, load_channel_polygons
from .config import (
    DEFAULT_CATCHMENT,
    DEFAULT_INITIAL_STATE,
    CatchmentConfig,
    EmulatorConfig,
    ForecastConfig,
    RiskConfig,
    StormConfig,
)
from .data import data_dir, data_path
from .emulator import DepthField, FloodEmulator
from .hydrology import FAO56, ModelFun, PDMmodel, RoutingFun
from .raster import DepthRaster, rasterise
from .risk import risk_fraction, wet_cell_count
from .riverflow import (
    EnsembleFlows,
    RiverFlowResult,
    default_initial_state,
    generate_river_flows,
    load_parameters,
    run_ensemble,
    spin_up,
)
from .scenarios import (
    constant_flow_samples,
    inject_storm,
    inject_storm_ensemble,
    scale_precip,
)
from .weather import (
    OpenMeteoClient,
    WeatherSeries,
    fetch_forecast,
    fetch_history,
    offset_time,
)

__all__ = [
    "__version__",
    # config
    "CatchmentConfig",
    "ForecastConfig",
    "StormConfig",
    "EmulatorConfig",
    "RiskConfig",
    "DEFAULT_CATCHMENT",
    "DEFAULT_INITIAL_STATE",
    # data
    "data_dir",
    "data_path",
    # weather
    "WeatherSeries",
    "OpenMeteoClient",
    "fetch_forecast",
    "fetch_history",
    "offset_time",
    # hydrology
    "FAO56",
    "PDMmodel",
    "RoutingFun",
    "ModelFun",
    # river flow
    "RiverFlowResult",
    "EnsembleFlows",
    "load_parameters",
    "default_initial_state",
    "generate_river_flows",
    "spin_up",
    "run_ensemble",
    # scenarios
    "inject_storm",
    "inject_storm_ensemble",
    "scale_precip",
    "constant_flow_samples",
    # emulator
    "FloodEmulator",
    "DepthField",
    # geometry / output
    "load_channel_polygons",
    "channel_mask",
    "cached_channel_mask",
    "DepthRaster",
    "rasterise",
    "wet_cell_count",
    "risk_fraction",
]
