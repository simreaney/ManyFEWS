"""
Interactive maps for the notebooks. Requires the ``[map]`` extra.

Uses **folium** rather than ipyleaflet: folium is preinstalled in Colab and
renders straight from a cell's repr, whereas ipyleaflet needs
``output.enable_custom_widget_manager()`` and is the most common reason a shared
Colab notebook renders nothing for the next reader.

The depth overlay is a single ``ImageOverlay``, not thousands of rectangles. The
Django app draws one ``L.rectangle`` per cell and pre-aggregates into a
32/64/128/256 tile pyramid to keep that tractable; with the whole grid as one
image none of that machinery is needed.
"""

from __future__ import annotations

import numpy as np

from .raster import DEPTH_RAMP, DepthRaster

__all__ = ["flood_map", "depth_overlay", "add_colorbar"]

# Majalaya. Matches MAP_CENTER in the Django settings.
DEFAULT_CENTRE = (-7.050465729629079, 107.75813455787436)

_OSM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_OSM_ATTR = "&copy; OpenStreetMap contributors"
_ESRI = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_ESRI_ATTR = "Tiles &copy; Esri"


def _folium():
    try:
        import folium
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "mapping needs folium: pip install 'manyfews-core[map]'"
        ) from exc
    return folium


def depth_overlay(raster: DepthRaster, vmax: float = 3.0, opacity: float = 0.75):
    """Build the depth ``ImageOverlay`` for a raster."""
    folium = _folium()
    return folium.raster_layers.ImageOverlay(
        image=raster.to_rgba(vmax=vmax),
        bounds=raster.leaflet_bounds,
        opacity=opacity,
        name="Flood depth",
        interactive=False,
        cross_origin=False,
        zindex=1,
    )


def add_colorbar(m, vmax: float = 3.0) -> None:
    """
    Add a depth legend.

    Continuous magnitude, so this is a gradient with a few labelled ticks rather
    than a swatch per bin.
    """
    folium = _folium()
    stops = ", ".join(
        f"rgb({r},{g},{b}) {100 * i / (len(DEPTH_RAMP) - 1):.0f}%"
        for i, (r, g, b) in enumerate(DEPTH_RAMP)
    )
    ticks = "".join(f"<span>{v:.2g}</span>" for v in np.linspace(0, vmax, 4))
    html = f"""
    <div style="position: fixed; bottom: 22px; left: 12px; z-index: 9999;
                background: #fcfcfb; padding: 8px 10px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(11,11,11,0.18);
                font: 11px system-ui, -apple-system, sans-serif; color: #52514e;">
      <div style="color:#0b0b0b; font-weight:600; margin-bottom:4px;">
        Flood depth (m)
      </div>
      <div style="width: 168px; height: 10px; border-radius: 2px;
                  background: linear-gradient(to right, {stops});"></div>
      <div style="display:flex; justify-content:space-between; width:168px;
                  margin-top:2px; font-variant-numeric: tabular-nums;">{ticks}</div>
      <div style="margin-top:4px; color:#898781;">
        {vmax:g} m and deeper shown at the darkest step
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def flood_map(
    raster: DepthRaster,
    vmax: float = 3.0,
    zoom: int = 15,
    opacity: float = 0.75,
    centre: tuple[float, float] = DEFAULT_CENTRE,
    satellite: bool = True,
):
    """
    An interactive flood-depth map.

    Dry ground is fully transparent rather than the palest blue, so the basemap
    reads through everywhere the model says there is no water.
    """
    folium = _folium()

    m = folium.Map(
        location=list(centre), zoom_start=zoom, tiles=None, control_scale=True
    )
    folium.TileLayer(_OSM, attr=_OSM_ATTR, name="Street map").add_to(m)
    if satellite:
        folium.TileLayer(_ESRI, attr=_ESRI_ATTR, name="Satellite").add_to(m)

    depth_overlay(raster, vmax=vmax, opacity=opacity).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    add_colorbar(m, vmax=vmax)

    m.fit_bounds(raster.leaflet_bounds)
    return m
