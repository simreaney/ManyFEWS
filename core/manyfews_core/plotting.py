"""
Matplotlib figures for the notebooks. Requires the ``[plot]`` extra.

Colour choices follow from what the data is doing, not from taste:

* River flow and depth are **continuous magnitudes of one quantity**, so they get
  a single blue hue varying in lightness. Nested uncertainty bands are discrete
  ordered marks, so their steps are held to the ordinal floor - the palest band
  still clears 2:1 against the chart surface rather than dissolving into it.
* The ``minQ`` flooding threshold is a **state**, not a series, so it wears the
  reserved critical status colour and always ships with a text label. Colour
  never carries that meaning alone.
* :func:`forecast_panel` puts rainfall and flow on **two stacked axes sharing an
  x-axis**, never a secondary y-axis. Two measures on two y-scales in one frame
  is the single most misleading thing a chart can do - the crossing point is an
  artefact of the scaling, and here it would corrupt exactly the rain-to-flow
  lead-lag the reader is trying to see.

These figures commit to a light surface: they render as static images in
notebooks, where the surrounding page is white.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "flow_fan",
    "rainfall_bars",
    "forecast_panel",
    "storm_response_curve",
    "depth_vs_flow",
    "depth_histogram",
    "apply_style",
]

# Ordinal steps from the blue sequential ramp; validated light->dark with visible
# gaps and a light end that clears the surface.
BAND_OUTER = "#86b6ef"  # p10-p90
BAND_INNER = "#2a78d6"  # p30-p50
LINE_MEDIAN = "#0d366b"
TRACE = "#5598e7"  # individual parameter-set traces
RAIN = "#2a78d6"

# Reserved status colour - state, never a series.
CRITICAL = "#d03b3b"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Lowest minQ anywhere in the emulator: below this, nothing floods at all.
FLOOD_THRESHOLD_M3S = 50.0


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "plotting needs matplotlib: pip install 'manyfews-core[plot]'"
        ) from exc
    return plt


def apply_style() -> None:
    """Apply the recessive chart chrome. Call once per notebook."""
    plt = _plt()
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "legend.frameon": False,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )
    for spine in ("top", "right"):
        plt.rcParams[f"axes.spines.{spine}"] = False


def _times(times) -> np.ndarray:
    return np.asarray(times, dtype="datetime64[s]").astype("datetime64[m]").tolist()


def flow_fan(
    times,
    flows: np.ndarray,
    threshold: float | None = FLOOD_THRESHOLD_M3S,
    ax=None,
    title: str = "River flow forecast",
    show_traces: bool = False,
):
    """
    Percentile fan of the pooled flow ensemble.

    :param flows: ``(n_steps, n_samples)`` pooled across members and parameter
        sets, or an :class:`~manyfews_core.riverflow.EnsembleFlows`-shaped
        ``(n_members, n_steps, n_sets)`` array.
    :param threshold: draws the flooding-onset line; ``None`` omits it.
    :param show_traces: overlay individual parameter-set trajectories.
    """
    plt = _plt()
    ax = ax or plt.subplots(figsize=(9, 4.2))[1]

    flows = np.asarray(flows, dtype=np.float64)
    if flows.ndim == 3:
        flows = flows.transpose(1, 0, 2).reshape(flows.shape[1], -1)
    x = _times(times)

    p10, p30, p50, p90 = np.percentile(flows, [10, 30, 50, 90], axis=1)

    if show_traces:
        for j in range(flows.shape[1]):
            ax.plot(x, flows[:, j], color=TRACE, lw=0.5, alpha=0.12, zorder=1)

    ax.fill_between(
        x, p10, p90, color=BAND_OUTER, lw=0, zorder=2, label="10th–90th percentile"
    )
    ax.fill_between(
        x, p30, p50, color=BAND_INNER, lw=0, zorder=3, label="30th–50th percentile"
    )
    ax.plot(x, p50, color=LINE_MEDIAN, lw=2, zorder=4, label="Median")

    if threshold is not None:
        ax.axhline(threshold, color=CRITICAL, lw=1.5, ls=(0, (5, 3)), zorder=5)
        ax.annotate(
            f"flooding starts — {threshold:g} m³/s",
            xy=(0.995, threshold),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=CRITICAL,
            weight="600",
        )

    ax.set_title(title)
    ax.set_ylabel("River flow (m³/s)")
    ax.margins(x=0.01)
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK_SECONDARY)
    ax.figure.autofmt_xdate()
    return ax


def rainfall_bars(times, rainfall_mm_day: np.ndarray, ax=None, storm_span=None):
    """
    Rainfall as bars on an inverted axis - the hydrograph convention, rain
    falling from the top of the frame onto the flow below it.
    """
    plt = _plt()
    ax = ax or plt.subplots(figsize=(9, 2))[1]

    x = _times(times)
    rain_mm = np.asarray(rainfall_mm_day, dtype=np.float64) * 0.25  # back to mm/bucket
    ax.bar(x, rain_mm, width=0.22, color=RAIN, lw=0)

    if storm_span is not None:
        ax.axvspan(*storm_span, color=CRITICAL, alpha=0.08, lw=0, zorder=0)

    ax.invert_yaxis()
    ax.set_ylabel("Rainfall (mm / 6 h)")
    ax.margins(x=0.01)
    return ax


def forecast_panel(ens, threshold: float | None = FLOOD_THRESHOLD_M3S, storm=None):
    """
    The operational summary: rainfall above, flow fan below, one shared x-axis.

    Deliberately two axes rather than one with a secondary y-scale.
    """
    plt = _plt()
    fig, (ax_rain, ax_flow) = plt.subplots(
        2,
        1,
        figsize=(9.5, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2.6], "hspace": 0.08},
    )

    rainfall_bars(ens.times, ens.median_rainfall(), ax=ax_rain)
    ax_rain.set_title("Median forecast rainfall across ensemble members")

    flow_fan(ens.times, ens.flow_m3s, threshold=threshold, ax=ax_flow, title="")
    ax_flow.set_title(
        "River flow — 100 parameter sets × %d weather member(s)" % len(ens.members)
    )

    if storm is not None and getattr(storm, "enabled", False):
        fig.text(
            0.5,
            0.005,
            f"Synthetic {storm.total_mm:g} mm storm injected "
            f"{storm.days_ahead} day(s) ahead — not a real forecast",
            ha="center",
            fontsize=9,
            color=CRITICAL,
            weight="600",
        )
    return fig


def storm_response_curve(
    totals, peak_p50, peak_p90, threshold: float = FLOOD_THRESHOLD_M3S
):
    """
    Peak flow against storm size, with the flooding threshold marked.

    Two series here, so both are direct-labelled as well as legended - identity
    never rests on colour alone.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    totals = np.asarray(totals, dtype=np.float64)
    series = (
        (np.asarray(peak_p90), BAND_INNER, "90th percentile"),
        (np.asarray(peak_p50), LINE_MEDIAN, "Median"),
    )
    for values, colour, label in series:
        ax.plot(totals, values, color=colour, lw=2, marker="o", ms=6, label=label)

    # Leave room on the right for the direct labels, then place them inside it.
    span = totals[-1] - totals[0]
    ax.set_xlim(totals[0] - 0.06 * span, totals[-1] + 0.30 * span)
    for values, _, label in series:
        ax.annotate(
            label,
            xy=(totals[-1], values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax.axhline(threshold, color=CRITICAL, lw=1.5, ls=(0, (5, 3)))
    # Sits on top of the dashed line, so it needs the surface behind it to stay
    # legible where a series crosses.
    ax.annotate(
        f"flooding starts — {threshold:g} m³/s",
        xy=(totals[0], threshold),
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=9,
        color=CRITICAL,
        weight="600",
        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"),
    )

    ax.set_title("Peak river flow against synthetic storm size")
    ax.set_xlabel("Storm total (mm in one day)")
    ax.set_ylabel("Peak river flow (m³/s)")
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK_SECONDARY)
    return fig


def depth_vs_flow(emulator, n_cells: int = 200, q_max: float = 600.0, seed: int = 0):
    """
    Sampled cells' depth response, showing why the input needs clamping.

    Every curve is the same quantity in a different place, so they are one hue at
    low opacity rather than distinct colours. The region beyond the emulator's
    calibration range is shaded and labelled: that is where the fitted cubics
    stop meaning anything.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    rng = np.random.default_rng(seed)
    cells = rng.choice(
        emulator.n_cells, size=min(n_cells, emulator.n_cells), replace=False
    )
    q = np.linspace(0, q_max, 400)

    # Deliberately bypass the clamp so the divergence is visible.
    b0, b1, b2, b3, mq = (
        emulator.beta0[cells],
        emulator.beta1[cells],
        emulator.beta2[cells],
        emulator.beta3[cells],
        emulator.min_q[cells],
    )
    depth = b0[:, None] + q * (b1[:, None] + q * (b2[:, None] + q * b3[:, None]))
    depth = np.where(q[None, :] < mq[:, None], 0.0, np.maximum(depth, 0.0))

    for row in depth:
        ax.plot(q, row, color=TRACE, lw=0.8, alpha=0.18)

    cap = emulator.q_cap
    ax.axvspan(cap, q_max, color=INK_MUTED, alpha=0.10, lw=0, zorder=0)
    ax.annotate(
        "beyond calibration —\ncubics diverge, so the\ninput is clamped here",
        xy=((cap + q_max) / 2, ax.get_ylim()[1] * 0.72),
        ha="center",
        va="top",
        fontsize=9,
        color=INK_SECONDARY,
        weight="600",
    )
    ax.axvline(cap, color=CRITICAL, lw=1.5, ls=(0, (5, 3)))

    ax.set_title(f"Depth response of {len(cells)} sampled cells")
    ax.set_xlabel("River flow (m³/s)")
    ax.set_ylabel("Flood depth (m)")
    ax.margins(x=0)
    return fig


def depth_histogram(field, pct: float = 50.0, ax=None, threshold_m: float = 0.01):
    """Distribution of depths across wet cells at one percentile."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(7.5, 3.6))[1]

    layer = field.layer(pct)
    wet = layer[np.isfinite(layer) & (layer > threshold_m)]

    if wet.size == 0:
        ax.text(
            0.5,
            0.5,
            "No cells flooded at this flow",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=INK_SECONDARY,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        return ax

    ax.hist(wet, bins=60, color=BAND_INNER, lw=0)
    ax.set_title(f"Flood depth distribution — {wet.size:,} wet cells (p{pct:g})")
    ax.set_xlabel("Flood depth (m)")
    ax.set_ylabel("Cells")
    return ax
