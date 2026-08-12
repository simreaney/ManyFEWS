"""
Verbatim numeric port of ``manyfews/calculations/generate_river_flows.py``
lines 20-350: ``ModelFun``, ``RoutingFun``, ``PDMmodel`` and ``FAO56``.

DO NOT REFACTOR. The CamelCase names, the argument orders and several genuine
oddities are preserved exactly so that a side-by-side diff against the Django
module stays reviewable, and so that the MATLAB reference outputs in
``Data/*_Benchmark.csv`` keep passing. Parity is asserted by
``tests/test_hydrology_benchmark.py``.

Removed relative to the original, none of which affect the numerics:

* ``from django.conf import settings`` and ``from .models import ...`` - neither
  is referenced by any of these four functions.
* The celery task logger, replaced by a stdlib logger.
* Five dead ``try: <name> / except NameError:`` blocks. In each case the name is
  a function parameter, so the bare lookup always succeeds and the ``except``
  body is unreachable: ``q0`` (RoutingFun), ``S0`` (PDMmodel), and ``T``, ``u2``
  and the ``ea`` fallback (FAO56).

A sixth block that *looks* identical is not. See the note above ``Rs`` in
``FAO56``.

Science references:

* Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). FAO Irrigation and
  Drainage Paper No. 56. FAO.
* Moore, R. J. (2007). The PDM rainfall-runoff model. Hydrology and Earth System
  Sciences, 11(1), 483-499.
* Mathias, S. A., McIntyre, N., & Oughton, R. H. (2016). A study of non-linearity
  in rainfall-runoff response using 120 UK catchments. Journal of Hydrology, 540,
  423-436.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ModelFun", "RoutingFun", "PDMmodel", "FAO56"]


def ModelFun(
    qp: np.ndarray,
    Ep: np.ndarray,
    dt: float,
    CatArea: float,
    X: np.ndarray,
    F0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Determine river flow from weather data by running PDM plus two routing stores
    once per parameter set.

    :param qp: rainfall (mm/day), shape (n_steps,)
    :param Ep: potential evapotranspiration (mm/day), shape (n_steps,)
    :param dt: time step (day)
    :param CatArea: catchment area (km2)
    :param X: model parameters, shape (n_sets, 4) - Smax, qmax, k, Tr
    :param F0: initial state, shape (n_sets, 3) - storage, slow flow, fast flow

    :return: ``(Q, F0)`` where Q is river flow (m3/s) with shape
        (n_steps, n_sets), and F0 is the end-of-run state.

    .. warning::
       ``F0`` is **mutated in place** as well as returned. This matches the
       original and callers rely on it, but it means every caller must pass a
       copy if it intends to reuse the starting state - which is exactly what
       running several ensemble members from one spin-up requires.
    """
    Q = np.zeros(((np.size(Ep[:])), (np.size(X[:, 1]))))  # initialize the matrix Q

    for n in range(np.size(X[:, 1])):
        # Extract parameters
        Smax = X[n, 0]  # (mm)
        qmax = X[n, 1]  # (mm/day)
        k = X[n, 2]  # (mm/day)
        Tr = X[n, 3]  # (days)

        # Extract initial conditions
        S0 = F0[n, 0]  # initial storage level for PDM (mm)
        qSLOW0 = F0[n, 1]  # initial slow flow rate (mm/day)
        qFAST0 = F0[n, 2]  # initial fast flow rate (mm/day)

        # Determine surface runoff and drainage.
        # UPSTREAM QUIRK, PRESERVED: gamma is pinned to 1 despite PDMmodel's
        # Pareto docstring, which makes the store's fraction-saturated term
        # F = S / Smax linear rather than Pareto-distributed.
        qro, qd, Ea, S = PDMmodel(qp, Ep, Smax, 1, k, dt, S0)

        # Determine slow flow (linear store, X is residence time in days)
        qSLOW = RoutingFun(qd, Tr, 1, dt, qSLOW0)

        # Determine fast flow (non-linear store, X is qmax in mm/day)
        qFAST = RoutingFun(qro, qmax, 5 / 3, dt, qFAST0)

        # Determine river flow
        q = qFAST + qSLOW

        # Convert mm/day over the catchment to m3/s
        Q[:, n] = (q * CatArea * (1e3) / 24) / (3600)

        # Update initial condition vector with final values of state variables
        F0[n, :] = [(S[-1]), (qSLOW[-1]), (qFAST[-1])]

    return Q, F0


def RoutingFun(
    qs: np.ndarray, X: float, b: float, dt: float, q0: float
) -> np.ndarray:
    """
    Route runoff through a non-linear store, ``q = a * v**b``, after
    Mathias et al. (2016).

    :param qs: inflow rate (mm/day)
    :param X: residence time in days when ``b == 1``, else qmax in mm/day
    :param b: exponent; 1 gives a linear store, 5/3 the non-linear fast store
    :param dt: time step (day)
    :param q0: initial river flow (mm/day)

    :return: river flow (mm/day), same length as ``qs``
    """
    numPoint = np.size(qs)  # Determine number of data points

    if b == 1:
        # This means it's a linear store
        # so X is the residence time in days
        Tr = X
        a = 1 / Tr
        vmax = float("inf")

    else:
        # This means it's a non-linear store
        # so X is qmax in mm/day
        qmax = X
        dtDAY = 1  # this is needed because qmax is determined with daily data
        a = (math.pow(qmax, (1 - b))) * (math.pow((b * dtDAY), (-b)))
        vmax = math.pow((a * b * dt), (1 / (1 - b)))  # Limit on v for stability

    # UPSTREAM QUIRK, PRESERVED: q is filled with the constant q0 and v derived
    # from all n+1 entries, but only v[0] survives the loop below. Kept because
    # it also sets how vmax interacts with the very first step.
    q = np.full((numPoint + 1), q0)
    v = np.power((q / a), (1 / b))

    for i in range(numPoint):  # Step through each time step
        # Trial values for q and v:
        qtrial = a * math.pow(v[i], b)
        vtrial = v[i] + ((qs[i] - qtrial) * dt)

        if vtrial < vmax:  # Ordinarily use trial values
            q[i] = qtrial  # River flow (mm/day)
            v[i + 1] = vtrial  # River storage (mm)
        else:  # Force v<=vmax
            q[i] = qs[i] - ((vmax - v[i]) / dt)
            v[i + 1] = vmax

    q = q[:-1]

    return q


def PDMmodel(
    qp: np.ndarray,
    Ep: np.ndarray,
    Smax: float,
    gamma: float,
    k: float,
    dt: float,
    S0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Probability Distributed Model store with a Pareto distribution of store
    capacity, after Moore (2007).

    :param qp: rainfall (mm/day)
    :param Ep: potential evapotranspiration (mm/day)
    :param Smax: maximum storage (mm)
    :param gamma: exponent for the Pareto distribution
    :param k: drainage parameter (mm/day)
    :param dt: time step (day)
    :param S0: initial storage (mm)

    :return: ``(qro, qd, Ea, S)`` - surface runoff, drainage, actual
        evapotranspiration (all mm/day) and catchment storage (mm)
    """
    numPoint = np.size(qp)

    # Initialise vectors
    qd = np.zeros(numPoint)
    qro = np.zeros(numPoint)
    Ea = np.zeros(numPoint)

    S = np.full((numPoint + 1), S0)

    for i in range(numPoint):
        # Pareto CDF
        F = 1 - (math.pow((1 - (S[i]) / Smax), gamma))

        # Determine drainage rate
        qd[i] = k * S[i] / Smax

        # Trial value for S
        Strial = S[i] + (((1 - F) * qp[i] - Ep[i] - qd[i]) * dt)

        # To start with try the following:
        S[i + 1] = Strial  # Catchment storage
        qro[i] = F * qp[i]  # River flow contribution
        Ea[i] = Ep[i]  # Actual evapotranspiration

        if Strial <= 0:
            S[i + 1] = 0
            qd[i] = 0
            Ea[i] = ((1 - F) * qp[i]) + (S[i] / dt)
        elif Strial >= Smax:  # Force S<=Smax
            S[i + 1] = Smax
            qro[i] = qp[i] - Ep[i] - ((Smax - S[i]) / dt) - qd[i]

    S = S[:-1]

    return qro, qd, Ea, S


def FAO56(
    dt: float,
    predictionDate: datetime,
    Tmin: np.ndarray,
    Tmax: np.ndarray,
    alt: float,
    lat: float,
    T: np.ndarray,
    u2: np.ndarray,
    RH: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    FAO56 Penman-Monteith reference evapotranspiration, with open-water
    evaporation as a by-product.

    :param dt: time step (day); see the note on day-of-year below
    :param predictionDate: start of the series, used for day-of-year
    :param Tmin: daily minimum air temperature (deg C)
    :param Tmax: daily maximum air temperature (deg C)
    :param alt: catchment mean altitude (m above sea level)
    :param lat: catchment mean latitude (degrees)
    :param T: air temperature (deg C)
    :param u2: wind speed at 2 m (m/s)
    :param RH: relative humidity (%)

    :return: ``(ETo, E0)`` - reference evapotranspiration and open-water
        evaporation, both mm/day
    """
    # UPSTREAM QUIRK, PRESERVED: the second line reads the Tmax that the first
    # line just reassigned, so `np.minimum(Tmax, Tmin)` is Tmin and the whole
    # statement is a no-op. Fixing it would change the numerics and break parity.
    Tmax = np.maximum(Tmax, Tmin)
    Tmin = np.minimum(Tmax, Tmin)

    # u2 (m/s), P (kPa), ea (kPa), Rn (MJ/m2/day)

    # Slope of saturation curve (Del) from Eq. 13
    Del = (4098 * (0.6108 * (np.exp(((17.27 * T) / (T + 237.3)))))) / np.square(
        T + 237.3
    )

    # Atmospheric pressure (P) from Eq. 7
    P = 101.3 * (math.pow(((293 - 0.0065 * alt) / 293), 5.26))

    # Psychrometric constant (gam) from Eq. 8
    cp = 1.013e-3
    lam = 2.45
    eps = 0.622
    gam = ((cp * P) / eps) / lam

    # Saturation vapour pressure (eo) at Tmax and Tmin from Eq. 11
    eoTmax = 0.6108 * (np.exp((17.27 * Tmax) / (Tmax + 237.3)))
    eoTmin = 0.6108 * (np.exp((17.27 * Tmin) / (Tmin + 237.3)))

    # Assume saturation vapour pressure is mean
    es = (eoTmax + eoTmin) / 2

    # Actual vapour pressure from relative humidity. The original wrapped this in
    # a try/except NameError whose fallback (dewpoint from Tmin) was unreachable,
    # because RH is a parameter.
    ea = (RH / 100) * es

    # Convert latitude from degrees to radians from Eq. 22
    varphi = (lat * math.pi) / 180

    # Determine day of the year as a number from 1 to 365.
    # NOTE: `np.size(Tmax) / 4` assumes four buckets per day, so this only yields
    # one J per input row when dt == 0.25. CatchmentConfig enforces that.
    beginDate = predictionDate.date()
    beginDateNum = (beginDate - date(beginDate.year - 1, 12, 31)).days
    J = beginDateNum + np.arange(0, ((np.size(Tmax[:])) / 4), dt)

    # Inverse relative distance Earth-Sun from Eq. 23
    dr = 1 + (0.033 * np.cos(((2 * math.pi) / 365) * J))

    # Solar declination from Eq. 24
    delta = 0.409 * np.sin((((2 * math.pi) / 365) * J) - 1.39)

    # Sunset hour angle from Eq. 25
    ws = np.arccos((-math.tan(varphi)) * (np.tan(delta)))

    # Extraterrestrial radiation from Eq. 21
    Gsc = 0.0820
    Ra = (
        (((24 * 60) / (math.pi)) * Gsc)
        * dr
        * (
            ws * (math.sin(varphi)) * (np.sin(delta))
            + (math.cos(varphi)) * (np.cos(delta)) * np.sin(ws)
        )
    )

    # Incoming solar radiation from Eq. 50 (Hargreaves radiation formula).
    #
    # CAREFUL. In the original this sat inside `try: Rs / except NameError:`,
    # which looks like the five dead blocks removed elsewhere in this module but
    # is the exact opposite: `Rs` is not a parameter and is never bound earlier,
    # so the bare lookup ALWAYS raises and the except body is the LIVE path.
    # Deleting this line "as dead code" makes Rso/RsRso below raise NameError.
    # tests/test_fao56.py::test_rs_assignment_is_live guards it.
    kRS = 0.16
    Rs = kRS * (np.sqrt(Tmax - Tmin)) * Ra

    # Clear-sky solar radiation from Eq. 37
    Rso = ((0.75 + (2e-5) * alt)) * Ra

    # Net solar radiation from Eq. 38
    alpha = 0.23
    Rns = (1 - alpha) * Rs

    # Outgoing net longwave radiation from Eq. 39
    sig = 4.903e-9
    sigTmax4 = sig * (np.power((Tmax + 273.15), 4))
    sigTmin4 = sig * (np.power((Tmin + 273.15), 4))
    sigT4 = (sigTmax4 + sigTmin4) / 2
    RsRso = Rs / Rso
    RsRso[RsRso > 1] = 1
    Rnl = sigT4 * (0.34 - (0.14 * np.sqrt(ea))) * (1.35 * RsRso - 0.35)

    # For the UK you would instead want:
    # Rnl = sigT4 * (0.56 - (0.25 * np.sqrt(ea))) * (1.35 * RsRso - 0.35)

    # Net radiation from Eq. 40
    Rn = Rns - Rnl

    # Assume zero soil heat flux
    G = 0

    # Calculate reference evapotranspiration from Eq. 6
    T1 = 0.408 * Del * (Rn - G)
    T2 = ((gam * 900) / (T + 273)) * u2 * (es - ea)
    T3 = Del + (gam * (1 + 0.34 * u2))
    ETo = (T1 + T2) / T3

    # Determine open water evaporation. T2 is deliberately not recomputed - it
    # has no albedo term - while T1 picks up the new net radiation and T3 drops
    # the surface resistance (rs = 0).
    alpha = 0.05  # Albedo for wet bare soil (p. 43)
    Rns = (1 - alpha) * Rs
    Rn = Rns - Rnl
    T1 = 0.408 * Del * (Rn - G)
    T3 = Del + gam
    E0 = (T1 + T2) / T3

    return ETo, E0
