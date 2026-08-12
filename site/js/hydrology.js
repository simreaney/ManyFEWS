/**
 * The ManyFEWS catchment model in JavaScript.
 *
 * A direct transcription of manyfews_core/hydrology.py, which is itself a
 * verbatim port of the Django implementation, which is itself a port of the
 * original MATLAB. The variable names are kept deliberately unidiomatic so the
 * three versions can be diffed against each other line by line.
 *
 * Parity with the MATLAB reference outputs in Data/*_Benchmark.csv is checked by
 * site/test/parity.test.mjs. If you change anything numeric here, run it.
 *
 * References:
 *   Allen et al. (1998), FAO Irrigation and Drainage Paper 56.
 *   Moore (2007), The PDM rainfall-runoff model, HESS 11(1), 483-499.
 *   Mathias et al. (2016), J. Hydrol. 540, 423-436.
 */

/** Surface roughness equivalent to the FAO56 reference crop. */
const Z0 = 0.006247;

/**
 * FAO56 Penman-Monteith reference evapotranspiration.
 *
 * @param {number} dt time step in days; must be 0.25 (see the day-of-year note)
 * @param {Date} predictionDate start of the series, for day-of-year
 * @param {Float64Array} Tmin daily minimum temperature, degrees C, per bucket
 * @param {Float64Array} Tmax daily maximum temperature, degrees C, per bucket
 * @param {number} alt catchment mean altitude, m
 * @param {number} lat catchment mean latitude, degrees
 * @param {Float64Array} T air temperature, degrees C
 * @param {Float64Array} u2 wind speed at 2 m, m/s
 * @param {Float64Array} RH relative humidity, %
 * @returns {{ETo: Float64Array, E0: Float64Array}} mm/day
 */
export function fao56(dt, predictionDate, Tmin, Tmax, alt, lat, T, u2, RH) {
  const n = Tmax.length;

  // UPSTREAM QUIRK, PRESERVED: the original reassigns Tmax and then derives
  // Tmin from the *new* value, so the intended swap-guard does nothing at all.
  // Reproduced here because parity depends on it.
  const tmax = new Float64Array(n);
  const tmin = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    tmax[i] = Math.max(Tmax[i], Tmin[i]);
    tmin[i] = Math.min(tmax[i], Tmin[i]);
  }

  // Atmospheric pressure, Eq. 7, and psychrometric constant, Eq. 8.
  const P = 101.3 * Math.pow((293 - 0.0065 * alt) / 293, 5.26);
  const gam = ((1.013e-3 * P) / 0.622) / 2.45;

  const varphi = (lat * Math.PI) / 180;
  const startOfYear = Date.UTC(predictionDate.getUTCFullYear() - 1, 11, 31);
  const dayOfYear = Math.round(
    (Date.UTC(
      predictionDate.getUTCFullYear(),
      predictionDate.getUTCMonth(),
      predictionDate.getUTCDate()
    ) - startOfYear) / 86400000
  );

  const ETo = new Float64Array(n);
  const E0 = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const t = T[i];

    // Slope of the saturation vapour pressure curve, Eq. 13.
    const Del = (4098 * (0.6108 * Math.exp((17.27 * t) / (t + 237.3)))) / ((t + 237.3) ** 2);

    // Saturation and actual vapour pressure, Eqs. 11 and 14.
    const eoTmax = 0.6108 * Math.exp((17.27 * tmax[i]) / (tmax[i] + 237.3));
    const eoTmin = 0.6108 * Math.exp((17.27 * tmin[i]) / (tmin[i] + 237.3));
    const es = (eoTmax + eoTmin) / 2;
    const ea = (RH[i] / 100) * es;

    // Day-of-year advances by dt per bucket - only correct at four per day.
    const J = dayOfYear + i * dt;

    const dr = 1 + 0.033 * Math.cos(((2 * Math.PI) / 365) * J);          // Eq. 23
    const delta = 0.409 * Math.sin(((2 * Math.PI) / 365) * J - 1.39);    // Eq. 24
    const ws = Math.acos(-Math.tan(varphi) * Math.tan(delta));           // Eq. 25

    // Extraterrestrial radiation, Eq. 21.
    const Ra =
      ((24 * 60) / Math.PI) * 0.082 * dr *
      (ws * Math.sin(varphi) * Math.sin(delta) +
        Math.cos(varphi) * Math.cos(delta) * Math.sin(ws));

    // Incoming solar radiation from the diurnal range, Eq. 50 (Hargreaves).
    const Rs = 0.16 * Math.sqrt(tmax[i] - tmin[i]) * Ra;
    const Rso = (0.75 + 2e-5 * alt) * Ra;                                // Eq. 37
    const RsRso = Math.min(Rs / Rso, 1);

    // Net longwave, Eq. 39.
    const sig = 4.903e-9;
    const sigT4 =
      (sig * (tmax[i] + 273.15) ** 4 + sig * (tmin[i] + 273.15) ** 4) / 2;
    const Rnl = sigT4 * (0.34 - 0.14 * Math.sqrt(ea)) * (1.35 * RsRso - 0.35);

    // Reference crop: albedo 0.23, Eqs. 38 and 6.
    const T2 = ((gam * 900) / (t + 273)) * u2[i] * (es - ea);
    let Rn = 0.77 * Rs - Rnl;
    ETo[i] = (0.408 * Del * Rn + T2) / (Del + gam * (1 + 0.34 * u2[i]));

    // Open water: albedo 0.05 and no surface resistance. T2 carries over
    // unchanged - it has no albedo term.
    Rn = 0.95 * Rs - Rnl;
    E0[i] = (0.408 * Del * Rn + T2) / (Del + gam);
  }

  return { ETo, E0 };
}

/**
 * Probability Distributed Model soil store (Moore 2007).
 *
 * @returns {{qro: Float64Array, qd: Float64Array, S: Float64Array}}
 *   surface runoff and drainage in mm/day, storage in mm
 */
export function pdmModel(qp, Ep, Smax, gamma, k, dt, S0) {
  const n = qp.length;
  const qd = new Float64Array(n);
  const qro = new Float64Array(n);
  const S = new Float64Array(n + 1);
  S[0] = S0;

  for (let i = 0; i < n; i++) {
    // With gamma pinned to 1 by the caller this is simply S/Smax.
    const F = 1 - Math.pow(1 - S[i] / Smax, gamma);
    qd[i] = (k * S[i]) / Smax;

    const Strial = S[i] + ((1 - F) * qp[i] - Ep[i] - qd[i]) * dt;
    S[i + 1] = Strial;
    qro[i] = F * qp[i];

    if (Strial <= 0) {
      S[i + 1] = 0;
      qd[i] = 0;
    } else if (Strial >= Smax) {
      S[i + 1] = Smax;
      qro[i] = qp[i] - Ep[i] - (Smax - S[i]) / dt - qd[i];
    }
  }

  return { qro, qd, S: S.subarray(0, n) };
}

/**
 * Non-linear routing store, q = a * v^b (Mathias et al. 2016).
 *
 * @param {number} X residence time in days when b === 1, else qmax in mm/day
 */
export function routingFun(qs, X, b, dt, q0) {
  const n = qs.length;
  let a;
  let vmax;

  if (b === 1) {
    a = 1 / X;                       // X is the residence time
    vmax = Infinity;
  } else {
    const dtDAY = 1;                 // qmax was determined from daily data
    a = Math.pow(X, 1 - b) * Math.pow(b * dtDAY, -b);
    vmax = Math.pow(a * b * dt, 1 / (1 - b));
  }

  const q = new Float64Array(n + 1).fill(q0);
  const v = new Float64Array(n + 1);
  v[0] = Math.pow(q0 / a, 1 / b);

  for (let i = 0; i < n; i++) {
    const qtrial = a * Math.pow(v[i], b);
    const vtrial = v[i] + (qs[i] - qtrial) * dt;

    if (vtrial < vmax) {
      q[i] = qtrial;
      v[i + 1] = vtrial;
    } else {
      q[i] = qs[i] - (vmax - v[i]) / dt;
      v[i + 1] = vmax;
    }
  }

  return q.subarray(0, n);
}

/**
 * Run every parameter set: PDM, then the slow and fast routing stores.
 *
 * @param {Float64Array} qp rainfall, mm/day
 * @param {Float64Array} Ep potential evapotranspiration, mm/day
 * @param {Array<Array<number>>} X parameter sets [Smax, qmax, k, Tr]
 * @param {Array<Float64Array>} F0 per-set state [storage, slow, fast];
 *   **mutated in place**, exactly as the Python and MATLAB versions do
 * @returns {Array<Float64Array>} flow in m3/s, one array per parameter set
 */
export function modelFun(qp, Ep, dt, catAreaKm2, X, F0) {
  const toCumecs = (catAreaKm2 * 1e3) / 24 / 3600;
  const flows = [];

  for (let s = 0; s < X.length; s++) {
    const [Smax, qmax, k, Tr] = X[s];
    const [S0, qSlow0, qFast0] = F0[s];

    // gamma is pinned to 1 upstream despite the Pareto docstring.
    const { qro, qd, S } = pdmModel(qp, Ep, Smax, 1, k, dt, S0);
    const qSlow = routingFun(qd, Tr, 1, dt, qSlow0);
    const qFast = routingFun(qro, qmax, 5 / 3, dt, qFast0);

    const q = new Float64Array(qp.length);
    for (let i = 0; i < qp.length; i++) q[i] = (qSlow[i] + qFast[i]) * toCumecs;
    flows.push(q);

    F0[s] = [S[S.length - 1], qSlow[qSlow.length - 1], qFast[qFast.length - 1]];
  }

  return flows;
}

/** Log-profile conversion of 10 m wind magnitude to 2 m. */
export function wind10mTo2m(u, v) {
  const u10 = Math.hypot(u, v);
  const uTau = u10 / 2.5 / Math.log(10 / Z0);
  return 2.5 * uTau * Math.log(2 / Z0);
}

/**
 * The whole catchment model for one weather realisation.
 *
 * @param {{times: Date[], rh, tmaxK, tminK, windU, windV, precip}} weather
 *   six-hourly, chronological, a whole number of days
 * @param {Array<Array<number>>} params 100 parameter sets
 * @param {Array<Array<number>>} state per-set initial state; not mutated
 * @param {{latitude_deg, altitude_m, area_km2, timestep_days}} catchment
 * @returns {{flows: Array<Float64Array>, state: Array, rainfall, pet}}
 */
export function generateRiverFlows(weather, params, state, catchment) {
  const dt = catchment.timestep_days;
  const n = weather.precip.length;
  if (n % 4 !== 0) {
    throw new Error(
      `weather has ${n} buckets, not a whole number of days; the daily ` +
        'temperature grouping assumes four 6-hour buckets per day'
    );
  }

  const tempMax = new Float64Array(n);
  const tempMin = new Float64Array(n);
  const T = new Float64Array(n);
  const u2 = new Float64Array(n);
  const qp = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    tempMax[i] = weather.tmaxK[i] - 273.15;
    tempMin[i] = weather.tminK[i] - 273.15;
    T[i] = (tempMin[i] + tempMax[i]) / 2;
    u2[i] = wind10mTo2m(weather.windU[i], weather.windV[i]);
    qp[i] = weather.precip[i] / dt;
  }

  // Collapse each day's four buckets to one min and one max, then broadcast back.
  const dailyMin = new Float64Array(n);
  const dailyMax = new Float64Array(n);
  for (let d = 0; d < n / 4; d++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let j = 0; j < 4; j++) {
      lo = Math.min(lo, tempMin[d * 4 + j]);
      hi = Math.max(hi, tempMax[d * 4 + j]);
    }
    for (let j = 0; j < 4; j++) {
      dailyMin[d * 4 + j] = lo;
      dailyMax[d * 4 + j] = hi;
    }
  }

  const { ETo } = fao56(
    dt, weather.times[0], dailyMin, dailyMax,
    catchment.altitude_m, catchment.latitude_deg, T, u2, weather.rh
  );

  // modelFun writes end state back into what it is given, so hand it a copy.
  const working = state.map((row) => [row[0], row[1], row[2]]);
  const flows = modelFun(qp, ETo, dt, catchment.area_km2, params, working);

  return { flows, state: working, rainfall: qp, pet: ETo };
}
