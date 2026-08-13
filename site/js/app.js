/**
 * Orchestration for the static ManyFEWS site.
 *
 * Fetches live weather, runs the catchment model and the flood emulator, and
 * renders a flow chart and a depth map. No server is involved at any point.
 */

import { FloodEmulator } from './emulator.js';
import { generateRiverFlows } from './hydrology.js';
import { fetchForecast, fetchHistory, injectStorm } from './weather.js';
import { getLocale, initLocale, setLocale, t } from './i18n.js';

const DEPTH_RAMP = [
  [0xcd, 0xe2, 0xfb], [0x9e, 0xc5, 0xf4], [0x55, 0x98, 0xe7],
  [0x2a, 0x78, 0xd6], [0x1c, 0x5c, 0xab], [0x10, 0x42, 0x81],
];
const MAX_DEPTH_M = 3.0;
const STORM_MM = 200.0;
const STORM_DAYS_AHEAD = 2;
const MODEL_SIZE_MB = 3.6;

const state = {
  emulator: null,
  meta: null,
  params: null,
  forecast: null,
  history: null,
  ensemble: null,      // { times, pooledPerStep: Float64Array[] (sorted) }
  mode: 'forecast',
  percentile: 90,
  flow: 160,
  storm: false,
  map: null,
  overlay: null,
  canvas: null,
  limitBox: null,
  userMarker: null,
  userAccuracyCircle: null,
  ready: false,
  status: null,        // { key, args, kind, busy } — last status, re-translated on language switch
};

const $ = (id) => document.getElementById(id);

// -------------------------------------------------------------------- i18n

/** Apply the active locale to every element carrying a data-i18n(-html) tag. */
function applyStaticTranslations() {
  document.title = t('docTitle');
  document.querySelector('meta[name="description"]')?.setAttribute('content', t('metaDescription'));
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-html]').forEach((el) => {
    el.innerHTML = t(el.dataset.i18nHtml);
  });
  document.querySelectorAll('[data-i18n-list]').forEach((el) => {
    el.innerHTML = t(el.dataset.i18nList).map((item) => `<li>${item}</li>`).join('');
  });
  $('lang-en').setAttribute('aria-pressed', String(getLocale() === 'en'));
  $('lang-id').setAttribute('aria-pressed', String(getLocale() === 'id'));
  $('locate-btn').textContent = state.userMarker ? t('locateButtonHide') : t('locateButton');
  $('flow-readout').textContent = `${state.flow} ${t('flowUnit')}`;
}

/** Re-run everything whose text depends on live model output, not just the DOM. */
function applyDynamicTranslations() {
  if (state.userMarker) state.userMarker.bindPopup(t('youAreHere'));
  if (state.limitBox) state.limitBox.setTooltipContent(t('limitTooltip'));
  if (state.status) setStatus(t(state.status.key, ...state.status.args), state.status.kind, state.status.busy);
  if (!state.ready) return;
  $('chart-sub').textContent = t('chartSub', state.ensemble.nMembers);
  render();
}

function wireLanguageSwitch() {
  const choose = (locale) => {
    if (locale === getLocale()) return;
    setLocale(locale);
    applyStaticTranslations();
    applyDynamicTranslations();
  };
  $('lang-en').addEventListener('click', () => choose('en'));
  $('lang-id').addEventListener('click', () => choose('id'));
}

function setStatus(text, kind = 'info', busy = true) {
  const panel = $('status-panel');
  const notice = $('status');
  panel.hidden = false;
  notice.className = `notice ${kind === 'error' ? '' : 'info'}`.trim();
  notice.querySelector('.icon').innerHTML = busy
    ? '<span class="spinner"></span>'
    : (kind === 'error' ? '!' : 'i');
  $('status-text').textContent = text;
}

/** Like setStatus, but remembers the translation key so a language switch can redraw it. */
function setStatusKey(key, args = [], kind = 'info', busy = true) {
  state.status = { key, args, kind, busy };
  setStatus(t(key, ...args), kind, busy);
}

// ---------------------------------------------------------------- model run

/** Run every member from the same spun-up state and pool flows per step. */
function runEnsemble(members, spunUpState) {
  const { catchment } = state.meta;
  const perMember = members.map((weather) =>
    generateRiverFlows(weather, state.params.sets, spunUpState, catchment)
  );

  const nSteps = perMember[0].flows[0].length;
  const nSets = state.params.sets.length;
  const pooledPerStep = [];

  for (let step = 0; step < nSteps; step++) {
    const pooled = new Float64Array(perMember.length * nSets);
    let k = 0;
    for (const result of perMember) {
      for (let s = 0; s < nSets; s++) pooled[k++] = result.flows[s][step];
    }
    pooled.sort();
    pooledPerStep.push(pooled);
  }

  return {
    times: members[0].times,
    pooledPerStep,
    rainfall: perMember[0].rainfall,
    nMembers: members.length,
  };
}

/** Round an axis maximum up so its quarter-points are readable numbers. */
function niceCeiling(value) {
  const magnitude = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 1.2, 1.6, 2, 2.4, 3, 4, 5, 6, 8, 10]) {
    if (value <= step * magnitude) return step * magnitude;
  }
  return 10 * magnitude;
}

function percentileOf(sorted, pct) {
  const pos = ((sorted.length - 1) * pct) / 100;
  const k = Math.floor(pos);
  const frac = pos - k;
  return sorted[k] + frac * (sorted[Math.min(k + 1, sorted.length - 1)] - sorted[k]);
}

function peakStep(ensemble, pct = 90) {
  let best = 0;
  let bestValue = -Infinity;
  ensemble.pooledPerStep.forEach((pooled, i) => {
    const value = percentileOf(pooled, pct);
    if (value > bestValue) { bestValue = value; best = i; }
  });
  return best;
}

async function recomputeForecast() {
  setStatusKey('statusRunning');
  await nextFrame();

  let members = state.forecast;
  if (state.storm) {
    members = members.map((m) => injectStorm(m, STORM_MM, STORM_DAYS_AHEAD));
  }
  state.ensemble = runEnsemble(members, state.spunUpState);
  $('chart-sub').textContent = t('chartSub', state.ensemble.nMembers);
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

// ------------------------------------------------------------------- chart

function drawChart() {
  const svg = $('chart');
  const { times, pooledPerStep } = state.ensemble;
  const w = svg.clientWidth || 900;
  const h = svg.clientHeight || 300;
  const pad = { top: 12, right: 16, bottom: 30, left: 52 };
  const iw = w - pad.left - pad.right;
  const ih = h - pad.top - pad.bottom;

  const threshold = state.meta.flood_threshold_m3s;
  const bands = pooledPerStep.map((p) => ({
    p10: percentileOf(p, 10), p30: percentileOf(p, 30),
    p50: percentileOf(p, 50), p90: percentileOf(p, 90),
  }));

  const yMax = niceCeiling(Math.max(threshold * 1.25, ...bands.map((b) => b.p90)) * 1.08);
  const x = (i) => pad.left + (i / (times.length - 1)) * iw;
  const y = (v) => pad.top + ih - (v / yMax) * ih;

  const area = (lo, hi) =>
    bands.map((b, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(b[hi]).toFixed(1)}`).join('') +
    bands.map((b, i) => `L${x(bands.length - 1 - i).toFixed(1)},${y(bands[bands.length - 1 - i][lo]).toFixed(1)}`).join('') +
    'Z';
  const line = (key) =>
    bands.map((b, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(b[key]).toFixed(1)}`).join('');

  // Day ticks.
  const ticks = [];
  times.forEach((time, i) => {
    if (time.getUTCHours() === 0 && i % 8 === 0) {
      ticks.push({ i, label: `${time.getUTCDate()}/${time.getUTCMonth() + 1}` });
    }
  });
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => yMax * f);

  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.innerHTML = `
    <g>
      ${yTicks.map((v) => `
        <line x1="${pad.left}" x2="${w - pad.right}" y1="${y(v)}" y2="${y(v)}"
              stroke="var(--grid)" stroke-width="1"/>
        <text x="${pad.left - 8}" y="${y(v) + 4}" text-anchor="end"
              font-size="11" fill="var(--ink-muted)">${v % 1 ? v.toFixed(1) : v}</text>`).join('')}
      ${ticks.map((tk) => `
        <text x="${x(tk.i)}" y="${h - 10}" text-anchor="middle"
              font-size="11" fill="var(--ink-muted)">${tk.label}</text>`).join('')}
      <text x="12" y="${pad.top + ih / 2}" transform="rotate(-90 12 ${pad.top + ih / 2})"
            text-anchor="middle" font-size="11.5" fill="var(--ink-2)">${t('chartAxisLabel')}</text>

      <path d="${area('p10', 'p90')}" fill="var(--band-outer)"/>
      <path d="${area('p30', 'p50')}" fill="var(--band-inner)"/>
      <path d="${line('p50')}" fill="none" stroke="var(--line)" stroke-width="2"/>

      <line x1="${pad.left}" x2="${w - pad.right}" y1="${y(threshold)}" y2="${y(threshold)}"
            stroke="var(--critical)" stroke-width="1.5" stroke-dasharray="5 3"/>
      <text x="${w - pad.right}" y="${y(threshold) - 6}" text-anchor="end"
            font-size="11.5" font-weight="600" fill="var(--critical)">
        ${t('chartFloodStart', threshold)}
      </text>

      <g font-size="11.5" fill="var(--ink-2)">
        <rect x="${pad.left + 8}" y="${pad.top + 6}" width="11" height="11" rx="2" fill="var(--band-outer)"/>
        <text x="${pad.left + 25}" y="${pad.top + 15}">${t('chartLegendP10P90')}</text>
        <rect x="${pad.left + 8}" y="${pad.top + 23}" width="11" height="11" rx="2" fill="var(--band-inner)"/>
        <text x="${pad.left + 25}" y="${pad.top + 32}">${t('chartLegendP30P50')}</text>
        <line x1="${pad.left + 8}" x2="${pad.left + 19}" y1="${pad.top + 45}" y2="${pad.top + 45}"
              stroke="var(--line)" stroke-width="2"/>
        <text x="${pad.left + 25}" y="${pad.top + 49}">${t('chartLegendMedian')}</text>
      </g>
    </g>`;
}

// --------------------------------------------------------------------- map

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/**
 * Build the map. The container must already be visible: Leaflet reads its size
 * on construction, and a hidden element is zero by zero, which makes fitBounds
 * settle on zoom 0 and show the whole world.
 */
function initMap() {
  const g = state.meta.grid;
  const bounds = [[g.lat_min, g.lng_min], [g.lat_max, g.lng_max]];

  state.map = L.map('map', { zoomControl: true });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
  }).addTo(state.map);

  state.canvas = document.createElement('canvas');
  state.canvas.width = g.n_col;
  state.canvas.height = g.n_row;
  state.overlay = L.imageOverlay('', bounds, { opacity: 0.78, interactive: false }).addTo(state.map);

  // Outline the extent the emulator actually has predictions for — the
  // basemap keeps panning past it, but depth is only ever computed inside.
  state.limitBox = L.rectangle(bounds, {
    color: cssVar('--ink-2') || '#52514e',
    weight: 1.5,
    dashArray: '6 4',
    fill: false,
    interactive: true,
  }).addTo(state.map).bindTooltip(t('limitTooltip'));

  state.map.invalidateSize();
  state.map.fitBounds(bounds);
}

/** Paint a depth field into the overlay canvas. */
function drawDepth(depth) {
  const g = state.meta.grid;
  const { position, channel, n } = state.emulator;
  const ctx = state.canvas.getContext('2d');
  const image = ctx.createImageData(g.n_col, g.n_row);
  const px = image.data;

  const last = DEPTH_RAMP.length - 1;
  for (let i = 0; i < n; i++) {
    if (channel[i]) continue;
    const d = depth[i];
    if (!(d > 0.01)) continue;

    // Gamma below 1 lifts shallow water, which is where the detail is.
    const ramp = Math.pow(Math.min(d / MAX_DEPTH_M, 1), 0.7) * last;
    const lo = Math.min(Math.floor(ramp), last);
    const hi = Math.min(lo + 1, last);
    const f = ramp - lo;

    const o = position[i] * 4;
    px[o] = DEPTH_RAMP[lo][0] + f * (DEPTH_RAMP[hi][0] - DEPTH_RAMP[lo][0]);
    px[o + 1] = DEPTH_RAMP[lo][1] + f * (DEPTH_RAMP[hi][1] - DEPTH_RAMP[lo][1]);
    px[o + 2] = DEPTH_RAMP[lo][2] + f * (DEPTH_RAMP[hi][2] - DEPTH_RAMP[lo][2]);
    px[o + 3] = 255;   // dry ground stays fully transparent, not palest blue
  }

  ctx.putImageData(image, 0, 0);
  state.overlay.setUrl(state.canvas.toDataURL('image/png'));
}

// ---------------------------------------------------------- geolocation

function geoErrorMessage(error) {
  switch (error.code) {
    case error.PERMISSION_DENIED: return t('locateDenied');
    case error.POSITION_UNAVAILABLE: return t('locateUnavailable');
    case error.TIMEOUT: return t('locateTimeout');
    default: return t('locateUnavailable');
  }
}

function removeUserMarker() {
  if (state.userMarker) { state.map.removeLayer(state.userMarker); state.userMarker = null; }
  if (state.userAccuracyCircle) { state.map.removeLayer(state.userAccuracyCircle); state.userAccuracyCircle = null; }
  $('locate-btn').setAttribute('aria-pressed', 'false');
  $('locate-btn').textContent = t('locateButton');
}

function placeUserMarker(lat, lng, accuracy) {
  removeUserMarker();
  state.userMarker = L.marker([lat, lng], {
    icon: L.divIcon({
      className: 'you-are-here-icon',
      html: '<span class="you-are-here-dot"></span>',
      iconSize: [14, 14],
    }),
  }).addTo(state.map).bindPopup(t('youAreHere'));
  if (accuracy) {
    state.userAccuracyCircle = L.circle([lat, lng], {
      radius: accuracy, color: cssVar('--accent') || '#2a78d6', weight: 1, fillOpacity: 0.08,
    }).addTo(state.map);
  }
  state.userMarker.openPopup();
  state.map.panTo([lat, lng]);
  $('locate-btn').setAttribute('aria-pressed', 'true');
  $('locate-btn').textContent = t('locateButtonHide');
}

function wireLocate() {
  $('locate-btn').addEventListener('click', () => {
    if (state.userMarker) {
      removeUserMarker();
      $('locate-status').textContent = '';
      return;
    }
    if (!navigator.geolocation) {
      $('locate-status').textContent = t('locateUnsupported');
      return;
    }
    $('locate-status').textContent = t('locateLocating');
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        $('locate-status').textContent = '';
        placeUserMarker(pos.coords.latitude, pos.coords.longitude, pos.coords.accuracy);
      },
      (error) => { $('locate-status').textContent = geoErrorMessage(error); },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 }
    );
  });
}

// ------------------------------------------------------------------ render

function renderStats(depth, headline) {
  const em = state.emulator;
  const wet = em.wetCells(depth);
  const cells = [
    [t('statSituation'), headline],
    [t('statFlooded'), wet.toLocaleString()],
    [t('statDeepest'), `${em.maxDepth(depth).toFixed(2)} m`],
    [t('statMeanWet'), `${em.meanWetDepth(depth).toFixed(2)} m`],
  ];
  $('stats').innerHTML = cells
    .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join('');
  return wet;
}

function renderVerdict(wet, peakFlow) {
  const threshold = state.meta.flood_threshold_m3s;
  $('verdict').innerHTML = wet === 0
    ? `<div class="notice info"><span class="icon">i</span><span>
         <strong>${t('noFloodTitle')}</strong>
         ${t('noFloodBody', peakFlow.toFixed(1), state.percentile, threshold)}</span></div>`
    : `<div class="notice"><span class="icon">!</span><span>
         <strong>${t('floodTitle')}</strong>
         ${t('floodBody', wet.toLocaleString(), state.percentile, peakFlow.toFixed(1))}
         ${state.storm ? t('floodStormNote') : ''}
       </span></div>`;
}

function renderSummaryTable() {
  const { times, pooledPerStep } = state.ensemble;
  const threshold = state.meta.flood_threshold_m3s;
  const rows = [];

  for (let day = 0; day * 4 < pooledPerStep.length; day++) {
    let p50 = 0, p90 = 0;
    for (let j = 0; j < 4 && day * 4 + j < pooledPerStep.length; j++) {
      const pooled = pooledPerStep[day * 4 + j];
      p50 = Math.max(p50, percentileOf(pooled, 50));
      p90 = Math.max(p90, percentileOf(pooled, 90));
    }
    const dayStart = times[day * 4];
    rows.push({
      date: dayStart.toISOString().slice(0, 10),
      p50, p90, flooding: p90 >= threshold,
    });
  }

  $('summary').innerHTML = `
    <thead><tr><th>${t('thDate')}</th><th>${t('thP50')}</th><th>${t('thP90')}</th><th>${t('thStatus')}</th></tr></thead>
    <tbody>${rows.map((r) => `
      <tr class="${r.flooding ? 'flooding' : ''}">
        <td>${r.date}</td><td>${r.p50.toFixed(1)}</td><td>${r.p90.toFixed(1)}</td>
        <td>${r.flooding ? t('rowFlooding') : t('rowNoFlood')}</td>
      </tr>`).join('')}</tbody>`;
}

function render() {
  const em = state.emulator;

  if (state.mode === 'explore') {
    const depth = em.depthAt(state.flow);
    drawDepth(depth);
    renderStats(depth, `${state.flow} ${t('flowUnit')}`);
    $('verdict').innerHTML = '';
    $('map-sub').textContent = t('mapSubExplore', state.flow);
    return;
  }

  const step = peakStep(state.ensemble, state.percentile);
  const pooled = state.ensemble.pooledPerStep[step];
  const peakFlow = percentileOf(pooled, state.percentile);
  const depth = em.depthPercentile(pooled, state.percentile);

  drawDepth(depth);
  const wet = renderStats(depth, wetHeadline(wetCount(em, depth)));
  renderVerdict(wet, peakFlow);
  renderSummaryTable();
  drawChart();

  const dateStr = state.ensemble.times[step].toISOString().slice(0, 16).replace('T', ' ');
  $('map-sub').textContent = t('mapSubForecast', dateStr, state.percentile);
}

const wetCount = (em, depth) => em.wetCells(depth);
const wetHeadline = (wet) => (wet === 0 ? t('headlineNoFlooding') : t('headlineFlooding'));

// ------------------------------------------------------------------- setup

function wireControls() {
  const setMode = (mode, initial = false) => {
    state.mode = mode;
    $('mode-forecast').setAttribute('aria-pressed', String(mode === 'forecast'));
    $('mode-explore').setAttribute('aria-pressed', String(mode === 'explore'));
    $('flow-control').hidden = mode !== 'explore';
    $('flow-readout-wrap').hidden = mode !== 'explore';
    $('pct-control').hidden = mode !== 'forecast';
    $('storm-control').hidden = mode !== 'forecast';
    $('chart-panel').hidden = mode !== 'forecast';
    $('table-panel').hidden = mode !== 'forecast';
    if (!initial) render();
  };

  $('mode-forecast').addEventListener('click', () => setMode('forecast'));
  $('mode-explore').addEventListener('click', () => setMode('explore'));

  // Apply the starting mode so the flow slider is hidden until it is relevant.
  setMode(state.mode, true);

  document.querySelectorAll('#pct-control button').forEach((button) => {
    button.addEventListener('click', () => {
      state.percentile = Number(button.dataset.pct);
      document.querySelectorAll('#pct-control button').forEach((b) =>
        b.setAttribute('aria-pressed', String(b === button)));
      render();
    });
  });

  const slider = $('flow-slider');
  slider.addEventListener('input', () => {
    state.flow = Number(slider.value);
    $('flow-readout').textContent = `${state.flow} ${t('flowUnit')}`;
    render();
  });

  $('storm-toggle').addEventListener('change', async (event) => {
    state.storm = event.target.checked;
    await recomputeForecast();
    render();
    setStatusKey(state.storm ? 'stormOn' : 'stormOff',
      state.storm ? [STORM_MM, STORM_DAYS_AHEAD] : [], 'info', false);
  });

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.mode === 'forecast') drawChart(); }, 150);
  });
}

async function main() {
  initLocale();
  applyStaticTranslations();
  wireLanguageSwitch();

  try {
    const t0 = performance.now();

    setStatusKey('statusLoadingModelSized', [MODEL_SIZE_MB]);
    state.emulator = await FloodEmulator.load('data');
    state.meta = state.emulator.meta;
    state.params = await (await fetch('data/params.json')).json();

    setStatusKey('statusFetchingHistory');
    state.history = await fetchHistory(state.meta.catchment);

    setStatusKey('statusSpinningUp');
    const seed = state.params.sets.map(() => [...state.params.initial_state]);
    state.spunUpState = generateRiverFlows(
      state.history, state.params.sets, seed, state.meta.catchment
    ).state;

    setStatusKey('statusFetchingForecast');
    state.forecast = await fetchForecast(state.meta.catchment);

    await recomputeForecast();

    // Reveal the panels *before* building the map - Leaflet measures its
    // container on construction, and a hidden one measures zero.
    $('controls-panel').hidden = false;
    $('chart-panel').hidden = false;
    $('map-panel').hidden = false;
    $('table-panel').hidden = false;
    await nextFrame();

    initMap();
    wireControls();
    wireLocate();
    render();
    state.ready = true;

    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    setStatusKey('statusReady', [
      elapsed, state.emulator.n.toLocaleString(), state.ensemble.nMembers, state.params.sets.length,
    ], 'info', false);
  } catch (error) {
    console.error(error);
    setStatusKey('statusError', [error.message], 'error', false);
  }
}

main();
