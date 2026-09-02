/**
 * Standalone TRMNL display: runs the same client-side forecast pipeline as
 * app.js (live weather -> hydrology model -> flood emulator), reduced to a
 * single output - a 5-day bar chart of max daily flood risk. No server, no
 * map, no controls; designed to be screenshotted by a TRMNL e-ink display.
 */

import { FloodEmulator } from './emulator.js';
import { generateRiverFlows } from './hydrology.js';
import { fetchForecast, fetchHistory } from './weather.js';

const DAYS = 5;
const DEPTH_THRESHOLD_M = 0.01;
const RISK_PERCENTILE = 50; // matches the Django backend's median-depth-based risk

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTH_NAMES = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

const $ = (id) => document.getElementById(id);

/** Run every ensemble member from the same spun-up state and pool flows per step. */
function runEnsemble(members, paramSets, spunUpState, catchment) {
  const perMember = members.map((weather) =>
    generateRiverFlows(weather, paramSets, spunUpState, catchment)
  );
  const nSteps = perMember[0].flows[0].length;
  const nSets = paramSets.length;
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
  return { times: members[0].times, pooledPerStep };
}

/**
 * Fraction of the floodable (non-channel) grid that is wet, at the median
 * flow for each 6-hourly step, maxed over each day.
 *
 * There is no canonical "risk %" to replicate here: the Django backend's
 * formula divides by an arbitrary, undocumented cell count that doesn't
 * match this grid (see docs/UPSTREAM_ISSUES.md). Risk is instead the share
 * of non-channel cells that are wet, which is self-calibrating to whatever
 * grid is loaded rather than relying on those broken constants.
 */
function dayRiskPercentages(emulator, ensemble) {
  let channelCells = 0;
  for (let i = 0; i < emulator.n; i++) channelCells += emulator.channel[i];
  const floodableCells = emulator.n - channelCells;

  const risks = [];
  for (let day = 0; day < DAYS; day++) {
    let maxRisk = 0;
    for (let j = 0; j < 4; j++) {
      const step = day * 4 + j;
      if (step >= ensemble.pooledPerStep.length) break;
      const depth = emulator.depthPercentile(ensemble.pooledPerStep[step], RISK_PERCENTILE);
      const wet = emulator.wetCells(depth, DEPTH_THRESHOLD_M);
      maxRisk = Math.max(maxRisk, wet / floodableCells);
    }
    risks.push(Math.round(Math.min(maxRisk, 1) * 100));
  }
  return risks;
}

function dayLabel(date) {
  return `${DAY_NAMES[date.getUTCDay()]}<br>${date.getUTCDate()} ${MONTH_NAMES[date.getUTCMonth()]}`;
}

function renderBars(times, risks) {
  $('chart').innerHTML = risks
    .map((pct, i) => `
      <div class="bar-column">
        <div class="bar-value">${pct}%</div>
        <div class="bar" style="height: ${pct}%;"></div>
        <div class="bar-label">${dayLabel(times[i * 4])}</div>
      </div>`)
    .join('');
}

function setStatus(text) {
  $('subtitle').textContent = text;
}

function formatTimestamp(date) {
  const hh = String(date.getHours()).padStart(2, '0');
  const mm = String(date.getMinutes()).padStart(2, '0');
  return `Updated ${DAY_NAMES[date.getDay()]} ${date.getDate()} ${MONTH_NAMES[date.getMonth()]}, ${hh}:${mm}`;
}

async function main() {
  try {
    setStatus('Loading forecast…');

    const emulator = await FloodEmulator.load('data');
    const meta = emulator.meta;
    const params = await (await fetch('data/params.json')).json();

    const history = await fetchHistory(meta.catchment);
    const seed = params.sets.map(() => [...params.initial_state]);
    const spunUpState = generateRiverFlows(
      history, params.sets, seed, meta.catchment
    ).state;

    const forecast = await fetchForecast(meta.catchment);
    const ensemble = runEnsemble(forecast, params.sets, spunUpState, meta.catchment);

    renderBars(ensemble.times, dayRiskPercentages(emulator, ensemble));
    setStatus(formatTimestamp(new Date()));
  } catch (error) {
    console.error(error);
    setStatus(`Unable to load forecast: ${error.message}`);
  }
}

main();
