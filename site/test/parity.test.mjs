/**
 * Parity of the JavaScript hydrology with the MATLAB reference.
 *
 * The same anchor test the Python port uses (core/tests/test_hydrology_benchmark.py):
 * feed the recorded rainfall and PET straight into the model alongside the
 * reference initial conditions, and compare flow and end state against the
 * MATLAB outputs in Data/.
 *
 * This exists so the browser and notebook paths cannot silently diverge.
 *
 *     node site/test/parity.test.mjs
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { modelFun, fao56 } from '../js/hydrology.js';

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'Data');

// Set by the benchmark files' three-decimal rounding, not by any looseness in
// the code. Do not loosen.
const ATOL = 1e-3;
const CATCHMENT_AREA_KM2 = 212.264;
const DT = 0.25;

function readMatrix(name, sep = /[,\s]+/) {
  return readFileSync(join(DATA, name), 'utf8')
    .trim()
    .split('\n')
    .map((line) => line.trim().split(sep).map(Number));
}

function readVector(name) {
  return Float64Array.from(readMatrix(name).map((row) => row[0]));
}

let failures = 0;

function check(label, got, want, atol = ATOL) {
  let worst = 0;
  for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs(got[i] - want[i]));
  const ok = worst <= atol;
  if (!ok) failures++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label.padEnd(34)} max abs err ${worst.toExponential(3)}`);
  return ok;
}

console.log('\nJavaScript hydrology vs MATLAB reference\n');

const qp = readVector('qp_Benchmark.csv');
const Ep = readVector('Eq_Benchmark.csv');
const X = readMatrix('RainfallRunoffModelParameters.csv', ',').map((r) => r.slice(0, 4));
const F0 = readMatrix('RainfallRunoffModelInitialConditions.csv', ',');

const Qref = readMatrix('Q_Benchmark.csv', ',');       // (64 steps, 100 sets)
const F0ref = readMatrix('F0_Benchmark.csv');          // whitespace-delimited

const state = F0.map((r) => [...r]);
const flows = modelFun(qp, Ep, DT, CATCHMENT_AREA_KM2, X, state);

// flows is per-parameter-set; the reference is per-step.
const flat = [];
const flatRef = [];
for (let step = 0; step < qp.length; step++) {
  for (let set = 0; set < X.length; set++) {
    flat.push(flows[set][step]);
    flatRef.push(Qref[step][set]);
  }
}
check('river flow (64 x 100)', flat, flatRef);
check('end state (100 x 3)', state.flat(), F0ref.flat());

// FAO56 has no MATLAB reference, so this pins the golden values captured from
// the Django implementation - the same ones core/tests/test_fao56.py asserts.
console.log('');
const n = 64;
const phase = (i) => i % 4;
const RH = Float64Array.from({ length: n }, (_, i) => 80 - 10 * (phase(i) === 2));
const tMaxC = Float64Array.from({ length: n }, (_, i) => 28 + 3 * (phase(i) === 2));
const tMinC = Float64Array.from({ length: n }, (_, i) => 21 - 2 * (phase(i) === 0));
const Tmean = Float64Array.from({ length: n }, (_, i) => (tMinC[i] + tMaxC[i]) / 2);
const dailyMin = new Float64Array(n);
const dailyMax = new Float64Array(n);
for (let d = 0; d < n / 4; d++) {
  let lo = Infinity;
  let hi = -Infinity;
  for (let j = 0; j < 4; j++) {
    lo = Math.min(lo, tMinC[d * 4 + j]);
    hi = Math.max(hi, tMaxC[d * 4 + j]);
  }
  for (let j = 0; j < 4; j++) { dailyMin[d * 4 + j] = lo; dailyMax[d * 4 + j] = hi; }
}

const { ETo, E0 } = fao56(
  DT, new Date(Date.UTC(2024, 0, 1)), dailyMin, dailyMax,
  1157.0, -7.125, Tmean, new Float64Array(n).fill(1.2), RH
);

const GOLDEN_ETO = [
  4.3642717952, 4.4067126252, 4.6181448789, 4.4073555842,
  4.3655365304, 4.4080128129, 4.6194904774, 4.4086839542,
];
const GOLDEN_E0 = [
  5.9843363098, 6.0289608110, 6.2569163434, 6.0298324349,
  5.9860570415, 6.0307234036, 6.2587310277, 6.0316332327,
];
check('FAO56 ETo vs Django golden', ETo.subarray(0, 8), GOLDEN_ETO, 1e-9);
check('FAO56 E0  vs Django golden', E0.subarray(0, 8), GOLDEN_E0, 1e-9);

console.log(`\n${failures === 0 ? 'ALL CHECKS PASS' : `${failures} CHECK(S) FAILED`}\n`);
process.exit(failures === 0 ? 0 : 1);
