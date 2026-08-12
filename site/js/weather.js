/**
 * Open-Meteo ingestion in the browser.
 *
 * Mirrors manyfews_core/weather.py. Both endpoints send
 * `access-control-allow-origin: *`, so a page served from GitHub Pages can call
 * them directly - no proxy, no key, no backend.
 *
 * Hourly values are aggregated into the six-hour buckets the model wants:
 * rainfall sums, temperature keeps the extremes, humidity and the decomposed
 * wind components average.
 */

const ENSEMBLE_URL = 'https://ensemble-api.open-meteo.com/v1/ensemble';
const ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive';

const HOURLY = [
  'precipitation',
  'temperature_2m',
  'windspeed_10m',
  'winddirection_10m',
  'relativehumidity_2m',
].join(',');

const KELVIN = 273.15;
const BUCKET_HOURS = 6;

function isoDate(date) {
  return date.toISOString().slice(0, 10);
}

function daysFromToday(offset) {
  const now = new Date();
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + offset));
}

async function getJson(url, params) {
  const query = new URLSearchParams({ ...params, timeformat: 'unixtime', timezone: 'UTC' });
  const response = await fetch(`${url}?${query}`);
  if (!response.ok) throw new Error(`Open-Meteo returned HTTP ${response.status}`);
  return response.json();
}

/** Wind speed and meteorological direction to components, m/s. */
export function windComponents(speedKmh, directionDeg) {
  const speed = speedKmh / 3.6;
  const rad = (directionDeg * Math.PI) / 180;
  return [-speed * Math.sin(rad), -speed * Math.cos(rad)];
}

/**
 * Aggregate one member's hourly series into six-hour buckets.
 *
 * Incomplete buckets are dropped rather than guessed at, which is what the
 * archive's null-padded recent days produce.
 */
export function bucketTo6h(hourly, suffix = '') {
  const times = hourly.time || [];
  const precip = hourly[`precipitation${suffix}`] || [];
  const temp = hourly[`temperature_2m${suffix}`] || [];
  const wspd = hourly[`windspeed_10m${suffix}`] || [];
  const wdir = hourly[`winddirection_10m${suffix}`] || [];
  const rh = hourly[`relativehumidity_2m${suffix}`] || [];

  const out = { times: [], rh: [], tmaxK: [], tminK: [], windU: [], windV: [], precip: [] };
  const end = times.length - (times.length % BUCKET_HOURS);

  for (let start = 0; start < end; start += BUCKET_HOURS) {
    let sumP = 0, sumRH = 0, sumU = 0, sumV = 0;
    let tMax = -Infinity, tMin = Infinity;
    let count = 0;
    let complete = true;

    for (let i = start; i < start + BUCKET_HOURS; i++) {
      if (precip[i] == null || temp[i] == null || rh[i] == null ||
          wspd[i] == null || wdir[i] == null) {
        complete = false;
        break;
      }
      sumP += precip[i];
      sumRH += rh[i];
      tMax = Math.max(tMax, temp[i]);
      tMin = Math.min(tMin, temp[i]);
      const [u, v] = windComponents(wspd[i], wdir[i]);
      sumU += u;
      sumV += v;
      count++;
    }
    if (!complete) continue;

    out.times.push(new Date(times[start] * 1000));
    out.precip.push(sumP);
    out.rh.push(sumRH / count);
    out.tmaxK.push(tMax + KELVIN);
    out.tminK.push(tMin + KELVIN);
    out.windU.push(sumU / count);
    out.windV.push(sumV / count);
  }

  // Truncate to whole days - the model groups four buckets per day.
  const whole = Math.floor(out.times.length / 4) * 4;
  for (const key of Object.keys(out)) out[key] = out[key].slice(0, whole);
  return out;
}

/** Ensemble member suffixes present in a response, control run first. */
export function memberSuffixes(hourly) {
  const found = [];
  for (const key of Object.keys(hourly)) {
    if (key === 'precipitation') found.push('');
    else if (key.startsWith('precipitation_member')) found.push(key.slice('precipitation'.length));
  }
  return found.sort((a, b) => (a === '' ? -1 : b === '' ? 1 : a.localeCompare(b)));
}

/**
 * Observed history for the spin-up.
 *
 * Ends `lagDays` back because Open-Meteo's archive trails real time by about
 * five days; asking for anything more recent returns nulls, which silently
 * shift the day grouping.
 */
export async function fetchHistory(catchment, days = 29, lagDays = 6) {
  const end = daysFromToday(-lagDays);
  const start = new Date(end.getTime() - (days - 1) * 86400000);

  const raw = await getJson(ARCHIVE_URL, {
    latitude: catchment.weather_lat,
    longitude: catchment.weather_lon,
    hourly: HOURLY,
    start_date: isoDate(start),
    end_date: isoDate(end),
  });

  const series = bucketTo6h(raw.hourly || {});
  if (series.times.length < days * 4 - 4) {
    throw new Error(
      `Archive returned only ${series.times.length} of ${days * 4} expected buckets.`
    );
  }
  return series;
}

/** The ensemble forecast, one series per member. */
export async function fetchForecast(catchment, { model = 'gfs_seamless', days = 16, maxMembers = 10 } = {}) {
  const start = daysFromToday(0);
  const end = new Date(start.getTime() + (days - 1) * 86400000);

  const raw = await getJson(ENSEMBLE_URL, {
    latitude: catchment.weather_lat,
    longitude: catchment.weather_lon,
    hourly: HOURLY,
    models: model,
    start_date: isoDate(start),
    end_date: isoDate(end),
  });

  const hourly = raw.hourly || {};
  let suffixes = memberSuffixes(hourly);
  if (!suffixes.length) throw new Error('Open-Meteo response contained no precipitation data');
  if (maxMembers > 0) suffixes = suffixes.slice(0, maxMembers);

  return suffixes.map((suffix) => ({
    member: suffix ? `member${suffix.slice('_member'.length)}` : 'control',
    ...bucketTo6h(hourly, suffix),
  }));
}

/**
 * Replace one day's rainfall with a design storm.
 *
 * Mirrors the Django test mode, which replaces rather than adds. Note that its
 * 100 mm default is not enough to flood this catchment - roughly 200 mm is.
 */
export function injectStorm(series, totalMm, daysAhead) {
  const first = series.times[0];
  const dayStart = Date.UTC(
    first.getUTCFullYear(), first.getUTCMonth(), first.getUTCDate() + daysAhead
  );
  const dayEnd = dayStart + 86400000;

  const target = [];
  series.times.forEach((t, i) => {
    if (t.getTime() >= dayStart && t.getTime() < dayEnd) target.push(i);
  });
  if (!target.length) return series;

  const precip = [...series.precip];
  for (const i of target) precip[i] = totalMm / target.length;
  return { ...series, precip };
}
