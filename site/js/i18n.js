/**
 * Minimal i18n: English + Indonesian, with locale auto-detected from the
 * browser and overridable by the user (persisted in localStorage).
 */

const STORAGE_KEY = 'manyfews-locale';

const STRINGS = {
  en: {
    docTitle: 'ManyFEWS — Majalaya flood forecast',
    metaDescription: 'Live ensemble flood forecast for Majalaya, West Java. Runs entirely in the browser.',
    title: 'ManyFEWS — Majalaya flood forecast',
    subtitle: 'Live ensemble forecast for the upper Citarum, West Java. Everything below is computed in your browser: there is no server.',
    langButtonLabel: 'Language',

    statusLoadingModel: 'Loading the flood model…',
    statusLoadingModelSized: (mb) => `Loading the flood model (${mb} MB)…`,
    statusFetchingHistory: 'Fetching observed weather for the spin-up…',
    statusSpinningUp: 'Spinning up the catchment model…',
    statusFetchingForecast: 'Fetching the ensemble forecast…',
    statusRunning: 'Running the catchment model…',
    statusReady: (elapsed, cells, members, sets) =>
      `Ready in ${elapsed}s — ${cells} flood cells, ${members} weather members × ${sets} parameter sets.`,
    statusError: (message) => `Could not load: ${message}`,
    stormOn: (mm, day) => `Synthetic ${mm} mm storm applied on day ${day}.`,
    stormOff: 'Live forecast.',

    modeLabel: 'View',
    modeForecast: 'Live forecast',
    modeExplore: 'Explore by flow',
    pctLabel: 'Confidence',
    pctMedian: 'Median',
    pct90: '90th percentile',
    flowLabel: 'River flow',
    flowUnit: 'm³/s',
    scenarioLabel: 'Scenario',
    stormCheck: 'Add a 200 mm storm on day 2',

    locateLabel: 'Location',
    locateButton: 'Show my location',
    locateButtonHide: 'Hide my location',
    locateLocating: 'Locating…',
    locateDenied: 'Location permission was denied.',
    locateUnavailable: 'Your location is unavailable.',
    locateTimeout: 'Locating timed out.',
    locateUnsupported: 'Geolocation is not supported by this browser.',
    youAreHere: 'You are here',

    chartTitleH2: 'River flow forecast',
    chartSub: (members) => `100 rainfall-runoff parameter sets × ${members} weather ensemble members, at six-hourly steps.`,
    chartTitleSvg: 'River flow forecast with uncertainty bands',
    chartAxisLabel: 'River flow (m³/s)',
    chartFloodStart: (threshold) => `flooding starts — ${threshold} m³/s`,
    chartLegendP10P90: '10th–90th percentile',
    chartLegendP30P50: '30th–50th percentile',
    chartLegendMedian: 'Median',

    mapTitleH2: 'Flood depth',
    mapSubDefault: 'Cells inside the river channel are excluded — the emulator reports water there, but that is just the river.',
    mapSubExplore: (flow) => `Inundation at a chosen river flow of ${flow} m³/s, independent of any forecast. Channel cells are excluded.`,
    mapSubForecast: (date, pct) => `Peak of the forecast, ${date} UTC, at the ${pct}th percentile. Channel cells are excluded.`,
    legendDepth: 'Depth',
    legendLimit: 'Prediction area limit',
    limitTooltip: 'Flood depth predictions are only available inside this box.',
    statSituation: 'Situation',
    statFlooded: 'Flooded cells',
    statDeepest: 'Deepest',
    statMeanWet: 'Mean where wet',
    headlineNoFlooding: 'No flooding',
    headlineFlooding: 'Flooding',
    noFloodTitle: 'No flooding forecast.',
    noFloodBody: (peakFlow, pct, threshold) =>
      `Peak flow reaches ${peakFlow} m³/s at the ${pct}th percentile, below the ${threshold} m³/s at which any cell begins to flood. The map will be empty — this is the normal result for this catchment. Try the storm scenario, or switch to Explore by flow.`,
    floodTitle: 'Flooding forecast.',
    floodBody: (wet, pct, peakFlow) =>
      `${wet} cells inundated at the ${pct}th percentile, peak flow ${peakFlow} m³/s.`,
    floodStormNote: 'Driven by a synthetic storm — this is not a real forecast.',

    tableTitleH2: 'Daily summary',
    tableSub: 'Peak flow per day across all members and parameter sets.',
    thDate: 'Date', thP50: 'Peak flow p50', thP90: 'Peak flow p90', thStatus: 'Status',
    rowFlooding: 'Flooding possible', rowNoFlood: 'No flooding',

    footerWhatTitle: 'What this is',
    footerWhatBody: 'A demonstration port of <a href="https://github.com/simreaney/ManyFEWS">ManyFEWS</a>, running the full chain — Open-Meteo ensemble forecast, FAO56 evapotranspiration, a PDM soil store, two routing stores, and a per-cell flood-depth emulator — entirely client-side. Weather comes from <a href="https://open-meteo.com">Open-Meteo</a>; basemap from OpenStreetMap.',
    footerLimitsTitle: 'Limitations',
    footerLimits: [
      'The depth model is a statistical emulator of a 2D hydraulic model, not hydraulics. It is valid roughly between 50 and 300 m³/s; outside that range the fitted cubics diverge, so the input is clamped.',
      'Depth depends on peak flow alone — no hydrograph shape, no antecedent floodplain wetness, no breaches or blockages.',
      'The map only predicts depth inside the flood grid shown as a dashed box — nothing is known outside it.',
      'Uncertainty shown is parametric only. It says nothing about whether the model structure or the underlying hydraulic model is right.',
      'Not an operational warning system. No alerts, no accounts, no oversight.',
    ],
    footerDevTitle: 'Developers',
    footerDevCredit: 'Created by Sim Reaney. Source code available on <a href="https://github.com/simreaney/ManyFEWS">GitHub</a>.',
    footerDevProject: 'This is from the <a href="https://gotw.nerc.ac.uk/list_full.asp?pcode=NE%2FS00310X%2F1&cookieConsent=A">JavaFloodOne research project</a>, co-funded by NERC in the UK and Ristekdikti in Indonesia.',
    footerDevPartners: 'The project was in partnership with colleagues from The Bandung Institute of Technology (ITB) and UKCEH.',
  },

  id: {
    docTitle: 'ManyFEWS — Prakiraan banjir Majalaya',
    metaDescription: 'Prakiraan banjir ansambel langsung untuk Majalaya, Jawa Barat. Berjalan sepenuhnya di peramban.',
    title: 'ManyFEWS — Prakiraan banjir Majalaya',
    subtitle: 'Prakiraan ansambel langsung untuk hulu Citarum, Jawa Barat. Semua di bawah ini dihitung di peramban Anda: tidak ada server.',
    langButtonLabel: 'Bahasa',

    statusLoadingModel: 'Memuat model banjir…',
    statusLoadingModelSized: (mb) => `Memuat model banjir (${mb} MB)…`,
    statusFetchingHistory: 'Mengambil data cuaca teramati untuk pemanasan model…',
    statusSpinningUp: 'Memanaskan model DAS…',
    statusFetchingForecast: 'Mengambil prakiraan ansambel…',
    statusRunning: 'Menjalankan model DAS…',
    statusReady: (elapsed, cells, members, sets) =>
      `Siap dalam ${elapsed} dtk — ${cells} sel banjir, ${members} anggota cuaca × ${sets} set parameter.`,
    statusError: (message) => `Tidak dapat memuat: ${message}`,
    stormOn: (mm, day) => `Badai sintetis ${mm} mm diterapkan pada hari ke-${day}.`,
    stormOff: 'Prakiraan langsung.',

    modeLabel: 'Tampilan',
    modeForecast: 'Prakiraan langsung',
    modeExplore: 'Jelajahi berdasarkan debit',
    pctLabel: 'Tingkat keyakinan',
    pctMedian: 'Median',
    pct90: 'Persentil ke-90',
    flowLabel: 'Debit sungai',
    flowUnit: 'm³/dtk',
    scenarioLabel: 'Skenario',
    stormCheck: 'Tambahkan badai 200 mm pada hari ke-2',

    locateLabel: 'Lokasi',
    locateButton: 'Tampilkan lokasi saya',
    locateButtonHide: 'Sembunyikan lokasi saya',
    locateLocating: 'Mencari lokasi…',
    locateDenied: 'Izin lokasi ditolak.',
    locateUnavailable: 'Lokasi Anda tidak tersedia.',
    locateTimeout: 'Pencarian lokasi habis waktu.',
    locateUnsupported: 'Peramban ini tidak mendukung geolokasi.',
    youAreHere: 'Anda di sini',

    chartTitleH2: 'Prakiraan debit sungai',
    chartSub: (members) => `100 set parameter hujan-limpasan × ${members} anggota ansambel cuaca, dengan langkah waktu enam jam.`,
    chartTitleSvg: 'Prakiraan debit sungai dengan pita ketidakpastian',
    chartAxisLabel: 'Debit sungai (m³/dtk)',
    chartFloodStart: (threshold) => `banjir mulai — ${threshold} m³/dtk`,
    chartLegendP10P90: 'Persentil ke-10–90',
    chartLegendP30P50: 'Persentil ke-30–50',
    chartLegendMedian: 'Median',

    mapTitleH2: 'Kedalaman banjir',
    mapSubDefault: 'Sel di dalam alur sungai tidak disertakan — emulator melaporkan air di sana, tetapi itu hanya sungai.',
    mapSubExplore: (flow) => `Genangan pada debit sungai pilihan ${flow} m³/dtk, terlepas dari prakiraan mana pun. Sel alur sungai tidak disertakan.`,
    mapSubForecast: (date, pct) => `Puncak prakiraan, ${date} UTC, pada persentil ke-${pct}. Sel alur sungai tidak disertakan.`,
    legendDepth: 'Kedalaman',
    legendLimit: 'Batas area prakiraan',
    limitTooltip: 'Prakiraan kedalaman banjir hanya tersedia di dalam kotak ini.',
    statSituation: 'Situasi',
    statFlooded: 'Sel tergenang',
    statDeepest: 'Terdalam',
    statMeanWet: 'Rata-rata di area basah',
    headlineNoFlooding: 'Tidak ada banjir',
    headlineFlooding: 'Banjir',
    noFloodTitle: 'Tidak ada prakiraan banjir.',
    noFloodBody: (peakFlow, pct, threshold) =>
      `Debit puncak mencapai ${peakFlow} m³/dtk pada persentil ke-${pct}, di bawah ${threshold} m³/dtk yang menyebabkan sel mulai banjir. Peta akan kosong — ini hasil normal untuk DAS ini. Coba skenario badai, atau beralih ke Jelajahi berdasarkan debit.`,
    floodTitle: 'Banjir diprakirakan.',
    floodBody: (wet, pct, peakFlow) =>
      `${wet} sel tergenang pada persentil ke-${pct}, debit puncak ${peakFlow} m³/dtk.`,
    floodStormNote: 'Didorong oleh badai sintetis — ini bukan prakiraan nyata.',

    tableTitleH2: 'Ringkasan harian',
    tableSub: 'Debit puncak per hari di semua anggota dan set parameter.',
    thDate: 'Tanggal', thP50: 'Debit puncak p50', thP90: 'Debit puncak p90', thStatus: 'Status',
    rowFlooding: 'Banjir mungkin terjadi', rowNoFlood: 'Tidak ada banjir',

    footerWhatTitle: 'Tentang situs ini',
    footerWhatBody: 'Sebuah demonstrasi dari <a href="https://github.com/simreaney/ManyFEWS">ManyFEWS</a>, menjalankan seluruh rantai proses — prakiraan ansambel Open-Meteo, evapotranspirasi FAO56, penyimpanan tanah PDM, dua penyimpanan aliran, dan emulator kedalaman banjir per sel — sepenuhnya di sisi klien. Data cuaca dari <a href="https://open-meteo.com">Open-Meteo</a>; peta dasar dari OpenStreetMap.',
    footerLimitsTitle: 'Keterbatasan',
    footerLimits: [
      'Model kedalaman adalah emulator statistik dari model hidraulik 2D, bukan hidraulika itu sendiri. Model ini valid kira-kira antara 50 dan 300 m³/dtk; di luar rentang itu kurva kubik yang dipasang menyimpang, sehingga masukan dibatasi (clamped).',
      'Kedalaman hanya bergantung pada debit puncak — tanpa bentuk hidrograf, tanpa kondisi kebasahan dataran banjir sebelumnya, tanpa jebolnya tanggul atau penyumbatan.',
      'Peta hanya memprakirakan kedalaman di dalam grid banjir yang ditampilkan sebagai kotak putus-putus — tidak ada informasi di luar kotak tersebut.',
      'Ketidakpastian yang ditampilkan hanya bersifat parametrik. Ini tidak mengatakan apa pun tentang apakah struktur model atau model hidraulik yang mendasarinya sudah benar.',
      'Bukan sistem peringatan operasional. Tidak ada peringatan, akun, atau pengawasan.',
    ],
    footerDevTitle: 'Pengembang',
    footerDevCredit: 'Dibuat oleh Sim Reaney. Kode sumber tersedia di <a href="https://github.com/simreaney/ManyFEWS">GitHub</a>.',
    footerDevProject: 'Ini berasal dari <a href="https://gotw.nerc.ac.uk/list_full.asp?pcode=NE%2FS00310X%2F1&cookieConsent=A">proyek penelitian JavaFloodOne</a>, didanai bersama oleh NERC di Inggris dan Ristekdikti di Indonesia.',
    footerDevPartners: 'Proyek ini dilaksanakan bekerja sama dengan rekan-rekan dari Institut Teknologi Bandung (ITB) dan UKCEH.',
  },
};

let locale = 'en';

function detectLocale() {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === 'en' || stored === 'id') return stored;

  const candidates = navigator.languages && navigator.languages.length
    ? navigator.languages
    : [navigator.language];
  const isIndonesian = candidates.some((tag) => (tag || '').toLowerCase().startsWith('id'));
  return isIndonesian ? 'id' : 'en';
}

export function getLocale() {
  return locale;
}

export function setLocale(next) {
  locale = STRINGS[next] ? next : 'en';
  localStorage.setItem(STORAGE_KEY, locale);
  document.documentElement.lang = locale;
}

export function initLocale() {
  setLocale(detectLocale());
}

/** Look up a string (or template function) for the active locale, falling back to English. */
export function t(key, ...args) {
  const entry = (STRINGS[locale] && STRINGS[locale][key] !== undefined)
    ? STRINGS[locale][key]
    : STRINGS.en[key];
  return typeof entry === 'function' ? entry(...args) : entry;
}
