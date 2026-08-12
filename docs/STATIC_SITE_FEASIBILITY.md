# Running ManyFEWS as a static site

An assessment of whether ManyFEWS can run from a static web server such as
GitHub Pages, with a working prototype in [`site/`](../site/).

**Verdict: yes, and it can be genuinely live rather than a pre-baked snapshot.**
The prototype fetches real forecasts, runs the full hydrological chain and the
flood emulator in the browser, and renders the inundation map — with no server,
no database, no API key and no scheduled job. First load transfers 3.7 MB;
after that a flood map redraws in about 45 ms.

Every number below was measured against the prototype, not estimated.

---

## Why it works

Two properties of this system make it possible. Neither is obvious from the
Django application's architecture, which implies the opposite.

**The whole flood map is a function of one number.** Each of the 302,748 ground
cells stores a cubic in river flow plus a threshold:

```
depth(Q) = max(0, P0 + Q·(P1 + Q·(P2 + Q·P3)))   if Q ≥ minQ, else 0
```

There is no spatial coupling, no time dependence and no state. Given a flow, the
entire inundation surface follows from 302,748 independent polynomial
evaluations — about 3 ms of arithmetic. The Django application spends hours per
forecast time on this (`flood_risk.py` carries a `# FIXME: took several hours for
1 time`) because it evaluates per pixel through the ORM and then builds a
32/64/128/256 aggregation pyramid to make the result servable. With the whole
grid held as typed arrays in a browser tab, none of that machinery is needed.

**Open-Meteo permits cross-origin requests.** Both endpoints return
`access-control-allow-origin: *`:

```
$ curl -sI -H "Origin: https://example.github.io" \
    "https://ensemble-api.open-meteo.com/v1/ensemble?..." | grep -i access-control
access-control-allow-origin: *
access-control-allow-methods: GET, POST, OPTIONS
```

So the browser can fetch both the ensemble forecast and the reanalysis archive
directly. No proxy, no key, no backend, and therefore no secret to protect and
no scheduled job to keep the data fresh. This is what separates "a static site
showing yesterday's model output" from "a static site that *is* the model".

---

## Measured performance

Chromium, cold cache, local server. The compute figures are the ones that matter
— network latency will dominate for real users.

| Stage | Time |
|---|---|
| Load and unpack the 302,748-cell emulator | 29 ms |
| Fetch 29 days of reanalysis (spin-up input) | 46 ms |
| Fetch the 16-day, 10-member ensemble forecast | 135 ms |
| Spin-up: 116 steps × 100 parameter sets | 1.8 ms |
| Forecast: 64 steps × 100 sets × 10 members | 10.7 ms |
| Flood depth at a single flow (302,748 cells) | 3 ms |
| Flood depth percentile across 1,000 pooled samples | 188 ms |
| Full page ready, cold | **1.9 s** |
| Flow-slider redraw (compute + paint + overlay) | **45 ms** |

The hydrology is essentially free — 64,000 model steps in 11 ms. The only
non-trivial cost is the percentile path, and that is entirely the ~9,500
non-monotone cells being evaluated in full (see below).

For comparison, the Django map round-trips `/depths/<day>/<hour>/<bbox>` to
PostGIS on every `moveend`. The static version has no such round-trip at all: it
already holds every cell.

---

## Payload

| Asset | Raw | Transferred |
|---|---|---|
| `grid.bin.gz` — position + 4 float32 coefficients + minQ index per cell | 6.36 MB | **3.57 MB** |
| `vendor/leaflet.js` + CSS | 162 kB | 52 kB |
| `channel.bin.gz`, `monotone.bin.gz` — bitsets, one bit per cell | 76 kB | 4 kB |
| `params.json` — 100 PDM parameter sets | 3.1 kB | 1.4 kB |
| `meta.json`, app JS, CSS | 30 kB | 10 kB |
| **First load** | | **≈ 3.7 MB** |

Well inside GitHub Pages' limits (1 GB per site, 100 MB per file, 100 GB/month
soft bandwidth cap — roughly 27,000 cold loads).

The binaries are **pre-compressed and served as `.gz`**, decompressed in the
browser with `DecompressionStream`. GitHub Pages will not gzip
`application/octet-stream` on the fly, and 6.4 MB versus 3.6 MB is worth one API
call. This needs Chrome 80+, Firefox 113+ or Safari 16.4+; the prototype says so
explicitly rather than failing obscurely.

### Encodings considered and rejected

| Option | Transferred | Why not |
|---|---|---|
| Precomputed depth stack, 16 flow levels, uint8 | 2.47 MB | Smaller, but quantises flow to 16 steps — the slider becomes a stepper, and forecast percentiles land between levels |
| Dense float32 grid, 1682 × 770 × 5 | 3.79 MB | Same transfer size but 25.8 MB resident in the tab, for a grid that is only 23.5% occupied |
| Sparse coefficients (**chosen**) | 3.57 MB | Exact at any flow, 6.4 MB resident |

---

## How the prototype is built

```
site/
  index.html            page, inline CSS, light/dark
  js/weather.js         Open-Meteo fetch + 6-hour bucketing
  js/hydrology.js       FAO56 / PDM / routing / ModelFun
  js/emulator.js        depth cubics + the percentile shortcut
  js/app.js             orchestration, Leaflet map, canvas overlay, chart
  data/                 packed model data, committed
  test/parity.test.mjs  checks the JS against the MATLAB reference
core/scripts/build_static_data.py
```

`build_static_data.py` imports `manyfews_core`, so the packing logic and the
notebook path share one definition of the model.

**Two implementation details worth carrying forward.**

*The lattice is rotated.* The emulator's cells form a regular grid rotated
0.3387° from north — almost certainly built in a projected CRS and reprojected.
Over the 3.4 km domain that is about 20 m, or ten cell widths. Reconstructing an
axis-aligned image from lattice indices would put the flood visibly in the wrong
place, so the build script bins the actual coordinates into a north-up
1682 × 770 grid instead. Web Mercator then stretches the overlay about 0.75%
vertically at this latitude, which is negligible.

*The percentile shortcut is exact.* For a cell whose clamped response is monotone
in flow, sorting depths is sorting flows, so the percentile of the depths equals
the depth at the percentile of the flows. That is 8 polynomial evaluations
instead of 1,000. 96.86% of cells qualify; the remaining 9,514 are evaluated in
full. Agreement with brute force is 4e-16 — the shortcut is not an
approximation. The monotonicity mask is computed once at build time and shipped
as a bitset, because the closed-form test for it is **wrong**: taking the roots of
the derivative misses 2,570 cells whose cubic decreases monotonically through
zero, which have no interior critical point but whose clamped response still
decreases.

---

## Correctness

The JavaScript is checked against the same MATLAB reference outputs as the Python
port, in CI, so the two cannot silently diverge:

```
$ node site/test/parity.test.mjs
  PASS  river flow (64 x 100)              max abs err 5.017e-4
  PASS  end state (100 x 3)                max abs err 7.861e-5
  PASS  FAO56 ETo vs Django golden         max abs err 4.780e-11
  PASS  FAO56 E0  vs Django golden         max abs err 3.342e-11
```

The 5e-4 figure is the benchmark files' own three-decimal rounding, not model
error; the Python port lands on exactly the same number.

Comparing the browser's flood depths against Python across six flows and a
1,000-sample percentile:

| Quantity | Agreement |
|---|---|
| Total depth summed over all cells | 2e-10 relative |
| Maximum depth | 4e-7 relative |
| Wet-cell count | exact, except at Q ≥ 300 |

At the clamp the browser reports 302,696 wet cells against Python's 302,746. The
cause is float32 packing: the largest depth difference anywhere is 23 microns,
but exactly 50 cells have a fitted depth of *precisely* 0.01 m — a floor in the
source data — and rounding nudges them below the 1 cm wet threshold. This is a
tie-breaking artefact at a threshold, not a modelling difference, and float64
coefficients would double the payload to fix 0.017% of cells.

---

## What a static site cannot do

Beyond the features deliberately out of scope (accounts, alerts, SMS,
scheduling, the Django admin):

- **No persistence.** A user's alert polygon, saved location or notification
  preference has nowhere to live. Anything user-specific needs a backend.
- **No server-side rate limiting or key management.** The site depends on
  Open-Meteo staying free and CORS-open to anonymous browsers. If that changed,
  every visitor would break at once and the only fix is a proxy — i.e. a server.
- **A 3.7 MB first load**, which is a real cost on the metered mobile
  connections a flood warning system's users may well be on. Mitigable by
  shipping a 16-level depth stack (2.5 MB) for a first paint and upgrading to
  coefficients in the background, at the cost of complexity.
- **Compute on the client.** A low-end phone will not manage 45 ms per redraw.
  The prototype has not been profiled on one.
- **No audit trail.** Nothing records what any given user was shown, which
  matters for a system whose output people might act on.
- **Catchment-specific.** The packed grid is Majalaya. Another catchment means
  another build, and `build_static_data.py` currently assumes one.

---

## Recommendation

For **demonstration, teaching, outreach and stakeholder review**, the static site
is straightforwardly better than the Django deployment: no infrastructure, no
running cost, no credentials, and a flood map that responds instantly instead of
round-tripping to PostGIS. The prototype in `site/` is complete enough to deploy
today via the included `.github/workflows/static-site.yml`.

For **operational warning**, it is not a replacement. Alerting, persistence,
audit and oversight are the reasons the Django application exists, and none of
them can live in a static page. The realistic split is a static public-facing
map, with the alerting service kept server-side.

The most interesting finding is one that applies to the existing application
regardless of this assessment: because depth is a pure function of flow, the
entire `DepthPrediction` / `AggregatedDepthPrediction` layer — up to 19 million
rows a day and the 87,000 spatial aggregate queries per forecast time that build
the tile pyramid — is computing and storing something that can be derived in
3 ms from data that fits in 6 MB. That is worth revisiting independently of
whether the static site is ever adopted.

---

## Related

- [`site/`](../site/) — the prototype
- [`core/`](../core/) — the shared Django-free package
- [`core/notebooks/`](../core/notebooks/) — the Colab notebooks
