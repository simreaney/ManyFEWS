# Issues found in the Django application

Found while building [`core/`](../core/) and confirmed against the real data.
None are fixed by that work — it is additive and does not touch `manyfews/` —
but each affects the running system and is worth triaging separately.

Ordered by consequence.

---

## 1. The flood emulator has no upper clamp

`manyfews/calculations/flood_risk.py` bounds negative depths but not large ones.
The fitted cubics are only valid over the flow range they were calibrated on:

| River flow | Max depth reported |
|---|---|
| 300 m³/s | 15.0 m — plausible |
| 500 m³/s | 118.2 m |
| 800 m³/s | 1121.6 m |

Worse, the **flooded-cell count falls** above 300 m³/s as cubics turn over and go
negative, so an extreme event can report *less* inundation than a moderate one.

`minQ` tops out at 300, which is a reasonable read of where calibration ended.
`manyfews_core` clamps the input at 300 m³/s; clamping the input rather than the
output also keeps the response monotone.

**Fix:** clamp `flow_values` to a configurable ceiling before evaluating.

## 2. Test mode's 100 mm storm cannot produce a flood

`TestModeSettings.STORM_TOTAL_MM = 100.0` (in the current working tree). Running
the real numerics with that storm two days ahead:

| Storm total | Peak p50 | Peak p90 | Floods? |
|---|---|---|---|
| 100 mm | 15.2 | 24.6 | no |
| 150 mm | 33.2 | 66.8 | marginal |
| **200 mm** | **62.2** | **140.0** | **yes** |
| 300 mm | 163.3 | 336.7 | yes, but past issue 1 |

The lowest threshold anywhere in the emulator is 50 m³/s, so at 100 mm every
centile in every cell is zero and the map stays empty — meaning the test mode
cannot demonstrate the thing it exists to demonstrate.

**Fix:** default to ~200 mm. (Exact figures depend on antecedent conditions;
these are from a dry-season spin-up, so a wet-season run would flood sooner.)

## 3. The risk percentage formula omits its baseline

`flood_risk.calculate_risk_percentages`:

```python
risk = n / (LARGE_FLOOD_COUNT - CHANNEL_CELL_COUNT)
```

Given the two constants are a floor and a ceiling, this looks like it should be
`(n - CHANNEL_CELL_COUNT) / (LARGE_FLOOD_COUNT - CHANNEL_CELL_COUNT)`. As
written, a catchment sitting at exactly the channel cell count reports 7% risk
rather than zero.

## 4. The risk constants do not match the shipped parameter file

`CHANNEL_CELL_COUNT = 93794` and `LARGE_FLOOD_COUNT = 1440811`, but
`Data/floodEmulatorParams-20230921.csv` contains **302,748 cells in total**. So:

- `LARGE_FLOOD_COUNT` is unreachable — risk saturates at about 0.225 and the
  headline number can never approach 100%.
- `CHANNEL_CELL_COUNT` is 31% of the whole grid, whereas the mapped channel
  actually covers 20,959 cells (6.9%).

These presumably came from an earlier, larger parameter file. Both need
recomputing for the file in use, ideally derived at load time rather than pinned
in settings.

## 5. `MODEL_TIMESTEP` is configurable but 0.25 is hardcoded

Two places assume four buckets per day:

- `generate_river_flows.py` — `np.array(TempMin).reshape((int(N/4)), 4)`
- `FAO56` — `J = beginDateNum + np.arange(0, np.size(Tmax)/4, dt)`

Any other `MODEL_TIMESTEP` silently produces a wrong day grouping and a
length-mismatched day-of-year vector, rather than an error. `CatchmentConfig` in
`manyfews_core` raises instead.

## 6. The archive's ingestion lag can corrupt the spin-up

`initialModelSetUp` requests Open-Meteo archive data up to *yesterday*, but the
ERA5T archive trails real time by roughly five days. Recent hours come back
`null`; `_bucket_to_6h` drops those buckets with a warning; the series is then
short and misaligned, and `reshape(N/4, 4)` groups the wrong hours into days.

`prepareOpenMeteoHistorical` does raise if too few buckets arrive, so this fails
loudly *most* of the time — but a whole-day shortfall passes the count check
while still shifting the alignment.

**Fix:** end the window several days back, and assert 6-hourly contiguity rather
than only the total count.

## 7. `Data/channel.geojson` is not valid JSON

Five stray characters (`\n\n] } }`) after the closing brace. `json.load` raises
`Extra data: line 874`. GEOS tolerates it, which is why nothing has noticed.
Anything that reads this file with a standard JSON parser will fail.

## 8. The temperature swap-guard in FAO56 is a no-op

`generate_river_flows.py:228-229`:

```python
Tmax = np.maximum(Tmax, Tmin)
Tmin = np.minimum(Tmax, Tmin)   # reads the Tmax just reassigned
```

The second line evaluates to `Tmin`, so swapped inputs are not corrected. In
practice inputs are never swapped, so fixing it would change nothing — but
fixing it *would* break parity with the MATLAB reference outputs, so it should
be corrected deliberately and with the benchmarks regenerated, or left alone with
a comment. `manyfews_core` preserves it and pins it with a test.

## 9. Five unreachable `try/except NameError` blocks — and one that is not

`generate_river_flows.py` has six blocks of the form:

```python
try:
    name
except NameError:
    name = <fallback>
```

In five, `name` is a function parameter, so the lookup always succeeds and the
fallback is dead: `q0` (L117), `S0` (L173), `T` (L236), `u2` (L242), and the
`ea` fallback (L270).

The sixth, `Rs` at L309, is the opposite: `Rs` is **not** a parameter and is never
bound, so the lookup always raises and the `except` body is the only live path.
Anyone tidying these by pattern-matching will delete the wrong branch and make
`FAO56` raise `NameError` at `Rso`.

## 10. Depth prediction stores what could be derived in milliseconds

Not a bug, but the largest structural cost in the system.

Because depth is a pure function of flow with no spatial coupling and no state,
the entire `DepthPrediction` / `AggregatedDepthPrediction` layer — up to ~19 M
rows per day, plus roughly 87,000 spatial aggregate queries per forecast time to
build the 32/64/128/256 tile pyramid, and the `# FIXME: took several hours for
1 time` in `flood_risk.py` — is materialising something derivable from 6 MB of
coefficients in about 3 ms.

Vectorising `predict_depths` over numpy arrays instead of looping per pixel
through the ORM would be a large win on its own. Not storing the result at all
would be larger. See [`STATIC_SITE_FEASIBILITY.md`](STATIC_SITE_FEASIBILITY.md).

---

## Reproducing these

All figures come from the shipped data and are checked by
[`core/tests/`](../core/tests/):

```bash
pip install -e './core[test]'
cd core && pytest -q
```

In particular `test_emulator.py` pins the clamp behaviour and the monotonicity
count, `test_channel.py` asserts that `json.load` fails on the GeoJSON, and
`test_raster.py` pins the legacy risk formula.
