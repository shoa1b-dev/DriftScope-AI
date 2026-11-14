# Phase 10.1 — Test Environment-Coupling Fix (triggered by your real laptop run)

## What happened

 laptop run (`river` and apparently `shap` genuinely installed) surfaced a real, valuable finding: several tests assumed the real dependency would *always* be absent, rather than explicitly simulating absence. Once `river` was actually available, `run_experiment(...)` without an injected monitor factory correctly constructed and ran **real `AdwinDriftMonitor`** — which is genuinely correct behavior — but on this project's default *weak* synthetic drift signal (small offset/scale, sine+noise base series), real ADWIN did not detect drift within the scanned range, raising `DriftNeverDetectedError` instead of the `RiverUnavailableError` those tests expected. That's not a bug in the runner — `DriftNeverDetectedError` is exactly the documented, correct outcome for "monitor ran for real and legitimately didn't detect." The bug was in the **tests**: they conflated "the runner propagates a real-dependency-unavailable error correctly" with "the ambient environment happens to lack that dependency" — two different things.

## Fix

Four confirmed sites (3 in `tests/unit/test_experiment_runner.py`, 1 in `tests/integration/test_full_matrix_validation.py`) were changed to inject a **deterministic fake that raises `RiverUnavailableError`**, rather than omitting the factory and hoping `river` is absent. This makes each test's actual claim — "the runner propagates this error without swallowing it" — true regardless of environment. Two more sites testing the equivalent `SHAPUnavailableError` path were fixed the same way, proactively, since they had the identical latent flaw and would break the moment `shap` is also installed for real.

**Added, not just fixed:** two new tests, gated with `@unittest.skipUnless(RIVER_AVAILABLE, ...)` / `@unittest.skipUnless(SHAP_AVAILABLE, ...)` (the same pattern Phase 3/5 already established), that exercise the **real success path** when the dependency genuinely is available — `TestRealRiverIntegration` uses a deliberately strong synthetic drift so real ADWIN detection isn't data-dependent, and `TestRealSHAPIntegration` checks real SHAP produces correctly-shaped, finite attributions through the full runner. These are currently skipped in this sandbox (still no `river`/`shap` here) but should **actually run** on your laptop.

## Result in this sandbox

```
python3 -m unittest discover -s tests -v
Ran 285 tests
OK (skipped=6)
```
279 passed, 6 skipped (the original 4 + the 2 new real-dependency tests, all correctly skipped here), 0 failed.
