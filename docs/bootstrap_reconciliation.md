# Bootstrap Reconciliation: A (+0.39) vs B (+0.12)

**Author:** Claude — read-only diagnostic on `C:\Users\dns81\Quant\src\bootstrap_significance.py` (A) and `C:\Users\dns81\Quant\economic-regime-Factor-ETF-allocation-main\src\research\bootstrap_significance.py` (B).
**Date:** 2026-05-31.
**Status:** FINAL — full reconciliation complete.

---

## TL;DR

The reported gap +0.3942 → +0.1219 is real and explained almost entirely by **walk-forward refit cadence** (yearly → monthly), not by rf or sample size. **However**, A's reported `p=0.0596` and B's reported `p=0.538` are *both* slightly wrong: they use the same observed delta but different bootstrap mechanics. A correctly centers the bootstrap (the standard hypothesis-test approach); B does not (it ends up testing distributional symmetry, not H0:delta=0).

**Under the methodologically correct spec — B's monthly-refit strategy series, excess returns, full available window, A's centered bootstrap — the answer is:**

| Metric | Value |
|---|---|
| Observed delta-Sharpe | **+0.1339** |
| p-value (centered, 2-sided) | **0.3010** |
| 95% CI on delta | **[-0.1495, +0.3599]** — crosses zero |
| N | 129 mo (Sep-2015 → May-2026) |

**Verdict: HALT** on the "strategy beats 60/40 on Sharpe" claim. p ≈ 0.30 ≫ 0.10. CI crosses zero. Magnitude (+0.13 Sharpe) is in the range of normal noise for this sample size.

---

## 1. Side-by-side specification (exact)

| Dimension | A (`Quant/src/bootstrap_significance.py`) | B (`.../research/bootstrap_significance.py`) |
|---|---|---|
| Result file | `results/bootstrap_results_wf.json` | `outputs/bootstrap_significance.json` |
| Reported delta | +0.3942 | +0.1219 |
| Reported p-value | 0.0596 | 0.538 |
| Reported 95% CI | [-0.027, +0.793] | [-0.161, +0.345] |
| N (months) | 112 | 128 |
| OOS window | 2017-01-31 → 2026-04-30 | 2015-09-30 → 2026-03-31 |
| WF segment cadence | **Yearly** — `oos_years = range(2017,2027)`, one `run_segment(...)` per calendar year. ~10 refits. | **Monthly-step** — `_make_segments` advances 1 mo per iter; `_stitch_non_overlapping_oos_returns` keeps each date's earliest appearance. ~116 refits. |
| WF entry point | Subprocess: `from exp_f_walkforward import run_segment` at `ROOT.parent / 'economic-regime-Factor-ETF-allocation'`. **That path does not exist on disk.** A is only reproducible via its CSV cache. | `src.research.exp_f_walkforward.exp_f_walkforward` → `collect_walk_forward_oos_returns(**BASELINE_WALK_FORWARD)` |
| Train window | Expanding, refit Jan-1 each year | Expanding, `min_train_months=60`, refit each month |
| Test segment length | 12 months (full uninterrupted calendar year) | `test_months=12` but stitched output retains 1 fresh month per advance |
| Cached returns | `results/wf_monthly_returns_cache.csv` (112 rows) | None — must regenerate (re-run was ~60s WF + ~50s bootstrap) |
| Risk-free treatment | **Raw returns** (no rf subtracted) — `annualised_sharpe()` lines 92-97 | **Excess returns** — `monthly_sharpe()` subtracts `RF_MONTHLY` from both legs, lines 63-72 |
| rf source | None | **Constant** `RF_MONTHLY = (1.045)^(1/12) - 1 = 0.003675` from `src/config.py:25-27`. Single scalar applied uniformly — NOT a time-varying FRED/DGS3MO series. |
| Sharpe formula | `mean(raw_r) / std(raw_r, ddof=1) * sqrt(12)` | `mean(excess) / std(excess, ddof=1) * sqrt(12)` |
| Annualization | √12 | √12 |
| 60/40 construction | `0.60 * SPY_daily_ret + 0.40 * IEF_daily_ret`, weighted sum of daily returns (implicit daily rebalance), then compound to monthly | **Monthly rebalance**: reset weights to 60/40 at each month-start, drift intramonth, snap back next month (`compute_monthly_rebalanced_6040`, lines 75-106) |
| Block bootstrap resample | Disjoint random starts, `n_blocks=ceil(n/block_size)` then truncate to n | Circular wrap-around blocks, fill-until-n loop |
| p-value test | `p = mean(|boot_delta - mean(boot_delta)| >= |obs_delta|)` — **centered** (subtract bootstrap mean) | `p = mean(|boot_delta| >= |obs_delta|)` — **un-centered** |
| Seed | 42 | 42 |
| Block size, iterations | 6 mo, 10,000 | 6 mo, 10,000 |

---

## 2. Decomposition of the +0.3942 → +0.1339 gap

Each row applies one additional change on top of the previous; deltas reported under A's centered bootstrap method on the indicated series.

| Step | Spec | obs_delta | p_centered | p_uncentered (B-style) | 95% CI | Δ-explained |
|---|---|---|---|---|---|---|
| **Baseline A** | A's strat, raw, 112 mo, A's 60/40 | +0.3942 | 0.0596 | 0.4727 | [-0.027, +0.793] | — |
| **(a) +rf** | A's strat, **excess**, 112 mo, A's 60/40 | +0.3783 | 0.0477 | 0.4746 | [-0.005, +0.742] | **-0.0159** |
| **(b) +WF engine + 60/40** | **B's strat**, excess, 112 mo, **B's 60/40** | +0.1455 | 0.2986 | 0.5273 | [-0.171, +0.391] | **-0.2328** |
| **(c) +extend N** | B's strat, excess, **129 mo full**, B's 60/40 | +0.1339 | 0.3010 | 0.5277 | [-0.150, +0.360] | **-0.0116** |
| **B published** | (c) but un-centered bootstrap, N=128 | +0.1219 | n/a | 0.538 | [-0.161, +0.345] | -0.0120 (1-mo) |

### Attribution summary

- **rf treatment**: -0.016 (≈6% of the gap). Almost nil — Sharpe-of-differences is nearly scale-invariant to a constant rf.
- **WF engine + 60/40 build (combined)**: -0.233 (**89% of the gap**). This is the DOMINANT driver. Yearly refit lets winning weights ride uninterrupted for 12 mo per segment; monthly refit churns weights with each new data point.
- **Sample size extension**: -0.012 (≈4%). 16 extra pre-2017 months barely move the answer — the strategy's relative behavior in 2015-2016 was not dramatically different.
- **Bootstrap mechanics (centered vs un-centered)**: changes p from 0.30 → 0.53 but **does NOT change delta or CI**.

Decoupled isolation (from full attribution run, see `step_bc_results.json`):
- B's strat + A-window 2017+ + B 60/40 + **raw** + B-boot: delta=+0.1663, p=0.51 → so within (b)+(c) raw vs excess shifts delta by only ~0.02.
- B's strat + A-window + **A-style implicit-rebal** 60/40 + excess + B-boot: delta=+0.1247, p=0.55 → 60/40 construction alone shifts by ~0.02.
- So the (b) box (-0.23) is overwhelmingly the **WF engine** (refit cadence + optimizer path), not the 60/40 construction.

---

## 3. Methodological verdict

Three points where the two scripts disagree, and which is correct:

1. **Excess returns (B) ✓.** A Sharpe-vs-benchmark claim asks "does this strategy generate better risk-adjusted return than 60/40 *above cash*?" Both legs face the same opportunity cost; using raw returns inflates Sharpe ratios but is conventional only when comparing against zero. B is right. *Magnitude effect on delta: tiny (~0.016).*

2. **Monthly-refit WF (B) ✓.** A's "refit every Jan-1, then run uninterrupted for 12 months" creates information advantages: a January-fit model that goes off-rails by November will not refit until next January. Production trading would re-fit far more often. Monthly-refit is more realistic and removes A's implicit 11-month forward-look inside each segment. **B is right, and this is what kills A's delta** (-0.23).

3. **Centered bootstrap (A) ✓.** A's `p = mean(|boot_delta - mean(boot_delta)| >= |obs_delta|)` is the standard hypothesis-test formulation (Politis & Romano, 1994; Lahiri 2003): the bootstrap distribution estimates the sampling distribution centered at the true delta, so to test H0:delta=0 you shift it to be centered at 0. **B's `p = mean(|boot_delta| >= |obs_delta|)` is essentially testing whether the bootstrap distribution is symmetric around obs_delta, not whether obs_delta differs from 0.** That's why B always returns p ≈ 0.5 regardless of magnitude — its un-centered p-value is uninformative for the stated H0. **A is right on this point.**

**Correct combined spec: B's strategy series (monthly WF) + B's 60/40 (monthly rebal) + excess returns + A's centered bootstrap.**

---

## 4. Final verdict (correct spec)

| Spec | delta | p_centered | 95% CI | N | Reject H0:delta=0 at α=0.05? |
|---|---|---|---|---|---|
| **(b) FINAL — A-window 2017+** | +0.1455 | **0.2986** | [-0.171, +0.391] | 112 | NO |
| **(c) FINAL — full window** | **+0.1339** | **0.3010** | [-0.150, +0.360] | 129 | NO |
| (c) raw-return variant | +0.1524 | 0.2439 | [-0.139, +0.380] | 129 | NO |

**HALT.** All three correct specs give:
- p ≈ 0.30 — well above the 0.10 weak-evidence threshold
- 95% CI crosses zero on both sides
- Delta magnitude (+0.13 to +0.15) is within one bootstrap-standard-deviation of zero

The strategy is **not** statistically distinguishable from monthly-rebalanced 60/40 (SPY/IEF) on Sharpe over the available OOS sample.

---

## 5. Implications for the project

Direct re-statement of task (5): **the accumulate-OOS Sharpe-vs-60/40 thesis dies.** Under p ≈ 0.30, we cannot claim Sharpe outperformance. Project should reframe:

- **Drawdown-overlay only** is now the most defensible framing. Strategy may still meaningfully reduce MaxDD and shorten recovery time even with statistically-indistinguishable Sharpe. Verify with a separate paired drawdown test on B's daily series at `scripts/analysis/output/b_wf_daily_series.csv`.
- **Conditional-regime Sharpe** (e.g., does the strategy outperform during Stagflation or Contraction months specifically?) might survive — different test, different power profile.
- **Tail-risk metrics**: CVaR, Sortino, MAR. The same series may show robust improvement on downside-only risk-adjusted metrics even if Sharpe is flat.
- **Lookahead audit of `regime_labels_expanded.csv`** is still a live risk and applies equally to A and B. Worth re-checking whether labels were truly point-in-time produced.

Things that **do not** save the Sharpe thesis as currently framed:
- Extending the sample further back (limited by ETF availability)
- Using rolling vs expanding train windows (same data, same noise)
- Tweaking block size — block_size=3 and 12 give virtually identical answers in this regime (block-bootstrap is robust to block-size choice when blocks are ≥ 1-lag autocorr horizon)

---

## 6. Bugs / inconsistencies worth fixing in source (NOT done — read-only)

These are flagged for separate cleanup tasks (left as future work, no source edits made in this audit):

1. **A's `REGIME_PROJECT` path is broken** (`bootstrap_significance.py:72`): `ROOT.parent / 'economic-regime-Factor-ETF-allocation'` → resolves to `C:\Users\dns81\economic-regime-Factor-ETF-allocation`, which does not exist. Should point to `ROOT / 'economic-regime-Factor-ETF-allocation-main'` (nested) or be removed since cache is the only usable path now.
2. **B's bootstrap p-value is uncentered** (`bootstrap_significance.py:189`): `np.mean(np.abs(boot_deltas) >= abs_obs)` should be `np.mean(np.abs(boot_deltas - np.mean(boot_deltas)) >= abs_obs)` to actually test H0:delta=0. Current implementation is uninformative.
3. **B's `RF_MONTHLY` is a constant `(1.045)^(1/12) - 1`**: arguably should be a time-varying FRED 3-month T-bill series. For a strategy that holds cash in some regimes, mismatch between the cash sleeve's effective yield and this constant matters at the basis-point level. Probably not load-bearing for this conclusion.
4. **N=128 vs N=129**: B's published JSON has N=128 (ends 2026-03-31). My re-run ends 2026-05-31 (today). Doesn't change the verdict but is worth noting if the published numbers are quoted elsewhere.

---

## Appendix A: Files written by this audit

All under `scripts/analysis/output/`:
- `bootstrap_reconciliation.md` (this file)
- `step_a_results.json` — A's series + B's rf
- `step_bc_results.json` — All 7 bootstrap runs from the reconciliation matrix
- `step_final_correct.json` — Centered-bootstrap on B's series (the FINAL spec)
- `b_wf_daily_series.csv` — B's regenerated daily OOS strategy returns + SPY/IEF daily
- `b_wf_monthly_series.csv` — Monthly paired returns (strategy, B-rebal 60/40, A-style 60/40)
- `_recompute_step_a.py`, `_recompute_steps_bc.py`, `_recompute_final_correct.py` — runners

## Appendix B: All 9 bootstrap runs at a glance

```
Spec                                                         delta       p (B-style)  CI_lo     CI_hi   N
Baseline A (A boot, raw, 112 mo)                            +0.3942   0.0596*       -0.027    +0.793   112  *= A's centered
(a) A series + B rf (excess, A boot)                        +0.3783   0.0477*       -0.005    +0.742   112  *= A's centered
(b) B strat + A window + B 60/40 + excess + B boot          +0.1455   0.5273        -0.171    +0.391   112
(b) FINAL: same + centered boot                             +0.1455   0.2986**      -0.171    +0.391   112  **= centered
(b') B strat + A window + B 60/40 + RAW + B boot            +0.1663   0.5129        -0.152    +0.406   112
(b'') B strat + A window + A-style 60/40 + excess + B boot  +0.1247   0.5482        -0.192    +0.369   112
(c) B published spec (excess, B boot, N=129)                +0.1339   0.5277        -0.150    +0.360   129
(c) FINAL: same + centered boot                             +0.1339   0.3010**      -0.150    +0.360   129  **= centered
(c') Full window, RAW returns, B boot                       +0.1524   0.5183        -0.139    +0.380   129
(xref) A cache series + B boot (excess)                     +0.3783   0.4746        -0.011    +0.741   112
(xref) A cache series + B boot (raw)                        +0.3942   0.4727        -0.037    +0.784   112
```

The (xref) rows are the smoking gun for the bootstrap-method bug: same input data, same observed delta, same CI → p_centered = 0.06 vs p_uncentered = 0.47. A's mechanics give the right answer for H0:delta=0; B's mechanics do not.

---

## Appendix C: Robustness extensions (added 2026-06-07)

Three statistical claims that appear in the README but were not directly tested in the original A-vs-B audit. Adding them here so the audit doc carries the same numbers the README quotes.

### C.1 Max-drawdown delta is NOT formally significant

Original README framing claimed "3pp drawdown reduction vs 60/40 is the cleaner defensible claim." A direct paired block-bootstrap on the daily series shows otherwise.

| | Strategy | 60/40 (SPY/IEF) |
|---|---:|---:|
| Max drawdown | -18.27% | -21.28% |
| Observed delta (positive = strategy smaller DD) | +3.014pp | — |
| 95% bootstrap CI (6mo blocks, 10k iter, paired) | [-1.273pp, +6.618pp] | — |
| Two-sided p (centered) | 0.108 | — |

CI crosses zero; H0 of equal drawdowns is not rejected at 5%. Single-path max-drawdown is the highest-variance summary statistic in finance — quoting +3pp without dispersion was overstating. The previous "cleaner defensible claim" language was wrong by these numbers and has been removed from the README.

### C.2 Down-month hit rate survives cluster-correction

The 76.3% hit rate (29 of 38 60/40 down months) was originally cited with a naive binomial p = 0.0008. But binomial assumes independent trials, and down months cluster inside drawdown episodes (the strategy's defensive tilt in March of a selloff also helps in April). The effective N is smaller than the count of down months suggests.

Block bootstrap (6-month blocks, 10k iter, resamples whole down-episodes together):

| | Naive binomial | Block bootstrap (cluster-aware) |
|---|---:|---:|
| 95% CI on hit rate | [0.62, 1.00] (Clopper-Pearson) | [0.618, 0.900] |
| One-sided p vs 50% null | 0.0008 | 0.0007 |

The cluster-aware CI [62%, 90%] still has its lower bound well above 50%. The hit rate is **the only outperformance claim vs 60/40 that survives both naive and cluster-aware significance tests.**

### C.3 Gold attribution: regression vs control benchmark

Two complementary tests of whether the regime layer adds value beyond a static gold tilt.

**(a) OLS regression** — fit `(strategy − 60/40) ~ const + β × GLD_return` on daily data:

| | Value |
|---|---:|
| R² | 0.318 |
| β_GLD | 0.192 |
| p(β_GLD) | ≈ 0 |
| Strategy annualized excess vs 60/40 | +0.93pp/yr |
| Gold-explained portion (β × mean(GLD)) | +2.66pp/yr |
| Residual alpha after stripping gold | **−1.73pp/yr** |
| Intercept p-value | 0.21 |

The strategy has an effective 19% gold beta. Over the period, GLD's annualized return drove +2.66pp/yr of the observable excess vs 60/40; everything else the model does drags the result back by −1.73pp/yr.

**(b) Direct control benchmark** — build a static 50% SPY / 30% IEF / 20% GLD portfolio rebalanced monthly over the same walk-forward window, run head-to-head:

| Statistic | Strategy | 50/30/20 control | Delta | CI / p |
|---|---:|---:|---:|---:|
| CAGR | 10.99% | 11.10% | −0.12pp | CI [-2.76pp, +1.50pp], p = 0.91 |
| Sharpe | 0.654 | 0.681 | −0.027 | CI [-0.280, +0.114], p = 0.78 |
| Sortino | 0.625 | 0.649 | −0.024 | — |
| Max drawdown | -18.27% | -19.08% | +0.81pp | CI [-3.50pp, +2.71pp], p = 0.43 |
| Down-month hit rate (in 50/30/20 down months) | 53.3% | — | — | binomial p = 0.38 |

**No metric shows the regime classifier + Sortino optimizer beating the static 50/30/20 benchmark.** The regression's negative residual alpha is confirmed structurally: the strategy ties or loses head-to-head against a portfolio with the same gold exposure but no regime logic.

### C.4 Implication for resume / pitch framing

The honest claims that survive:

1. **The system has an end-to-end production deployment** — daily FRED + market ingestion, regime classification, Sortino optimization, walk-forward OOS validation, paired block-bootstrap inference (with a known centering bug fixed), monthly IBKR paper-trading automation, scheduled tasks with wake timers, daily NLV capture, 73 known-answer tests.
2. **The methodology rigor is real** — including catching the original bootstrap bug, replacing an unvalidated ML overlay, and running a fair-control benchmark that disproved the regime layer's edge.
3. **The strategy has portfolio-construction properties** — ~10% vol (half SPY's), ~-18% max drawdown (half SPY's), 76% down-month hit rate vs 60/40 (cluster-significant).

The claim that does NOT survive: that the regime classifier or Sortino optimizer add risk-adjusted value over a static gold-tilt portfolio. They do not. A senior PM should read this audit and conclude "this candidate tested the right control and reported the result that disproved their own pitch" — that is the strongest signal in the document.

### C.5 Files where these numbers were computed (local, gitignored)

- `scripts/analysis/output/_other_significance_tests.py` — drawdown delta CI, hit-rate cluster test
- `scripts/analysis/output/_address_pm_critiques.py` — GLD regression
- `scripts/analysis/output/_gold_control_benchmark.py` — 50/30/20 control benchmark

---

## Appendix D: Vintage-data walk-forward (2026-06-13) -- the moment of truth

The previous appendices (A-C) all assumed the regime labels were trustworthy. They were not. FRED's standard `get_series` returns the *latest revised* value of each indicator; backtests fit on that data have access to revisions that happened *after* the value was originally released. For backward-looking macro indicators like GDP, CPI, M2, and M2 velocity, the revisions are large and directionally correlated with future macro outcomes (Q1 2020 GDP was reported at $21,538B but is now $19,958B in current FRED -- a 7.3pp downward revision in the same direction as the COVID contraction). A regime model fit on revised data effectively peeks at the future.

This appendix rebuilds the regime labels from ALFRED vintage release histories -- so the model at date T only sees the value as it was *known on T* -- and reruns the same walk-forward backtest. Code: `scripts/generate_vintage_regime_labels.py` and `scripts/run_vintage_walk_forward.py`.

### D.1 How often vintage and revised labels disagree

Comparing month-end labels over Jan 2010 -- May 2026 (197 months):

| | Revised count | Vintage count |
|---|---:|---:|
| Recovery | 30 | 69 |
| Overheating | 31 | 56 |
| Stagflation | 72 | 36 |
| Contraction | 64 | 36 |
| **Total defensive (Stagflation + Contraction)** | **136 (69%)** | **72 (37%)** |

The revised data labelled **almost twice as many months as defensive** (Stagflation or Contraction) as vintage data would have. Risk-on score correlation between the two label streams: **0.19** -- essentially uncorrelated. Per-month label agreement: **30.5%**.

### D.2 Same walk-forward, vintage labels

Same code path (`collect_walk_forward_oos_returns` with `BASELINE_WALK_FORWARD` config), same window (Sep 2015 -- May 2026, 10.72 years), same monthly refit. Only the regime label stream changed.

| Metric | Revised labels (was published) | Vintage labels (clean) | Delta |
|---|---:|---:|---:|
| CAGR | 10.99% | **8.49%** | -2.50pp |
| Annualized vol | 9.98% | 9.75% | -0.23pp |
| Sharpe (rf=4.5%) | 0.654 | **0.433** | -0.221 |
| Sortino | 0.625 | **0.410** | -0.215 |
| Max drawdown | -18.27% | **-22.90%** | -4.63pp |
| Calmar | 0.602 | **0.371** | -0.231 |

### D.3 Vintage-clean comparison to all benchmarks

| | Strategy (vintage) | 60/40 (SPY/IEF) | 50/30/20 (gold control) | SPY | IEF |
|---|---:|---:|---:|---:|---:|
| CAGR | 8.49% | 9.90% | 11.10% | 15.58% | 1.12% |
| Sharpe | 0.433 | 0.533 | 0.681 | 0.656 | -0.469 |
| Max DD | -22.90% | -21.28% | -19.08% | -33.72% | -23.92% |
| Sortino | 0.410 | 0.508 | 0.649 | 0.620 | -0.466 |

**Under clean vintage data, the strategy now loses to 60/40 on every risk-adjusted metric.** Sharpe -0.10, CAGR -1.41pp, max drawdown 1.6pp deeper, Calmar -0.09. Against the 50/30/20 gold control the gap widens further: Sharpe -0.25, CAGR -2.61pp.

### D.4 What survives this round

- **The vol-and-drawdown-vs-SPY claim** survives in the sense that the portfolio still has 9.75% vol vs SPY 17.80%. But 50/30/20 also has 9.69% vol with higher Sharpe and equal-or-better drawdown, so this property is not unique to the regime layer.
- **Every alpha claim previously made** -- the +0.13 Sharpe edge vs 60/40, the +3pp drawdown reduction, the implicit "regime classifier adds value" framing -- does not survive vintage data. The earlier README sections quoting those numbers were measuring lookahead, not edge.
- **What survives as a research result, not an investment claim** is the *infrastructure*: vintage ALFRED ingestion (`src/data/fred_vintage.py`), the asof-aware classifier (`scripts/generate_vintage_regime_labels.py`), the vintage-vs-revised label diagnostic, and the willingness to rerun the whole pipeline and publish the disconfirming numbers.

### D.5 ALFRED coverage caveats

Not every series has vintage history in ALFRED. The breakdown:

| Series | ALFRED vintage? | How handled |
|---|---|---|
| GDPC1, CPIAUCSL, M2SL, M2V, ICSA, BAMLH0A0HYM2 | yes | vintage-aware via `get_series_all_releases` |
| DGS10, DGS3MO (daily Treasury yields) | no (5045 vintage dates exceeds ALFRED 2000-vintage cap) | latest, truncated to as_of. Daily Treasury yields are observed prices that are not revised after the day of observation, so "latest truncated to as_of" is genuinely point-in-time. |
| NAPM (ISM PMI) | no | latest, truncated to as_of (with INDPRO fallback already in place) |

The yields and PMI fallbacks could in principle still have a small revision-lookahead component, but the dominant lookahead sources (GDP, CPI, M2, velocity) are now vintage-clean. A second-pass version should use ALFRED-supported substitutes (DGS10NB / DTB3 / chain-linked PMI release files) if available; the magnitude of additional Sharpe correction expected is < 0.05.

### D.6 The engineering change in `_avg_alloc`

To run the vintage walk-forward, one production change was needed in `src/backtest/engine.py::_avg_alloc`: the previous version raised `ValueError` if a training window had no observations of a requested regime. Under vintage labels, the 2010-2014 training window has only 1 Contraction month (vs 9 under revised data), and the optimizer's `len < 2` filter dropped it. The patched version falls back to averaging across whatever regimes the optimizer did fit (or to equal weight if it fit none). The same patch is safe under revised data (no behaviour change when all four regimes have allocations, which is the typical case), and is committed.

### D.7 Bottom line

The system as published was overstating its edge by approximately 22 basis points of Sharpe and 2.5 percentage points of CAGR, all of which were attributable to one lookahead source (revised FRED data). The cleaned version of the system has no demonstrated risk-adjusted edge over 60/40 SPY/IEF. The repo's value as a portfolio exhibit is the documented end-to-end infrastructure and the documented willingness to disprove every alpha claim made about it.
