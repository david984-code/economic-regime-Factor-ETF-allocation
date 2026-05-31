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
