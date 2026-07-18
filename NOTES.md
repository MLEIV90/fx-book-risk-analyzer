# Statistical assumptions and applicability conditions

This project runs several statistical tests and estimators whose validity
depends on conditions that don't always hold on ~2 years of free daily FX
data. This note states them explicitly, and how the project stands relative
to each — consistent with the "declared, not hidden" approach used
throughout (see DESIGN.md for the data-sourcing side of the same principle).

## Backtests (Kupiec, Christoffersen)

- **Asymptotic chi-squared validity.** Both tests' classical p-value relies
  on the likelihood-ratio statistic being approximately chi-squared (1 df),
  which holds asymptotically as the number of exceptions grows. At 99%
  confidence over a ~2-year/250-day window, the expected exception count is
  small (roughly 2-5 for a single test, 10-20 for a longer rolling backtest)
  — often too few for the approximation to be reliable. `kupiec_backtest`,
  `christoffersen_independence` and `rolling_backtest` (which reuses Kupiec)
  therefore all also report an EXACT Monte Carlo p-value (Dufour 2006,
  `p_value_mc`), which does not rely on the asymptotic approximation.
  `passed` / `independent` are decided on the Monte Carlo value;
  `mc_agrees_with_asymptotic` flags when the two disagree — exactly when the
  asymptotic shortcut would have been misleading.
- **Low power with few exceptions.** Even with an exact p-value, a test with
  only a handful of exceptions has LOW STATISTICAL POWER: it cannot reliably
  distinguish a correctly-calibrated model from a moderately miscalibrated
  one. A "pass" at 99% confidence over a few hundred to a couple of thousand
  days is correspondingly WEAK evidence of correctness, not proof — it means
  the data did not contradict the model, not that the model is right.
- **Christoffersen is first-order Markov.** The independence test only checks
  whether an exception TODAY predicts one TOMORROW (a first-order Markov
  chain on the hit indicator). It is blind to higher-order or longer-range
  clustering (e.g. exceptions bunching within a 5-day window without
  consecutive-day repeats). A "pass" rules out simple day-to-day clustering,
  not every form of clustering.

## VaR methods

- **Historical VaR** makes NO distributional assumption — it reads the
  empirical percentile of realised P&L. Its validity instead rests on the
  TRAILING WINDOW being representative of the risk horizon: if the recent
  window was unusually calm (or turbulent), the VaR inherits that bias. It is
  also sensitive to WINDOW LENGTH: a short window reacts fast but is noisy; a
  long window is smoother but stale. This project uses the available
  ~2-year yfinance history — a declared, not tuned, choice.
- **Parametric (normal) VaR** assumes returns are jointly NORMALLY
  distributed. FX returns are empirically fat-tailed (more extreme moves
  than a normal predicts), so this method systematically UNDERSTATES tail
  risk — exactly why the Student-t and historical methods are also reported
  side by side, rather than relied on alone.
- **Monte Carlo VaR** assumes the FITTED data-generating process (here, a
  multivariate normal calibrated to the sample covariance) is a good model of
  the return distribution; with a normal engine it inherits the same
  fat-tail understatement as the parametric method, and additionally carries
  SIMULATION (sampling) error from a finite number of draws.
  `var_montecarlo_stability` quantifies that seed-to-seed sampling error
  (mean, spread, and relative spread across independent seeds), so it is
  reported, not assumed away.

## Estimation risk (backtests)

Every backtest here (`kupiec_backtest`, `christoffersen_independence`,
`rolling_backtest`) treats the VaR figure being tested as if it were the
TRUE, known model — when in fact it is itself ESTIMATED from a finite sample
(a covariance matrix, a trailing window, a fitted Student-t). These are
practitioner-standard backtests, not the "conditional coverage under known
parameters" ideal that some academic treatments assume; they do NOT correct
for that extra layer of estimation uncertainty. A "pass" is evidence the
ESTIMATED VaR process performed adequately out-of-sample, not a formal
guarantee about a hypothetical true underlying model.
