# Model card: six-week store-day sales forecast (and the causal estimates beside it)

## 1. Forecast model

| | |
|---|---|
| **Model** | `HistGradientBoostingRegressor` on log(1 + sales); 600 iterations, learning rate 0.08, 63 leaves, min 50 rows per leaf, L2 1.0, native categorical handling for format, assortment and state (`etl/build_data.py`, reproduced in `ds/features.py`) |
| **Intended use** | Daily sales per store for the next 1-43 days, made on a Friday, to set staff rosters and stock orders; network totals for planning |
| **Not for** | Pricing or promotion decisions (it is a forecast, not a causal model: see section 2); days a store is closed; stores with no history |
| **Owner** | Data science, retail operations analytics. Review monthly with store operations |
| **Data** | Rossmann Store Sales (Kaggle, 2015): 1,115 German drugstores, open store-days with sales > 0, Jan 2013 - Jul 2015 (844,338 rows). 180 stores are missing Jul-Dec 2014 in the source |
| **Target** | Daily net sales (EUR), modelled on the log scale; forecasts are back-transformed with `expm1` |

### Features (27, in six families)

* **Calendar:** weekday, day of month, month, year, week of year, day of year.
* **Promotion and holidays:** promotion flag, school holiday, public holiday, loyalty-promotion month; days to the next and since the last public and school holiday (capped at 30).
* **Store attributes:** format, assortment, state, competitor distance, competitor age in months, loyalty-promotion membership.
* **Store encodings of log sales, rebuilt from data before each forecast origin only:**
  * the store's mean and standard deviation;
  * its means by weekday, by promotion, by month and by weekday x promotion;
  * its last 8 weeks against its long-run mean (`recent_shift`).
* **No lags:** a six-week horizon makes yesterday's sales unavailable, and the recent-level feature carries that memory instead.
* **Leakage guard:** `tests/test_features.py` rewrites every sale after the origin and checks that no feature changes.

### Validation design

A rolling-origin backtest with four six-week windows in 2015. Each fold trains on all history before its origin and
rebuilds the encodings from that history only. The windows are 13 Feb, 27 Mar and 8 May (42 days each) and 19 Jun - 31 Jul
(43 days, the published backtest). They include Easter, 1 May, Ascension and Whit Monday. Random splits are not used; on
the same stores and days, a random split shows about 15% less error (MAPE 6.2% vs 7.3%) than an honest forecast achieves.

### Performance

| Fold | RMSPE | MAPE | WAPE | Network bias | Naive rule MAPE |
|---|---|---|---|---|---|
| 1 (13 Feb) | 0.164 | 8.2% | 7.8% | −3.3% | 11.2% |
| 2 (27 Mar) | 0.104 | 8.0% | 8.1% | −3.8% | 12.2% |
| 3 (8 May) | 0.103 | 7.7% | 7.5% | −2.4% | 10.3% |
| Holdout (19 Jun, published) | **0.110** | **7.9%** | 7.7% | **−0.6%** | 11.0% |

* **Uncertainty.** The champion's RMSPE advantage over the naive rule on the holdout is 0.038 (95% store-level block
  bootstrap CI 0.035-0.041).
* **Challenger.** An Optuna-tuned LightGBM was not better on every fold and was worse on the holdout (ΔRMSPE −0.0012,
  CI −0.0019 to −0.0003), so the champion is kept.
* **RMSPE is fragile.** In fold 1, ten store-days carry 28% of the squared percentage error, so MAPE and WAPE are the
  decision metrics.

### Calibration, intervals and bias

* **Systematic under-forecast.** Network bias is −0.6% to −3.8% across folds: trees cannot extrapolate the roughly 3%
  growth. A previous-fold scaling factor did not fix it reliably.
* **Intervals.** LightGBM quantile models (P10/P90) cover 74.6% of holdout store-days against a target of 80%, with
  symmetric misses. Residual-quantile intervals from folds 1-3 cover 76.2% but are lopsided. Treat the bands as 75% bands
  until conformal calibration is added.

### Segment performance (four folds pooled)

| Segment | MAPE | Bias |
|---|---|---|
| Format A / B / C / D | 7.9% / 8.0% / 8.2% / 7.9% | −2.2% to −3.8% |
| Promotion day / regular day | 7.8% / 8.0% | −2.3% / −2.8% |
| Within 2 days of a public holiday | 9.0% | −5.2% |
| Week 1 / weeks 2-6 | 8.9% / 7.4-8.0% | |

The worst 50 stores carry 23% of the squared error. Two causes are over-represented among them: a closure of five or more
weekdays (a refurbishment) near the test window, 12% vs 1% of all stores, and a competitor opening in 2014-15, 18% vs 10%.

### Explainability

* **SHAP** (1,500 holdout store-days, log scale). The store x weekday x promo mean carries 37% of mean |SHAP|, followed
  by public holiday, store x month mean, store mean, week of year and year.
* **Permutation importance** (RMSPE) agrees on the top encodings. It ranks `recent_shift` 4th where SHAP ranks it 16th:
  the feature is small on average but decisive for stores whose level moved.

### Limitations

* No backtest fold covers December.
* There is no trend term, hence the under-forecast.
* Refurbishments and competitor openings are not inputs.
* The staffing value (about 63,000 mis-planned staff-hours avoided per six weeks against the naive rule) rests on an
  assumed labour standard of EUR 125 per staff-hour; the source has no rostering data.
* Competition-licensed data, so the model cannot be shared with its training data.

### Ethics and fairness

The model forecasts store demand, not people. The fairness question is operational: do some store groups get
systematically worse rosters? Error is even across formats and promotion states. It is worse around public holidays,
which differ by state, so holiday weeks deserve manual review in the states with the most holidays.

### Monitoring and retraining

* **Weekly:** re-forecast with the latest actuals.
* **Track** MAPE, WAPE, network bias and P10-P90 coverage per store and per state.
* **Retrain** monthly, or sooner if network bias stays outside ±2% for two consecutive weeks or MAPE exceeds 9% for two weeks.
* **Re-run the four-fold backtest** before any change to features or parameters, and apply the same champion/challenger rule.

## 2. Causal estimates (not a deployed model; notebook 03)

| Estimate | Value | Method and checks |
|---|---|---|
| Promotion lift, daily sales | +39.1% (95% CI +36.9% to +41.4%) | Store, year-month, weekday and day-of-month fixed effects plus holiday controls, on Mon-Fri store-days. Errors are clustered by ISO week (the assignment unit); iid errors would be 20× too small. Matches statsmodels on a subsample |
| Traffic / ticket | +20.8% / +15.2% | Same model on log customers and log sales per customer |
| Week after a promotion | +1.5% (CI −2.9% to +6.0%) | No detectable pull-forward |
| Loyalty promotion month | −0.3% (CI −0.5% to 0.0%) | Store and date fixed effects; the published −1.9% is seasonality |
| Competitor opening | mean −7.9% (CI −9.8% to −6.0%), median −6.2% | Event study on 131 openings against same-state comparison stores; leads flat (max 0.8%); placebo 95% range −0.7% to +0.7% (p = 0.003); −16% under 500 m to under 2% beyond 5 km |

**Assumptions.**
* Promotions: after the calendar controls, promotion weeks are as good as random. They alternate network-wide, so this
  is close to a natural experiment.
* Competitors: no selection on future trends. Where competitors choose to open, the month-only timing and the
  nearest-competitor-only record all bias the estimate towards zero.
