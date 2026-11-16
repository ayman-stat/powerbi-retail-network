"""Project tests on tiny synthetic frames (the Rossmann files are competition-licensed and never needed in CI)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ds"))
import features as F  # noqa: E402

RNG = np.random.default_rng(7)


def synthetic_panel(n_stores: int = 6, start: str = "2015-01-01", days: int = 200) -> pd.DataFrame:
    """Open store-days with the columns store_encodings and the baselines need."""
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for s in range(1, n_stores + 1):
        level = 5000 + 800 * s
        for d in dates:
            if d.day_of_week == 6:
                continue
            promo = int((d.isocalendar().week % 2 == 0) and d.day_of_week < 5)
            sales = level * (1.3 if promo else 1.0) * (1.1 if d.day_of_week == 0 else 1.0) * RNG.lognormal(0, 0.05)
            rows.append((s, d, round(sales), promo))
    X = pd.DataFrame(rows, columns=["StoreKey", "Date", "Sales", "Promo"])
    X["dow"] = X["Date"].dt.day_of_week
    X["month"] = X["Date"].dt.month
    return X


def test_store_encodings_never_look_past_the_cutoff():
    X = synthetic_panel()
    cut = pd.Timestamp("2015-05-01")
    a = F.store_encodings(X, cut)
    future = X["Date"] >= cut
    X2 = X.copy()
    X2.loc[future, "Sales"] = X2.loc[future, "Sales"] * 50 + 12345      # rewrite the future completely
    b = F.store_encodings(X2, cut)
    pd.testing.assert_frame_equal(a[F.ENCODING_COLS], b[F.ENCODING_COLS])
    assert a.loc[~future, F.ENCODING_COLS].notna().all().all()          # every training row is encoded


def test_recent_level_uses_only_the_eight_weeks_before_the_cutoff():
    X = synthetic_panel(n_stores=1)
    cut = pd.Timestamp("2015-05-01")
    old = X["Date"] < cut - pd.Timedelta(weeks=8)
    X.loc[old, "Sales"] = 1                                               # nothing older than 8 weeks may matter
    enc = F.store_encodings(X, cut)
    window = X[(X["Date"] >= cut - pd.Timedelta(weeks=8)) & (X["Date"] < cut)]
    assert enc["store_recent_mean"].iloc[0] == pytest.approx(np.log1p(window["Sales"]).mean())


def test_loyalty_month_matches_the_etl_row_rule():
    dates = pd.Series(pd.to_datetime(["2014-09-15", "2014-09-15", "2014-01-05", "2013-01-05", "2014-02-10", "2014-06-30"]))
    start = pd.Series(pd.to_datetime(["2014-01-06", "2014-01-06", "2013-03-04", "2013-03-04", pd.NaT, "2014-06-02"]))
    interval = pd.Series(["Mar,Jun,Sept,Dec", "Jan,Apr,Jul,Oct", "Jan,Apr,Jul,Oct", "Jan,Apr,Jul,Oct", np.nan, "Mar,Jun,Sept,Dec"])

    def etl_rule(d, s, iv):          # the ETL's row-wise lambda, verbatim logic
        return int(pd.notna(s) and d >= s and isinstance(iv, str) and d.strftime("%b").replace("Sep", "Sept") in iv)

    expected = [etl_rule(d, s, iv) for d, s, iv in zip(dates, start, interval)]
    assert list(F.loyalty_month(dates, start, interval)) == expected == [1, 0, 1, 0, 0, 1]


def test_holiday_distances_count_days_to_next_and_since_last():
    dates = pd.date_range("2015-01-01", periods=10, freq="D")
    tr = pd.DataFrame({"Store": 1, "Date": dates, "StateHoliday": ["0"] * 10, "SchoolHoliday": 0})
    tr.loc[3, "StateHoliday"] = "a"                                       # 4 January is a holiday
    tr.loc[[7, 8], "SchoolHoliday"] = 1
    out = F.holiday_distances(tr, pd.DataFrame({"Store": [1], "State": ["BE"]})).set_index("Date")
    assert out.loc["2015-01-01", "days_to_hol"] == 3 and out.loc["2015-01-04", "days_to_hol"] == 0
    assert out.loc["2015-01-06", "days_since_hol"] == 2 and out.loc["2015-01-01", "days_since_hol"] == 30   # none yet: capped
    assert out.loc["2015-01-10", "days_to_school"] == 30 and out.loc["2015-01-10", "days_since_school"] == 1


def test_naive_baseline_is_the_recent_median_and_honest_fallback_ignores_the_test_window():
    X = synthetic_panel(n_stores=2)
    cut = pd.Timestamp("2015-06-01")
    train, test = X[X["Date"] < cut], X[X["Date"] >= cut].copy()
    b = F.naive_baseline(train, test, cut)
    recent = train[train["Date"] >= cut - pd.Timedelta(weeks=8)]
    r0 = test.iloc[0]
    exp = recent[(recent.StoreKey == r0.StoreKey) & (recent.dow == r0.dow) & (recent.Promo == r0.Promo)]["Sales"].median()
    assert b[0] == pytest.approx(exp)
    test.loc[test.index[0], "Promo"] = 7                                  # unseen combination -> store median fallback
    honest = F.naive_baseline(train, test, cut, use_test_fallback=False)
    assert honest[0] == pytest.approx(train[train.StoreKey == r0.StoreKey]["Sales"].median())


def test_rmspe_and_mape_definitions():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert F.rmspe(y, p) == pytest.approx(0.1)
    assert F.mape(y, p) == pytest.approx(0.1)


def test_fe_ols_recovers_a_week_level_treatment_and_clusters_widen_the_interval():
    stores, weeks = 60, 80
    s = np.repeat(np.arange(stores), weeks)
    w = np.tile(np.arange(weeks), stores)
    treat_week = RNG.integers(0, 2, weeks)                                # treatment assigned by week, like Rossmann promos
    week_shock = RNG.normal(0, 0.10, weeks)                               # common shock -> errors correlated within week
    y = 0.3 * treat_week[w] + RNG.normal(0, 1, stores)[s] + week_shock[w] + RNG.normal(0, 0.05, stores * weeks)
    df = pd.DataFrame({"y": y, "d": treat_week[w], "store": s, "week": w, "half": w // 40})
    r_week = F.fe_ols(df, "y", "d", ["store", "half"], cluster="week").loc["d"]
    r_iid = F.fe_ols(df, "y", "d", ["store", "half"]).loc["d"]
    assert abs(r_week["coef"] - 0.3) < 4 * r_week["se"]
    assert r_week["se"] > 3 * r_iid["se"]


def test_event_profile_recovers_a_drop_with_flat_leads():
    weeks = pd.date_range("2014-01-06", periods=60, freq="7D")
    t0 = weeks[20] + pd.Timedelta(days=2)                                 # mid-week event
    P = pd.DataFrame(RNG.normal(0, 0.01, (3, 60)), index=[1, 2, 3], columns=weeks)
    P.loc[1, weeks >= weeks[20]] += np.log(0.9)                           # store 1 loses 10% from the event week
    E = F.event_profile(P, [1, 2], [t0, t0])
    eff = F.event_effects(E)
    assert eff.loc[1] == pytest.approx(-0.10, abs=0.01) and abs(eff.loc[2]) < 0.01
    assert E.loc[1, -13:-1].abs().max() < 0.05


def test_fold_windows_tile_the_test_period_and_the_last_one_matches_the_etl_backtest():
    folds = F.fold_windows(["2015-02-13", "2015-03-27", "2015-05-08", "2015-06-19"])
    assert all(e == folds[i + 1][0] for i, (_, e) in enumerate(folds[:-1]))          # contiguous, no overlap
    assert [(e - c).days for c, e in folds] == [42, 42, 42, 43]


def test_closure_spells_find_weekday_runs_and_skip_sundays():
    dates = pd.date_range("2015-03-02", periods=21, freq="D")                       # starts on a Monday
    tr = pd.DataFrame({"Store": 1, "Date": dates, "Open": 1, "StateHoliday": "0"})
    tr.loc[(tr["Date"] >= "2015-03-09") & (tr["Date"] <= "2015-03-17"), "Open"] = 0  # Mon 9 - Tue 17 March incl. a Sunday
    tr.loc[tr["Date"].dt.day_of_week == 6, "Open"] = 0
    sp = F.closure_spells(tr, min_days=5)
    assert len(sp) == 1 and sp.loc[0, "days"] == 8 and sp.loc[0, "start"] == pd.Timestamp("2015-03-09")
