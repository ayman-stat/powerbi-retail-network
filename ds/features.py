"""Retail Network: data loading, feature engineering and estimators shared by the notebooks and the tests.

The forecast features mirror etl/build_data.py exactly (same columns, same row order, same parameters), so the
notebooks can reproduce the published backtest before extending it. The ETL builds its store encodings once, for one
cut-off date; here every encoding takes the cut-off as an argument, so the same code serves a rolling-origin backtest
and a test can prove that nothing on or after the cut-off leaks into the features.

Rossmann data are competition-licensed: functions return frames for analysis, never files for redistribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATES = {"HE": "Hesse", "TH": "Thuringia", "NW": "North Rhine-Westphalia", "BE": "Berlin", "SN": "Saxony", "BY": "Bavaria",
          "BW": "Baden-Wuerttemberg", "HB,NI": "Lower Saxony & Bremen", "HH": "Hamburg", "RP": "Rhineland-Palatinate",
          "SH": "Schleswig-Holstein", "ST": "Saxony-Anhalt"}
HOLIDAY_NAMES = {"0": "None", "a": "Public holiday", "b": "Easter", "c": "Christmas"}
MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}

HOLDOUT_CUT = pd.Timestamp("2015-06-19")          # the ETL's backtest origin (the Kaggle six-week horizon)
DATA_END = pd.Timestamp("2015-07-31")
FEATS = ["dow", "dom", "month", "year", "woy", "doy", "Promo", "SchoolHoliday", "state_hol", "LoyaltyPromoMonth", "StoreType", "Assortment",
         "State", "comp_dist", "comp_months", "Promo2", "store_mean", "store_std", "store_dow_mean", "store_promo_mean", "store_month_mean",
         "store_dow_promo_mean", "recent_shift", "days_to_hol", "days_since_hol", "days_to_school", "days_since_school"]
CAT_FEATS = ["StoreType", "Assortment", "State"]
ENCODING_COLS = ["store_mean", "store_std", "store_dow_mean", "store_promo_mean", "store_month_mean", "store_dow_promo_mean",
                 "store_recent_mean", "recent_shift"]
HGB_PARAMS = dict(max_iter=600, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0, random_state=42)


# ------------------------------------------------------------------ loading
def load_raw(root: Path):
    """train.csv, store.csv and store_states.csv from .raw/rossmann (not redistributable, so not in the repo)."""
    raw = Path(root) / ".raw" / "rossmann"
    tr = pd.read_csv(raw / "train.csv", dtype={"StateHoliday": str}, parse_dates=["Date"])
    st = pd.read_csv(raw / "store.csv")
    ss = pd.read_csv(raw / "store_states.csv")
    return tr, st, ss


def prepare_stores(st: pd.DataFrame, ss: pd.DataFrame) -> pd.DataFrame:
    """Store attributes as in the ETL: state, competitor opening date, loyalty-promo (Promo2) start date."""
    st = st.merge(ss, on="Store", how="left")
    st["StateName"] = st["State"].map(STATES)
    st["Format"] = "Format " + st["StoreType"].str.upper()
    st["AssortmentName"] = st["Assortment"].map({"a": "Basic", "b": "Extra", "c": "Extended"})
    st["CompetitionBand"] = competition_band(st["CompetitionDistance"])
    st["CompOpen"] = pd.to_datetime(dict(year=st["CompetitionOpenSinceYear"], month=st["CompetitionOpenSinceMonth"], day=1), errors="coerce")
    st["Promo2Start"] = pd.to_datetime(st["Promo2SinceYear"].astype("Int64").astype(str) + "-W"
                                       + st["Promo2SinceWeek"].astype("Int64").astype(str).str.zfill(2) + "-1", format="%G-W%V-%u", errors="coerce")
    return st


def competition_band(distance: pd.Series) -> pd.Series:
    bands = pd.cut(distance, [0, 500, 2000, 5000, 15000, 1e9], labels=["Under 0.5 km", "0.5-2 km", "2-5 km", "5-15 km", "15 km+"], right=False)
    return bands.astype(str).where(distance.notna(), "Not recorded")


def loyalty_month(dates: pd.Series, promo2_start: pd.Series, interval: pd.Series) -> np.ndarray:
    """1 when the store's loyalty promotion (Promo2) is running that month: started, and the month is in its interval.
    Vectorised version of the ETL's row-wise rule (month abbreviation, 'Sept' style, contained in PromoInterval)."""
    dates, start = pd.Series(dates).reset_index(drop=True), pd.Series(promo2_start).reset_index(drop=True)
    interval = pd.Series(interval).reset_index(drop=True)
    month = dates.dt.month.to_numpy()
    in_interval = np.zeros(len(dates), bool)
    for iv in interval.dropna().unique():
        months = [m for m, ab in MONTH_ABBR.items() if ab in iv]
        in_interval |= (interval == iv).to_numpy() & np.isin(month, months)
    started = start.notna().to_numpy() & (dates >= start).fillna(False).to_numpy()
    return (started & in_interval).astype(int)


def open_days(tr: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """One row per open store-day with positive sales (the ETL's fact table, same row order)."""
    f = tr[(tr["Open"] == 1) & (tr["Sales"] > 0)].copy()
    f = f.merge(stores[["Store", "Promo2Start", "PromoInterval"]], on="Store", how="left")
    f["LoyaltyPromoMonth"] = loyalty_month(f["Date"], f["Promo2Start"], f["PromoInterval"])
    f["StateHoliday"] = f["StateHoliday"].map(HOLIDAY_NAMES).fillna("None")
    cols = ["Store", "Date", "Sales", "Customers", "Promo", "SchoolHoliday", "StateHoliday", "LoyaltyPromoMonth"]
    return f[cols].rename(columns={"Store": "StoreKey"})


def holiday_distances(tr: pd.DataFrame, ss: pd.DataFrame) -> pd.DataFrame:
    """Days to the next / since the last state holiday and school holiday, per state and date, capped at 30.
    Holidays are published in advance, so looking forward along the calendar is not leakage."""
    cal = tr.merge(ss, on="Store")[["State", "Date", "StateHoliday", "SchoolHoliday"]].drop_duplicates(["State", "Date"])
    cal["is_hol"] = cal["StateHoliday"].astype(str) != "0"
    out = None
    for flag, name in [("is_hol", "hol"), ("SchoolHoliday", "school")]:
        parts = []
        for s_, g in cal.sort_values("Date").groupby("State"):
            days = g.loc[g[flag].astype(bool), "Date"].values
            dts = g["Date"].values
            idx = np.searchsorted(days, dts)
            nxt = np.where(idx < len(days), (days[np.minimum(idx, len(days) - 1)] - dts) / np.timedelta64(1, "D"), 99)
            prv = np.where(idx > 0, (dts - days[np.maximum(idx - 1, 0)]) / np.timedelta64(1, "D"), 99)
            parts.append(pd.DataFrame({"StateCode": s_, "Date": dts, f"days_to_{name}": np.clip(nxt, 0, 30),
                                       f"days_since_{name}": np.clip(prv, 0, 30)}))
        part = pd.concat(parts)
        out = part if out is None else out.merge(part, on=["StateCode", "Date"])
    return out


def base_frame(fact: pd.DataFrame, stores: pd.DataFrame, hol: pd.DataFrame) -> pd.DataFrame:
    """Every cut-off-independent feature: calendar, store attributes, competition age, holiday distances."""
    attrs = stores[["Store", "StoreType", "Assortment", "CompetitionDistance", "CompOpen", "Promo2", "State"]].rename(columns={"Store": "StoreKey"})
    X = fact.merge(attrs, on="StoreKey")
    X["StateCode"] = X["State"]
    X["dow"] = X["Date"].dt.day_of_week
    X["dom"] = X["Date"].dt.day
    X["month"] = X["Date"].dt.month
    X["year"] = X["Date"].dt.year
    X["woy"] = X["Date"].dt.isocalendar().week.astype(int)
    X["doy"] = X["Date"].dt.day_of_year
    X["comp_months"] = ((X["Date"] - X["CompOpen"]).dt.days / 30.4).fillna(-1).clip(-1, 240)
    X["comp_dist"] = X["CompetitionDistance"].fillna(100000)
    X["state_hol"] = (X["StateHoliday"] != "None").astype(int)
    for c in CAT_FEATS:
        X[c] = X[c].astype("category").cat.codes
    X = X.merge(hol, on=["StateCode", "Date"], how="left")
    return X


def store_encodings(X: pd.DataFrame, cut, train_mask=None) -> pd.DataFrame:
    """Store-level target encodings of log sales from rows strictly before `cut` (the forecast origin).
    Returns X with the encoding columns (re)attached; rows after the cut never contribute.
    `train_mask` (optional) restricts the encoding rows further, e.g. to the training rows of a random split."""
    cut = pd.Timestamp(cut)
    X = X.drop(columns=[c for c in ENCODING_COLS if c in X.columns])
    keep = (X["Date"] < cut) if train_mask is None else ((X["Date"] < cut) & np.asarray(train_mask, bool))
    trX = X[keep].assign(ls=lambda d: np.log1p(d["Sales"]))
    enc = trX.groupby("StoreKey").agg(store_mean=("ls", "mean"), store_std=("ls", "std"))
    enc_dow = trX.groupby(["StoreKey", "dow"])["ls"].mean().rename("store_dow_mean")
    enc_promo = trX.groupby(["StoreKey", "Promo"])["ls"].mean().rename("store_promo_mean")
    enc_month = trX.groupby(["StoreKey", "month"])["ls"].mean().rename("store_month_mean")
    enc_dp = trX.groupby(["StoreKey", "dow", "Promo"])["ls"].mean().rename("store_dow_promo_mean")
    recent = trX[trX["Date"] >= cut - pd.Timedelta(weeks=8)].groupby("StoreKey")["ls"].mean().rename("store_recent_mean")
    X = (X.join(enc, on="StoreKey").join(enc_dow, on=["StoreKey", "dow"]).join(enc_promo, on=["StoreKey", "Promo"])
          .join(enc_month, on=["StoreKey", "month"]).join(enc_dp, on=["StoreKey", "dow", "Promo"]).join(recent, on="StoreKey"))
    X["recent_shift"] = X["store_recent_mean"] - X["store_mean"]
    return X


def closure_spells(tr: pd.DataFrame, min_days: int = 5) -> pd.DataFrame:
    """Runs of consecutive closed Monday-Saturday non-holiday days per store (refurbishments and similar)."""
    wk = tr[(tr["Date"].dt.day_of_week < 6) & (tr["StateHoliday"].astype(str) == "0")].sort_values(["Store", "Date"])
    run_id = ((wk["Open"] != wk.groupby("Store")["Open"].shift()) | (wk["Store"] != wk["Store"].shift())).cumsum()
    runs = wk[wk["Open"] == 0].groupby(run_id).agg(Store=("Store", "first"), start=("Date", "min"), end=("Date", "max"), days=("Date", "size"))
    return runs[runs["days"] >= min_days].reset_index(drop=True)


def build_frame(root: Path):
    """Raw files -> (stores, fact, X without encodings). The slow, cut-off-independent part, done once per notebook."""
    tr, st, ss = load_raw(root)
    stores = prepare_stores(st, ss)
    fact = open_days(tr, stores)
    X = base_frame(fact, stores, holiday_distances(tr, ss))
    return tr, stores, fact, X


# ------------------------------------------------------------------ baselines and metrics
def naive_baseline(train: pd.DataFrame, test: pd.DataFrame, cut, use_test_fallback: bool = True) -> np.ndarray:
    """The ETL's naive forecast: same store, weekday and promo state, median of the 8 weeks before the cut-off.
    Combinations unseen in those 8 weeks fall back to the store's median. The ETL takes that median over the TEST window
    (`use_test_fallback=True`, kept for parity); the honest version uses the training window instead."""
    cut = pd.Timestamp(cut)
    hist = train[train["Date"] >= cut - pd.Timedelta(weeks=8)].groupby(["StoreKey", "dow", "Promo"])["Sales"].median().rename("Baseline")
    b = test[["StoreKey", "dow", "Promo"]].join(hist, on=["StoreKey", "dow", "Promo"])["Baseline"]
    src = test if use_test_fallback else train
    fallback = test["StoreKey"].map(src.groupby("StoreKey")["Sales"].median())
    return b.fillna(fallback).to_numpy(float)


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Textbook weekly seasonal naive: each store's sales on the same weekday of its last trading week before the cut-off
    (falls back to the store's last-8-week median for weekdays it did not trade). Ignores promotions on purpose."""
    last = train.sort_values("Date").groupby(["StoreKey", "dow"])["Sales"].last().rename("sn")
    b = test[["StoreKey", "dow"]].join(last, on=["StoreKey", "dow"])["sn"]
    recent = train[train["Date"] >= train["Date"].max() - pd.Timedelta(weeks=8)].groupby("StoreKey")["Sales"].median()
    return b.fillna(test["StoreKey"].map(recent)).to_numpy(float)


def profile_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Store x weekday x promo profile: geometric mean of sales over the whole training history (fallback store x promo)."""
    t = train.assign(ls=np.log1p(train["Sales"]))
    m1 = t.groupby(["StoreKey", "dow", "Promo"])["ls"].mean().rename("m1")
    m2 = t.groupby(["StoreKey", "Promo"])["ls"].mean().rename("m2")
    j = test[["StoreKey", "dow", "Promo"]].join(m1, on=["StoreKey", "dow", "Promo"]).join(m2, on=["StoreKey", "Promo"])
    return np.expm1(j["m1"].fillna(j["m2"]).to_numpy(float))


def rmspe(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / y))


def fold_windows(cut_list, horizon_days: int = 42, last_end=DATA_END):
    """(cut, test_end) pairs, test_end exclusive: each window runs `horizon_days` from its cut, except the last, which runs
    to the end of the data (as the ETL's backtest does: 19 Jun - 31 Jul 2015 is 43 days)."""
    cuts = [pd.Timestamp(c) for c in cut_list]
    stop = pd.Timestamp(last_end) + pd.Timedelta(days=1)
    return [(c, stop if i == len(cuts) - 1 else min(c + pd.Timedelta(days=horizon_days), stop)) for i, c in enumerate(cuts)]


# ------------------------------------------------------------------ causal helpers
def demean(df: pd.DataFrame, cols, fe, tol: float = 1e-8, max_iter: int = 200) -> pd.DataFrame:
    """Remove one or more sets of fixed effects from `cols` by alternating projections (method of Gaure, 2013).
    With a single FE this is the ordinary within transformation."""
    fe = [fe] if isinstance(fe, str) else list(fe)
    out = df[cols].astype(float).copy()
    codes = [pd.factorize(df[f])[0] for f in fe]
    for _ in range(max_iter):
        prev = out.to_numpy().copy()
        for c in codes:
            means = out.groupby(c).transform("mean")
            out = out - means
        if np.max(np.abs(out.to_numpy() - prev)) < tol:
            break
    return out


def fe_ols(df: pd.DataFrame, y: str, x, fe, cluster: str | None = None) -> pd.DataFrame:
    """OLS of y on x after absorbing the fixed effects in `fe`; cluster-robust (CR1) standard errors by `cluster`.
    Returns coef, se, t, 95% CI. Degrees of freedom use the number of clusters (G - 1), the usual conservative choice."""
    x = [x] if isinstance(x, str) else list(x)
    d = demean(df, [y, *x], fe)
    Y, Xm = d[y].to_numpy(), d[x].to_numpy()
    xtx_inv = np.linalg.pinv(Xm.T @ Xm)
    beta = xtx_inv @ Xm.T @ Y
    u = Y - Xm @ beta
    n, k = Xm.shape
    if cluster is None:
        meat = (Xm * u[:, None]).T @ (Xm * u[:, None])
        g = n
        adj = n / (n - k)
    else:
        codes = pd.factorize(df[cluster])[0]
        g = codes.max() + 1
        scores = np.zeros((g, k))
        np.add.at(scores, codes, Xm * u[:, None])
        meat = scores.T @ scores
        adj = g / (g - 1) * (n - 1) / (n - k)
    V = adj * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(V))
    from scipy import stats
    tcrit = stats.t.ppf(0.975, max(g - 1, 1))
    return pd.DataFrame({"coef": beta, "se": se, "t": beta / se, "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se,
                         "clusters": g}, index=x)


def etl_competition_events(fact: pd.DataFrame, stores: pd.DataFrame, open_col: str = "CompOpen") -> pd.DataFrame:
    """The ETL's competitor-opening estimator, one row per event: change in a store's mean daily sales from the 90 days
    before the opening to days 30-150 after, net of the network's change over the same windows."""
    daily = fact.groupby(["StoreKey", "Date"])["Sales"].sum()
    net = fact.groupby("Date")["Sales"].mean()
    rows = []
    for _, r in stores.dropna(subset=[open_col]).iterrows():
        t0 = r[open_col]
        if not (pd.Timestamp("2013-04-01") <= t0 <= pd.Timestamp("2015-01-01")) or r["Store"] not in daily.index.get_level_values(0):
            continue
        s = daily.loc[r["Store"]]
        pre = s[(s.index >= t0 - pd.Timedelta(days=90)) & (s.index < t0)]
        post = s[(s.index >= t0 + pd.Timedelta(days=30)) & (s.index < t0 + pd.Timedelta(days=150))]
        if len(pre) < 40 or len(post) < 60:
            continue
        npre = net[(net.index >= t0 - pd.Timedelta(days=90)) & (net.index < t0)].mean()
        npost = net[(net.index >= t0 + pd.Timedelta(days=30)) & (net.index < t0 + pd.Timedelta(days=150))].mean()
        sc, nc = post.mean() / pre.mean() - 1, npost / npre - 1
        rows.append((r["Store"], t0, pre.mean(), post.mean(), sc, nc, (1 + sc) / (1 + nc) - 1))
    cols = ["StoreKey", "CompetitorOpened", "PreDailySales", "PostDailySales", "StoreChange", "NetworkChange", "ImpactPct"]
    return pd.DataFrame(rows, columns=cols)


def week_start(dates: pd.Series) -> pd.Series:
    return dates - pd.to_timedelta(dates.dt.day_of_week, unit="D")


def relative_weekly_sales(fact: pd.DataFrame, state_of: pd.Series, control_ids, min_days: int = 4) -> pd.DataFrame:
    """Stores x weeks matrix of a store's mean log daily sales (Monday-Saturday) minus the mean of the control stores
    in the same state that week. Same-state controls absorb regional calendars (school and public holidays differ by state).
    Weeks with fewer than `min_days` trading days are left missing."""
    f = fact[fact["Date"].dt.day_of_week < 6]
    w = (f.assign(week=week_start(f["Date"]), ls=np.log(f["Sales"])).groupby(["StoreKey", "week"])
          .agg(ls=("ls", "mean"), days=("ls", "size")).reset_index())
    w = w[w["days"] >= min_days].copy()
    w["State"] = w["StoreKey"].map(state_of)
    ctrl = w[w["StoreKey"].isin(set(control_ids))].groupby(["State", "week"])["ls"].mean().rename("ctrl")
    w = w.join(ctrl, on=["State", "week"])
    return (w["ls"] - w["ctrl"]).set_axis(pd.MultiIndex.from_frame(w[["StoreKey", "week"]])).unstack("week")


def event_profile(P: pd.DataFrame, store_ids, dates, kmin: int = -13, kmax: int = 26, pre=(-13, -1), post=(5, 21),
                  min_pre: int = 8, min_post: int = 12) -> pd.DataFrame:
    """Event-time matrix (one row per event, one column per week relative to the event week) of relative log sales,
    re-based to the event's own pre-period mean. Week 0 is the week containing the event date. Events without
    `min_pre` pre-period weeks and `min_post` post-period weeks are dropped (e.g. the 180 stores missing H2 2014)."""
    weeks = pd.DatetimeIndex(P.columns)
    ks = range(kmin, kmax + 1)
    rows, ids = [], []
    for s, t0 in zip(store_ids, pd.to_datetime(pd.Series(list(dates)))):
        if s not in P.index:
            continue
        k = ((weeks - (t0 - pd.Timedelta(days=t0.day_of_week))).days // 7).to_numpy()
        v = P.loc[s].to_numpy(float)
        m = (k >= kmin) & (k <= kmax) & ~np.isnan(v)
        rows.append(pd.Series(v[m], index=k[m]).reindex(ks).to_numpy())
        ids.append(s)
    E = pd.DataFrame(rows, index=pd.Index(ids, name="StoreKey"), columns=list(ks))
    ok = (E.loc[:, pre[0]:pre[1]].notna().sum(axis=1) >= min_pre) & (E.loc[:, post[0]:post[1]].notna().sum(axis=1) >= min_post)
    E = E[ok]
    return E.sub(E.loc[:, pre[0]:pre[1]].mean(axis=1), axis=0)


def event_effects(E: pd.DataFrame, post=(5, 21)) -> pd.Series:
    """Per-event effect: mean relative log sales over the post window, as a % change."""
    return np.expm1(E.loc[:, post[0]:post[1]].mean(axis=1))


FEATURE_LABELS = {
    "dow": "Weekday", "dom": "Day of month", "month": "Month", "year": "Year", "woy": "Week of year", "doy": "Day of year",
    "Promo": "Promotion day", "SchoolHoliday": "School holiday", "state_hol": "Public holiday", "LoyaltyPromoMonth": "Loyalty promo month",
    "StoreType": "Store format", "Assortment": "Assortment", "State": "State", "comp_dist": "Competitor distance",
    "comp_months": "Competitor age (months)", "Promo2": "In loyalty promo", "store_mean": "Store mean (log)",
    "store_std": "Store volatility (log sd)", "store_dow_mean": "Store x weekday mean", "store_promo_mean": "Store x promo mean",
    "store_month_mean": "Store x month mean", "store_dow_promo_mean": "Store x weekday x promo mean",
    "recent_shift": "Recent level shift (8 wk)", "days_to_hol": "Days to public holiday", "days_since_hol": "Days since public holiday",
    "days_to_school": "Days to school holiday", "days_since_school": "Days since school holiday",
}
