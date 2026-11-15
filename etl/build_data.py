"""ETL + forecast model for the Retail Network report. Run with the project venv:  ..\\.venv\\Scripts\\python etl\\build_data.py

Source: Rossmann Store Sales (Kaggle competition, 2015) - real daily sales and customer counts for 1,115 German drugstores,
Jan 2013 - Jul 2015, with promotions, holidays, store format, assortment and nearest-competitor data.
Store -> federal state mapping from the 3rd-place solution's public repo (entron/entity-embedding-rossmann).

Get the source files (the competition rules do not allow redistribution, so they are not in this repo):
  .raw/rossmann/train.csv, store.csv  <- https://www.kaggle.com/competitions/rossmann-store-sales/data  (accept the rules, download)
  .raw/rossmann/store_states.csv      <- https://raw.githubusercontent.com/entron/entity-embedding-rossmann/master/store_states.csv

Outputs (data/):
  dim_store.csv           store attributes (state, format, assortment, competition, loyalty-promo participation)
  fact_store_day.csv      one row per open store-day: sales, customers, promo, holidays, loyalty-promo flag
  fact_competition_event.csv  difference-in-differences impact of a competitor opening nearby
  fact_forecast.csv       6-week backtest: actual vs gradient-boosting forecast vs naive baseline, per store-day
  model_metrics.csv       forecast accuracy
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW, OUT = os.path.join(ROOT, ".raw", "rossmann"), os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)
metrics = {}


def save(df, name, fmt="%.6g"):
    df.to_csv(os.path.join(OUT, name), index=False, float_format=fmt)
    print(f"  {name:28s} {len(df):>9,} rows")


tr = pd.read_csv(os.path.join(RAW, "train.csv"), dtype={"StateHoliday": str}, parse_dates=["Date"])
st = pd.read_csv(os.path.join(RAW, "store.csv"))
ss = pd.read_csv(os.path.join(RAW, "store_states.csv"))
assert len(tr) == 1_017_209 and tr["Store"].nunique() == 1115
assert tr["Date"].min() == pd.Timestamp("2013-01-01") and tr["Date"].max() == pd.Timestamp("2015-07-31")
metrics["rows_raw"] = len(tr)

STATES = {"HE": "Hesse", "TH": "Thuringia", "NW": "North Rhine-Westphalia", "BE": "Berlin", "SN": "Saxony", "BY": "Bavaria",
          "BW": "Baden-Wuerttemberg", "HB,NI": "Lower Saxony & Bremen", "HH": "Hamburg", "RP": "Rhineland-Palatinate",
          "SH": "Schleswig-Holstein", "ST": "Saxony-Anhalt"}
st = st.merge(ss, on="Store", how="left")
assert st["State"].notna().all()
st["StateName"] = st["State"].map(STATES)
st["Format"] = "Format " + st["StoreType"].str.upper()
st["AssortmentName"] = st["Assortment"].map({"a": "Basic", "b": "Extra", "c": "Extended"})
d = st["CompetitionDistance"]
st["CompetitionBand"] = pd.cut(d, [0, 500, 2000, 5000, 15000, 1e9], labels=["Under 0.5 km", "0.5-2 km", "2-5 km", "5-15 km", "15 km+"],
                               right=False).astype(str).replace("nan", "Not recorded")
st.loc[st["CompetitionDistance"].isna(), "CompetitionBand"] = "Not recorded"
st["CompetitionBandOrder"] = st["CompetitionBand"].map({"Under 0.5 km": 1, "0.5-2 km": 2, "2-5 km": 3, "5-15 km": 4, "15 km+": 5, "Not recorded": 6})
st["CompOpen"] = pd.to_datetime(dict(year=st["CompetitionOpenSinceYear"], month=st["CompetitionOpenSinceMonth"], day=1), errors="coerce")
st["Promo2Start"] = pd.to_datetime(st["Promo2SinceYear"].astype("Int64").astype(str) + "-W" + st["Promo2SinceWeek"].astype("Int64").astype(str).str.zfill(2) + "-1",
                                   format="%G-W%V-%u", errors="coerce")
st["LoyaltyPromo"] = np.where(st["Promo2"] == 1, "Loyalty promo", "No loyalty promo")

# ---- open store-days only (closed days and zero-sales days carry no information about demand)
f = tr[(tr["Open"] == 1) & (tr["Sales"] > 0)].copy()
metrics["open_store_days"] = len(f)
f = f.merge(st[["Store", "Promo2Start", "PromoInterval"]], on="Store", how="left")
mon = f["Date"].dt.strftime("%b").replace({"Sep": "Sept"})
f["LoyaltyPromoMonth"] = ((f["Promo2Start"].notna()) & (f["Date"] >= f["Promo2Start"])
                          & f.apply(lambda r: isinstance(r["PromoInterval"], str) and r["Date"].strftime("%b").replace("Sep", "Sept") in r["PromoInterval"], axis=1)).astype(int)
f["StateHoliday"] = f["StateHoliday"].map({"0": "None", "a": "Public holiday", "b": "Easter", "c": "Christmas"}).fillna("None")
fact = f[["Store", "Date", "Sales", "Customers", "Promo", "SchoolHoliday", "StateHoliday", "LoyaltyPromoMonth"]].rename(columns={"Store": "StoreKey"})
gap = tr[(tr["Date"] >= "2014-07-01") & (tr["Date"] <= "2014-12-31")].groupby("Store").size()
metrics["stores_with_h2_2014_gap"] = int(1115 - len(gap))

# ---- competitor-opening event study (difference in differences vs stores without a new competitor)
daily = fact.groupby(["StoreKey", "Date"])["Sales"].sum()
net = fact.groupby("Date")["Sales"].mean()                          # network average sales per open store-day
ev_rows = []
for _, r in st.dropna(subset=["CompOpen"]).iterrows():
    t0 = r["CompOpen"]
    if not (pd.Timestamp("2013-04-01") <= t0 <= pd.Timestamp("2015-01-01")):
        continue
    s = daily.loc[r["Store"]]
    pre = s[(s.index >= t0 - pd.Timedelta(days=90)) & (s.index < t0)]
    post = s[(s.index >= t0 + pd.Timedelta(days=30)) & (s.index < t0 + pd.Timedelta(days=150))]
    if len(pre) < 40 or len(post) < 60:
        continue
    npre = net[(net.index >= t0 - pd.Timedelta(days=90)) & (net.index < t0)].mean()
    npost = net[(net.index >= t0 + pd.Timedelta(days=30)) & (net.index < t0 + pd.Timedelta(days=150))].mean()
    store_chg = post.mean() / pre.mean() - 1
    net_chg = npost / npre - 1
    ev_rows.append((r["Store"], t0, round(pre.mean(), 1), round(post.mean(), 1), round(store_chg, 4), round(net_chg, 4),
                    round((1 + store_chg) / (1 + net_chg) - 1, 4)))
events = pd.DataFrame(ev_rows, columns=["StoreKey", "CompetitorOpened", "PreDailySales", "PostDailySales", "StoreChange", "NetworkChange", "ImpactPct"])
events["ImpactBand"] = pd.cut(events["ImpactPct"], [-10, -0.2, -0.1, 0, 0.1, 10], labels=["Below -20%", "-20% to -10%", "-10% to 0%", "0% to +10%", "Above +10%"]).astype(str)
events["ImpactBandOrder"] = events["ImpactBand"].map({"Below -20%": 1, "-20% to -10%": 2, "-10% to 0%": 3, "0% to +10%": 4, "Above +10%": 5})
metrics["competition_events"] = len(events)
metrics["competition_median_impact"] = round(events["ImpactPct"].median(), 4)

# ---- forecast: gradient boosting on log sales, 6-week backtest (the Kaggle horizon)
cut = pd.Timestamp("2015-06-19")
X = fact.merge(st[["Store", "StoreType", "Assortment", "CompetitionDistance", "CompOpen", "Promo2", "State"]].rename(columns={"Store": "StoreKey"}), on="StoreKey")
X["dow"] = X["Date"].dt.dayofweek
X["dom"] = X["Date"].dt.day
X["month"] = X["Date"].dt.month
X["year"] = X["Date"].dt.year
X["woy"] = X["Date"].dt.isocalendar().week.astype(int)
X["doy"] = X["Date"].dt.dayofyear
X["comp_months"] = ((X["Date"] - X["CompOpen"]).dt.days / 30.4).fillna(-1).clip(-1, 240)
X["comp_dist"] = X["CompetitionDistance"].fillna(100000)
X["state_hol"] = (X["StateHoliday"] != "None").astype(int)
for c in ["StoreType", "Assortment", "State"]:
    X[c] = X[c].astype("category").cat.codes
# store-level target encodings from the training window only (no leakage)
train_mask = X["Date"] < cut
enc = X[train_mask].assign(ls=np.log1p(X.loc[train_mask, "Sales"])).groupby("StoreKey").agg(store_mean=("ls", "mean"), store_std=("ls", "std"))
enc_dow = X[train_mask].assign(ls=np.log1p(X.loc[train_mask, "Sales"])).groupby(["StoreKey", "dow"])["ls"].mean().rename("store_dow_mean")
enc_promo = X[train_mask].assign(ls=np.log1p(X.loc[train_mask, "Sales"])).groupby(["StoreKey", "Promo"])["ls"].mean().rename("store_promo_mean")
LS = np.log1p(X["Sales"])
trX = X[train_mask].assign(ls=LS[train_mask])
enc_month = trX.groupby(["StoreKey", "month"])["ls"].mean().rename("store_month_mean")
enc_dp = trX.groupby(["StoreKey", "dow", "Promo"])["ls"].mean().rename("store_dow_promo_mean")
recent = trX[trX["Date"] >= cut - pd.Timedelta(weeks=8)].groupby("StoreKey")["ls"].mean().rename("store_recent_mean")
X = (X.join(enc, on="StoreKey").join(enc_dow, on=["StoreKey", "dow"]).join(enc_promo, on=["StoreKey", "Promo"])
      .join(enc_month, on=["StoreKey", "month"]).join(enc_dp, on=["StoreKey", "dow", "Promo"]).join(recent, on="StoreKey"))
X["recent_shift"] = X["store_recent_mean"] - X["store_mean"]
# distance to the nearest state holiday and school holiday (per state) - demand surges before closures
cal = tr.merge(ss, on="Store")[["State", "Date", "StateHoliday", "SchoolHoliday"]].drop_duplicates(["State", "Date"])
cal["is_hol"] = cal["StateHoliday"].astype(str) != "0"
for flag, name in [("is_hol", "hol"), ("SchoolHoliday", "school")]:
    parts = []
    for s_, g in cal.sort_values("Date").groupby("State"):
        days = g.loc[g[flag].astype(bool), "Date"].values
        dts = g["Date"].values
        idx = np.searchsorted(days, dts)
        nxt = np.where(idx < len(days), (days[np.minimum(idx, len(days) - 1)] - dts) / np.timedelta64(1, "D"), 99)
        prv = np.where(idx > 0, (dts - days[np.maximum(idx - 1, 0)]) / np.timedelta64(1, "D"), 99)
        parts.append(pd.DataFrame({"State_": s_, "Date": dts, f"days_to_{name}": np.clip(nxt, 0, 30), f"days_since_{name}": np.clip(prv, 0, 30)}))
    X = X.merge(pd.concat(parts).rename(columns={"State_": "StateCode"}),
                left_on=[X["StoreKey"].map(st.set_index("Store")["State"]), "Date"], right_on=["StateCode", "Date"], how="left").drop(columns=["StateCode"])
    X = X.loc[:, ~X.columns.str.startswith("key_")]
train_mask = X["Date"] < cut
FEATS = ["dow", "dom", "month", "year", "woy", "doy", "Promo", "SchoolHoliday", "state_hol", "LoyaltyPromoMonth", "StoreType", "Assortment",
         "State", "comp_dist", "comp_months", "Promo2", "store_mean", "store_std", "store_dow_mean", "store_promo_mean", "store_month_mean",
         "store_dow_promo_mean", "recent_shift", "days_to_hol", "days_since_hol", "days_to_school", "days_since_school"]
trn, tst = X[train_mask], X[~train_mask]
model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0,
                                      categorical_features=[FEATS.index(c) for c in ["StoreType", "Assortment", "State"]], random_state=42)
model.fit(trn[FEATS], np.log1p(trn["Sales"]))
pred = np.expm1(model.predict(tst[FEATS]))
# naive baseline: same store, same weekday, same promo state, median of the previous 8 weeks
hist = trn[trn["Date"] >= cut - pd.Timedelta(weeks=8)].groupby(["StoreKey", "dow", "Promo"])["Sales"].median().rename("Baseline")
tst = tst.join(hist, on=["StoreKey", "dow", "Promo"])
tst["Baseline"] = tst["Baseline"].fillna(tst.groupby("StoreKey")["Sales"].transform("median"))


def rmspe(y, p):
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


metrics["forecast_horizon_days"] = int((X["Date"].max() - cut).days + 1)
metrics["forecast_rmspe_model"] = round(rmspe(tst["Sales"].values, pred), 4)
metrics["forecast_rmspe_baseline"] = round(rmspe(tst["Sales"].values, tst["Baseline"].values), 4)
metrics["forecast_mape_model"] = round(float(np.mean(np.abs(tst["Sales"].values - pred) / tst["Sales"].values)), 4)
metrics["forecast_mape_baseline"] = round(float(np.mean(np.abs(tst["Sales"].values - tst["Baseline"].values) / tst["Sales"].values)), 4)
net_err = abs(pred.sum() / tst["Sales"].sum() - 1)
metrics["forecast_network_bias"] = round(float(pred.sum() / tst["Sales"].sum() - 1), 4)
print("forecast:", {k: v for k, v in metrics.items() if k.startswith("forecast")})
fc = tst[["StoreKey", "Date", "Sales"]].rename(columns={"Sales": "Actual"})
fc["Forecast"] = np.round(pred, 0)
fc["Baseline"] = np.round(tst["Baseline"].values, 0)

# ---- write
dim = st.rename(columns={"Store": "StoreKey"})[["StoreKey", "State", "StateName", "Format", "AssortmentName", "CompetitionDistance", "CompetitionBand",
                                               "CompetitionBandOrder", "LoyaltyPromo"]].rename(columns={"AssortmentName": "Assortment"})
dim["StoreName"] = "Store " + dim["StoreKey"].astype(str).str.zfill(4)
dim["StateGeo"] = dim["StateName"].replace({"Lower Saxony & Bremen": "Lower Saxony", "Baden-Wuerttemberg": "Baden-Wurttemberg"}) + ", Germany"
dim["CompetitionDistance"] = dim["CompetitionDistance"].fillna(-1).astype(int)
save(dim, "dim_store.csv")
save(fact, "fact_store_day.csv")
save(events, "fact_competition_event.csv")
save(fc, "fact_forecast.csv")
save(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]), "model_metrics.csv", fmt="%.10g")
print(json.dumps(metrics, indent=2))
