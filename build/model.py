"""Semantic model for the Retail Network report (writes TMDL)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import col, meas, m_table, calc_table, measure_table, rel, write_model, csv_expressions  # noqa: E402

REPO = "powerbi-retail-network"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "RetailNetwork"

DATE = '''ADDCOLUMNS(
    CALENDAR(DATE(2013, 1, 1), DATE(2015, 7, 31)),
    "Year", YEAR([Date]),
    "Half", "H" & IF(MONTH([Date]) <= 6, 1, 2),
    "Month", FORMAT([Date], "MMM"),
    "Month Number", MONTH([Date]),
    "Year Month", FORMAT([Date], "MMM yy"),
    "Year Month Sort", YEAR([Date]) * 100 + MONTH([Date]),
    "Month Start", DATE(YEAR([Date]), MONTH([Date]), 1),
    "Weekday", FORMAT([Date], "ddd"),
    "Weekday Number", WEEKDAY([Date], 2))'''

FLAGS = '''    Result = Table.AddColumn(Table.AddColumn(Typed,
        "Promo Day", each if [Promo] = 1 then "Promo day" else "Regular day", type text),
        "School Holiday", each if [SchoolHoliday] = 1 then "School holiday" else "School term", type text)'''

TABLES = [
    calc_table("Date", DATE, [
        col("Date", "dateTime", "dd mmm yyyy", key=True), col("Year", "int64", "0"), col("Half"), col("Month", sort="Month Number"),
        col("Month Number", "int64", "0", hidden=True), col("Year Month", sort="Year Month Sort"), col("Year Month Sort", "int64", "0", hidden=True),
        col("Month Start", "dateTime", "mmm yyyy"), col("Weekday", sort="Weekday Number"), col("Weekday Number", "int64", "0", hidden=True)],
        desc="Calendar Jan 2013 - Jul 2015 (the full data range).", is_date=True),
    m_table("Store", "dim_store.csv", [
        col("StoreKey", "int64", "0", hidden=True, key=True), col("StoreName"), col("State", hidden=True), col("StateName", desc="Federal state"),
        col("StateGeo", cat="StateOrProvince", hidden=True), col("Format", desc="Store format (A-D as coded by the retailer)"),
        col("Assortment", desc="Basic, Extra or Extended range"), col("CompetitionDistance", "int64", "#,0", hidden=True, desc="Metres to nearest competitor; -1 = not recorded"),
        col("CompetitionBand", sort="CompetitionBandOrder", desc="Distance to the nearest competitor"), col("CompetitionBandOrder", "int64", "0", hidden=True),
        col("LoyaltyPromo", desc="Takes part in the continuous loyalty promotion")],
        desc="1,115 drugstores across 12 German federal states."),
    m_table("Store Day", "fact_store_day.csv", [
        col("StoreKey", "int64", "0", hidden=True), col("Date", "dateTime", "dd mmm yyyy", hidden=True), col("Sales", "double", "#,0", hidden=True),
        col("Customers", "int64", "#,0", hidden=True), col("Promo", "int64", "0", hidden=True), col("SchoolHoliday", "int64", "0", hidden=True),
        col("StateHoliday", desc="Public holiday type on the day"), col("LoyaltyPromoMonth", "int64", "0", hidden=True),
        dict(col("Promo Day", desc="Store ran its short-term promotion that day"), computed=True),
        dict(col("School Holiday"), computed=True)],
        desc="Fact: one row per open store-day (844,338 rows). Sales in EUR; Customers = transactions (footfall).", extra_steps=FLAGS),
    m_table("Competition Event", "fact_competition_event.csv", [
        col("StoreKey", "int64", "0", hidden=True), col("CompetitorOpened", "dateTime", "mmm yyyy"), col("PreDailySales", "double", "#,0", hidden=True),
        col("PostDailySales", "double", "#,0", hidden=True), col("StoreChange", "double", "0.0%", hidden=True), col("NetworkChange", "double", "0.0%", hidden=True),
        col("ImpactPct", "double", "0.0%", hidden=True), col("ImpactBand", sort="ImpactBandOrder"), col("ImpactBandOrder", "int64", "0", hidden=True)],
        desc="Fact: 131 stores where a competitor opened nearby during the data window. Impact = store change vs network change "
             "(difference in differences), 90 days before vs days 30-150 after."),
    m_table("Forecast", "fact_forecast.csv", [
        col("StoreKey", "int64", "0", hidden=True), col("Date", "dateTime", "dd mmm yyyy", hidden=True), col("Actual", "double", "#,0", hidden=True),
        col("Forecast", "double", "#,0", hidden=True), col("Baseline", "double", "#,0", hidden=True)],
        desc="Fact: 6-week backtest (19 Jun - 31 Jul 2015) of the gradient-boosting forecast and a naive baseline, per open store-day."),
    m_table("Model Metrics", "model_metrics.csv", [col("Metric"), col("Value", "double", "0.####", hidden=True)], hidden=True),
]

FS, FN, FP, FC, FK, FF, FU = "1 Sales", "2 Stores", "3 Promotions", "4 Competition", "5 Calendar", "6 Forecast", "9 Report UI"
SD = "'Store Day'"
MEASURES = [
    meas("Sales", f"SUM({SD}[Sales])", "€#,0", FS, "Net sales, EUR."),
    meas("Customers", f"SUM({SD}[Customers])", "#,0", FS, "Transactions (footfall proxy)."),
    meas("Avg Ticket", "DIVIDE([Sales], [Customers])", "€0.00", FS, "Average transaction value."),
    meas("Store Days", f"COUNTROWS({SD})", "#,0", FS, "Open store-days."),
    meas("Sales per Store Day", "DIVIDE([Sales], [Store Days])", "€#,0", FS, "Average daily sales of an open store: the like-for-like productivity measure."),
    meas("Customers per Store Day", "DIVIDE([Customers], [Store Days])", "#,0", FS),
    meas("Trading Stores", f"DISTINCTCOUNT({SD}[StoreKey])", "#,0", FS),
    meas("Sales LY", "CALCULATE([Sales], SAMEPERIODLASTYEAR('Date'[Date]))", "€#,0", FS),
    meas("LFL Sales", f'''SUMX(CROSSJOIN(VALUES('Store'[StoreKey]), VALUES('Date'[Year Month Sort])),
    VAR _s = [Sales] VAR _ly = [Sales LY] RETURN IF(_s > 0 && _ly > 0, _s))''', "€#,0", FS,
         "Sales of store-months that also traded in the same month last year."),
    meas("LFL Sales LY", f'''SUMX(CROSSJOIN(VALUES('Store'[StoreKey]), VALUES('Date'[Year Month Sort])),
    VAR _s = [Sales] VAR _ly = [Sales LY] RETURN IF(_s > 0 && _ly > 0, _ly))''', "€#,0", FS),
    meas("LFL Growth", "DIVIDE([LFL Sales], [LFL Sales LY]) - 1", "+0.0%;-0.0%;0.0%", FS,
         "Like-for-like sales growth: same stores, same months, this year vs last."),
    meas("Customers LY", "CALCULATE([Customers], SAMEPERIODLASTYEAR('Date'[Date]))", "#,0", FS),
    meas("LFL Footfall Growth", f'''VAR _t = SUMX(CROSSJOIN(VALUES('Store'[StoreKey]), VALUES('Date'[Year Month Sort])),
    VAR _c = [Customers] VAR _ly = [Customers LY] RETURN IF(_c > 0 && _ly > 0, _c))
VAR _l = SUMX(CROSSJOIN(VALUES('Store'[StoreKey]), VALUES('Date'[Year Month Sort])),
    VAR _c = [Customers] VAR _ly = [Customers LY] RETURN IF(_c > 0 && _ly > 0, _ly))
RETURN DIVIDE(_t, _l) - 1''', "+0.0%;-0.0%;0.0%", FS),
    meas("Share of Sales", "DIVIDE([Sales], CALCULATE([Sales], ALLSELECTED('Store')))", "0.0%", FN),
    meas("Store Rank", "IF(HASONEVALUE('Store'[StoreKey]), RANKX(ALLSELECTED('Store'), [Sales per Store Day]))", "0", FN,
         "Rank by sales per open day within the current selection."),
    meas("Top Decile Sales Share", '''VAR _n = DISTINCTCOUNT('Store'[StoreKey])
VAR _top = TOPN(ROUNDUP(_n / 10, 0), VALUES('Store'[StoreKey]), [Sales], DESC)
RETURN DIVIDE(CALCULATE([Sales], _top), [Sales])''', "0%", FN, "Share of sales from the top 10% of stores."),
    meas("Promo Sales Uplift", '''AVERAGEX(FILTER(VALUES('Date'[Weekday Number]), 'Date'[Weekday Number] <= 5),
    DIVIDE(CALCULATE([Sales per Store Day], 'Store Day'[Promo] = 1), CALCULATE([Sales per Store Day], 'Store Day'[Promo] = 0)) - 1)''',
         "+0.0%;-0.0%", FP, "Weekday-matched: promo vs regular days, Monday to Friday only (promotions never run at weekends)."),
    meas("Promo Traffic Uplift", '''AVERAGEX(FILTER(VALUES('Date'[Weekday Number]), 'Date'[Weekday Number] <= 5),
    DIVIDE(CALCULATE([Customers per Store Day], 'Store Day'[Promo] = 1), CALCULATE([Customers per Store Day], 'Store Day'[Promo] = 0)) - 1)''',
         "+0.0%;-0.0%", FP, "Weekday-matched change in transactions on promo days."),
    meas("Promo Ticket Uplift", '''AVERAGEX(FILTER(VALUES('Date'[Weekday Number]), 'Date'[Weekday Number] <= 5),
    DIVIDE(CALCULATE([Avg Ticket], 'Store Day'[Promo] = 1), CALCULATE([Avg Ticket], 'Store Day'[Promo] = 0)) - 1)''',
         "+0.0%;-0.0%", FP, "Weekday-matched change in average ticket on promo days."),
    meas("Promo Day Share", "DIVIDE(CALCULATE([Store Days], 'Store Day'[Promo] = 1), [Store Days])", "0%", FP),
    meas("Promo Sales per Store Day", "CALCULATE([Sales per Store Day], 'Store Day'[Promo] = 1)", "€#,0", FP),
    meas("Regular Sales per Store Day", "CALCULATE([Sales per Store Day], 'Store Day'[Promo] = 0, 'Date'[Weekday Number] <= 5)", "€#,0", FP,
         "Monday-Friday regular days, comparable with promo days."),
    meas("Loyalty Month Effect", '''DIVIDE(CALCULATE([Sales per Store Day], 'Store Day'[LoyaltyPromoMonth] = 1, 'Store'[LoyaltyPromo] = "Loyalty promo"),
       CALCULATE([Sales per Store Day], 'Store Day'[LoyaltyPromoMonth] = 0, 'Store'[LoyaltyPromo] = "Loyalty promo")) - 1''', "+0.0%;-0.0%", FP,
         "Loyalty-promo stores: sales per day in promotion months vs other months (not seasonally adjusted)."),
    meas("Competition Events", "COUNTROWS('Competition Event')", "#,0", FC),
    meas("Median Competition Impact", "MEDIAN('Competition Event'[ImpactPct])", "+0.0%;-0.0%", FC,
         "Median sales impact of a competitor opening (difference in differences vs the network)."),
    meas("Share of Events Losing 10%+", "DIVIDE(CALCULATE(COUNTROWS('Competition Event'), 'Competition Event'[ImpactPct] <= -0.1), [Competition Events])", "0%", FC),
    meas("Avg Pre Sales", "AVERAGE('Competition Event'[PreDailySales])", "€#,0", FC),
    meas("Avg Post Sales", "AVERAGE('Competition Event'[PostDailySales])", "€#,0", FC),
    meas("Avg Impact", "AVERAGE('Competition Event'[ImpactPct])", "+0.0%;-0.0%", FC),
    meas("Near Competitor Share", '''DIVIDE(CALCULATE(DISTINCTCOUNT('Store'[StoreKey]), 'Store'[CompetitionDistance] >= 0, 'Store'[CompetitionDistance] < 500),
       DISTINCTCOUNT('Store'[StoreKey]))''', "0%", FC, "Stores with a competitor under 500 m."),
    meas("School Holiday Uplift", '''DIVIDE(CALCULATE([Sales per Store Day], 'Store Day'[SchoolHoliday] = 1), CALCULATE([Sales per Store Day], 'Store Day'[SchoolHoliday] = 0)) - 1''',
         "+0.0%;-0.0%", FK),
    meas("Sunday Store Days", "CALCULATE([Store Days], 'Date'[Weekday Number] = 7)", "#,0", FK),
    meas("Peak to Trough Weekday", '''VAR _t = ADDCOLUMNS(FILTER(VALUES('Date'[Weekday Number]), 'Date'[Weekday Number] <= 6), "@v", [Customers per Store Day])
RETURN DIVIDE(MAXX(_t, [@v]), MINX(_t, [@v])) - 1''', "+0%", FK, "Busiest vs quietest Monday-Saturday footfall: the staffing swing."),
    meas("FC Actual", "SUM('Forecast'[Actual])", "€#,0", FF),
    meas("FC Forecast", "SUM('Forecast'[Forecast])", "€#,0", FF),
    meas("FC Baseline", "SUM('Forecast'[Baseline])", "€#,0", FF),
    meas("Forecast MAPE", "AVERAGEX('Forecast', DIVIDE(ABS('Forecast'[Forecast] - 'Forecast'[Actual]), 'Forecast'[Actual]))", "0.0%", FF,
         "Mean absolute percentage error per store-day."),
    meas("Baseline MAPE", "AVERAGEX('Forecast', DIVIDE(ABS('Forecast'[Baseline] - 'Forecast'[Actual]), 'Forecast'[Actual]))", "0.0%", FF,
         "Naive baseline: same store, weekday and promo state, median of the previous 8 weeks."),
    meas("Error Reduction", "1 - DIVIDE([Forecast MAPE], [Baseline MAPE])", "0%", FF),
    meas("Forecast RMSPE", "SQRT(AVERAGEX('Forecast', DIVIDE('Forecast'[Actual] - 'Forecast'[Forecast], 'Forecast'[Actual]) ^ 2))", "0.000", FF,
         "Root mean square percentage error (the Kaggle competition metric)."),
    meas("Forecast Bias", "DIVIDE([FC Forecast], [FC Actual]) - 1", "+0.0%;-0.0%", FF),
    meas("Filter Summary", '''VAR _y = IF(ISFILTERED('Date'[Year]), CONCATENATEX(VALUES('Date'[Year]), 'Date'[Year], ", "), "All years")
VAR _s = IF(ISFILTERED('Store'[StateName]), IF(COUNTROWS(VALUES('Store'[StateName])) = 1, VALUES('Store'[StateName]), COUNTROWS(VALUES('Store'[StateName])) & " states"), "All states")
VAR _f = IF(ISFILTERED('Store'[Format]), CONCATENATEX(VALUES('Store'[Format]), 'Store'[Format], ", "), "All formats")
RETURN _y & "  ·  " & _s & "  ·  " & _f''', None, FU, "Breadcrumb of the filter-panel selections."),
]

RELS = [rel("Store Day", "StoreKey", "Store", "StoreKey"), rel("Store Day", "Date", "Date", "Date"),
        rel("Competition Event", "StoreKey", "Store", "StoreKey"), rel("Forecast", "StoreKey", "Store", "StoreKey"), rel("Forecast", "Date", "Date", "Date")]


def build(local=True):
    tables = TABLES + [measure_table("KPI", MEASURES, "All report measures, organised in display folders.")]
    exprs = csv_expressions(REPO, os.path.join(ROOT, "data") + "\\", default="Local" if local else "GitHub")
    write_model(os.path.join(ROOT, f"{NAME}.SemanticModel"), NAME, tables, RELS, exprs,
                desc="Retail Network: 844K open store-days of a 1,115-store German drugstore chain, Jan 2013 - Jul 2015.")


if __name__ == "__main__":
    build(local="--github" not in sys.argv)
    print("model written")
