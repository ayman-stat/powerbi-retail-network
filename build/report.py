"""Report definition (PBIR) for the Retail Network report.

Identity: RETAIL - warm white, orange for the data in focus, charcoal for context, underline-tab navigation, soft card shadows,
KPI tiles with an orange underline. Every page has a slide-in filter panel (button + bookmarks) holding that page's slicers,
and a breadcrumb that states the active filters.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import Report, write_pbip, C, M, proj, b, i, n, s, props, idprops, color  # noqa: E402
import design as D  # noqa: E402
from design import (S, GUT, W, use, theme, chrome, kpi_band, bar, combo, line, matrix, table, button_slicer, dropdown, filter_panel,  # noqa: E402
                    azure_map, about_page, P, _title, fill_for)

use(D.RETAIL)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "RetailNetwork"
TITLE = "Retail Network Performance"
SRC = "Rossmann Store Sales (Kaggle competition, 2015): daily sales and transactions of 1,115 German drugstores, Jan 2013 - Jul 2015. Non-commercial portfolio use."
K = lambda m: M("KPI", m)  # noqa: E731
YM = (C("Date", "Year Month"), "Month")
YMS = C("Date", "Year Month Sort")
NP = 7
TW = 1300   # title width (leaves room for the filter button and breadcrumb)


# ---- filter-panel slicer factories: each returns the height it used
def f_buttons(field, title, cols, rows=1, selected=None, h=76):
    def draw(page, x, y, w, parent):
        button_slicer(page, "fp-" + title, x, y, w, h, field, title, selected=selected, columns=cols, rows=rows, parent=parent, hidden=not os.environ.get("PANEL_DEBUG"))
        return h
    return draw


def f_dropdown(field, title):
    def draw(page, x, y, w, parent):
        dropdown(page, "fp-" + title, x, y, w, 76, field, title, parent=parent, hidden=not os.environ.get("PANEL_DEBUG"))
        return 76
    return draw


YEAR = lambda sel=None: f_buttons(C("Date", "Year"), "Year", 3, selected=sel)  # noqa: E731
STATE = f_dropdown(C("Store", "StateName"), "Federal state")
FORMAT = f_buttons(C("Store", "Format"), "Store format", 2, rows=2, h=120)
ASSORT = f_buttons(C("Store", "Assortment"), "Assortment", 3)
LOYAL = f_buttons(C("Store", "LoyaltyPromo"), "Loyalty promotion", 2)
COMPBAND = f_dropdown(C("Store", "CompetitionBand"), "Nearest competitor")
SCHOOL = f_buttons(C("Store Day", "School Holiday"), "School calendar", 2)


def breadcrumb(p):
    p.add("crumb", "cardVisual", S.x0 + S.cw - 560, S.title_y + 60, 560, 28, query={"Data": [proj(K("Filter Summary"))]},
          objects={"value": idprops(fontSize=n(11), fontColor=color(S.muted), fontFamily=s(S.font), horizontalAlignment=s("right")),
                   "label": idprops(show=b(False)), "outline": idprops(show=b(False)), "fillCustom": [{"properties": {"show": b(False)}}],
                   "padding": idprops(leftMargin=i(0), rightMargin=i(0), topMargin=i(0), bottomMargin=i(0))},
          vco=dict(D.NOBOX), alt="Active filters")


def page_frame(r, key, display, title, subtitle, no, slicers):
    p = r.page(key, display, bg=S.page)
    chrome(p, TITLE, title, subtitle, f"{no} / {NP}", SRC, title_w=TW)
    breadcrumb(p)
    return p, slicers


def finish(r, p, slicers):
    filter_panel(r, p, slicers, label="Filters")


def build():
    r = Report(os.path.join(ROOT, f"{NAME}.Report"), NAME, f"{NAME}.SemanticModel", theme())
    r.n_pages = NP
    x0, cw, by, bh = S.x0, S.cw, S.body_y, S.body_h
    L = 1232
    R = cw - L - GUT

    # ------------------------------------------------------------------ 1 Network overview
    p, sl = page_frame(r, "overview", "Network overview", "2015 sales up 3.3% like-for-like, driven by bigger baskets, not more shoppers", "Jan-Jul 2015 vs Jan-Jul 2014, same stores and months  ·  like-for-like footfall -0.4%  ·  open Filters to change year, state or format", 1, [YEAR([2015]), STATE, FORMAT, ASSORT])
    kpi_band(p, "kpi", [(K("Sales"), "Net sales", 1000000), (K("Customers"), "Transactions", 1000000), (K("Avg Ticket"), "Average ticket"),
                        (K("Sales per Store Day"), "Sales per store-day"), (K("LFL Growth"), "Like-for-like growth"), (K("Trading Stores"), "Trading stores")])
    th = 330
    line(p, "trend", x0, by, L, th, YM, [(K("Sales"), "This year"), (K("Sales LY"), "Last year")], "Monthly sales against last year",
         "Orange: selected year  ·  grey: same months last year", colors=(S.primary, S.grey), sort_cat=YMS, area=False, markers=True)
    matrix(p, "states", x0, by + th + GUT, L, bh - th - GUT, [(C("Store", "StateName"), "Federal state")],
           [(K("Trading Stores"), "Stores"), (K("Sales"), "Sales"), (K("Sales per Store Day"), "Sales / store-day"), (K("Avg Ticket"), "Avg ticket"),
            (K("LFL Growth"), "LFL growth"), (K("LFL Footfall Growth"), "LFL footfall")],
           "State scorecard", "Data bars: sales  ·  growth shaded red (decline) to green (growth), centred on zero",
           gradients=[(K("LFL Growth"), S.soft_neg, S.soft_pos, S.panel, 0), (K("LFL Footfall Growth"), S.soft_neg, S.soft_pos, S.panel, 0)],
           databars=[K("Sales")], sort=[(K("Sales"), "Descending")], font=11)
    mh = 400
    azure_map(p, "map", x0 + L + GUT, by, R, mh, C("Store", "StateGeo"), (K("Sales"), "Sales"), "Where the sales are", "Bubble size: net sales by federal state")
    bar(p, "fmt", x0 + L + GUT, by + mh + GUT, R, bh - mh - GUT, (C("Store", "Format"), "Format"), (K("Sales per Store Day"), "Sales / store-day"),
        "Productivity by store format", "Average daily sales of an open store, EUR")
    finish(r, p, sl)

    # ------------------------------------------------------------------ 2 Store league table
    p, sl = page_frame(r, "league", "Store league table", "A balanced network: the top 10% of stores make 18% of sales", "Ranked by sales per open day  ·  Format B: 17 stores with 2.5x the transactions at half the ticket, open on Sundays", 2, [STATE, FORMAT, COMPBAND, LOYAL])
    kpi_band(p, "kpi", [(K("Trading Stores"), "Trading stores"), (K("Sales per Store Day"), "Sales per store-day"),
                        (K("Top Decile Sales Share"), "Sales from top 10% of stores"), (K("Avg Ticket"), "Average ticket"),
                        (K("Customers per Store Day"), "Transactions per store-day")])
    table(p, "league", x0, by, L, bh,
          [(C("Store", "StoreName"), "Store"), (C("Store", "StateName"), "State"), (C("Store", "Format"), "Format"),
           (C("Store", "CompetitionBand"), "Competitor"), (K("Store Rank"), "Rank"), (K("Sales per Store Day"), "Sales / day"),
           (K("Customers per Store Day"), "Transactions / day"), (K("Avg Ticket"), "Avg ticket"), (K("Share of Sales"), "Share of sales")],
          "Store league table, ranked by sales per open day", "Data bars: sales per day  ·  average ticket shaded light to dark orange",
          gradients=[(K("Avg Ticket"), S.panel, S.light)], databars=[K("Sales per Store Day")],
          sort=[(K("Sales per Store Day"), "Descending")], font=10)
    p.add("scatter", "scatterChart", x0 + L + GUT, by, R, bh,
          query={"Category": [P((C("Store", "StoreName"), "Store"))], "Series": [P((C("Store", "Format"), "Format"))],
                 "X": [P((K("Customers per Store Day"), "Transactions per day"))], "Y": [P((K("Avg Ticket"), "Average ticket"))]},
          objects={"legend": props(show=b(True), position=s("Top")), "bubbles": props(bubbleSize=n(-10))},
          vco=_title("Traffic vs basket, every store", "Two routes to a strong store: more transactions or a bigger ticket"),
          alt="Scatter of transactions per day vs average ticket by store")
    finish(r, p, sl)

    # ------------------------------------------------------------------ 3 Promotions
    p, sl = page_frame(r, "promo", "Promotions", "Promotions lift daily sales 39%: half from more shoppers, half from bigger baskets", "Weekday-matched, Monday to Friday  ·  stores are on promotion 45% of trading days  ·  the loyalty promotion shows no lift (-1.9%)", 3, [YEAR(), FORMAT, SCHOOL, STATE])
    kpi_band(p, "kpi", [(K("Promo Sales Uplift"), "Promo sales uplift"), (K("Promo Traffic Uplift"), "of which: traffic"),
                        (K("Promo Ticket Uplift"), "of which: ticket"), (K("Promo Day Share"), "Share of days on promo"),
                        (K("Loyalty Month Effect"), "Loyalty-month effect")])
    w3 = (cw - 2 * GUT) // 3
    th = 330
    bar(p, "wd", x0, by, w3, th, (C("Date", "Weekday"), "Weekday"), (K("Promo Sales Uplift"), "Uplift"), "Uplift by weekday",
        "Promo vs regular days, same weekday", horizontal=False, sort_desc=False, sort_cat=C("Date", "Weekday Number"),
        filters=[D.in_filter("Date", "Weekday Number", [1, 2, 3, 4, 5], "wd")])
    bar(p, "pf", x0 + w3 + GUT, by, w3, th, (C("Store", "Format"), "Format"), (K("Promo Sales Uplift"), "Uplift"), "Uplift by store format",
        "Weekday-matched sales uplift on promo days")
    p.add("decomp", "clusteredColumnChart", x0 + 2 * (w3 + GUT), by, cw - 2 * (w3 + GUT), th,
          query={"Category": [P((C("Store", "Format"), "Format"))],
                 "Y": [P((K("Promo Traffic Uplift"), "Traffic")), P((K("Promo Ticket Uplift"), "Ticket"))]},
          objects={"dataPoint": [fill_for(K("Promo Traffic Uplift"), S.primary), fill_for(K("Promo Ticket Uplift"), S.accent)],
                   "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(True), fontSize=n(10), color=color(S.ink))},
          vco=_title("Traffic or basket?", "Promo uplift split into transactions and average ticket"), alt="Promo uplift decomposition by format")
    line(p, "pvr", x0, by + th + GUT, L, bh - th - GUT, YM, [(K("Promo Sales per Store Day"), "Promo days"), (K("Regular Sales per Store Day"), "Regular weekdays")],
         "Promo days vs regular weekdays, every month", "Average daily sales of an open store, EUR", colors=(S.primary, S.grey), sort_cat=YMS)
    matrix(p, "pa", x0 + L + GUT, by + th + GUT, R, bh - th - GUT, [(C("Store", "Assortment"), "Assortment")],
           [(K("Promo Sales Uplift"), "Sales uplift"), (K("Promo Traffic Uplift"), "Traffic"), (K("Promo Ticket Uplift"), "Ticket")],
           "Uplift by assortment", "Weekday-matched", gradients=[(K("Promo Sales Uplift"), S.panel, S.light)], font=11)
    finish(r, p, sl)

    # ------------------------------------------------------------------ 4 Competition
    p, sl = page_frame(r, "competition", "Competition", "A new rival nearby costs a typical store 7% of sales; one in three loses 10%+", "131 competitor openings, Apr 2013 - Jan 2015, measured against the network trend  ·  yet stores close to rivals are not weaker overall", 4, [STATE, FORMAT, ASSORT])
    kpi_band(p, "kpi", [(K("Competition Events"), "Competitor openings studied"), (K("Median Competition Impact"), "Median sales impact"),
                        (K("Share of Events Losing 10%+"), "Stores losing 10%+"), (K("Near Competitor Share"), "Stores with a rival < 500 m"),
                        (K("Sales per Store Day"), "Sales per store-day")])
    bar(p, "band", x0, by, w3, th, (C("Store", "CompetitionBand"), "Nearest competitor"), (K("Sales per Store Day"), "Sales / store-day"),
        "Productivity by distance to the nearest rival", "Close rivals mark busy urban sites, not weak stores", horizontal=False, sort_desc=False,
        sort_cat=C("Store", "CompetitionBandOrder"))
    bar(p, "hist", x0 + w3 + GUT, by, w3, th, (C("Competition Event", "ImpactBand"), "Sales impact"), (K("Competition Events"), "Stores"),
        "What a new competitor did to sales", "Stores by difference-in-differences impact", horizontal=False, sort_desc=False,
        sort_cat=C("Competition Event", "ImpactBandOrder"), gradient=(S.soft_neg, S.neg))
    bar(p, "impband", x0 + 2 * (w3 + GUT), by, cw - 2 * (w3 + GUT), th, (C("Store", "CompetitionBand"), "Distance"), (K("Median Competition Impact"), "Median impact"),
        "Impact by distance of the new rival", "Median difference-in-differences impact", horizontal=True, sort_desc=False,
        sort_cat=C("Store", "CompetitionBandOrder"), gradient=(S.neg, S.soft_neg))
    table(p, "events", x0, by + th + GUT, L, bh - th - GUT,
          [(C("Store", "StoreName"), "Store"), (C("Store", "StateName"), "State"), (C("Store", "Format"), "Format"),
           (C("Store", "CompetitionBand"), "Rival distance"), (C("Competition Event", "CompetitorOpened"), "Rival opened"),
           (K("Avg Pre Sales"), "Sales / day before"), (K("Avg Post Sales"), "Sales / day after"), (K("Avg Impact"), "Impact vs network")],
          "Every competitor opening, with its impact", "Before: 90 days pre-opening  ·  after: days 30-150  ·  impact net of the network trend",
          gradients=[(K("Avg Impact"), S.soft_neg, S.soft_pos, S.panel, 0)], sort=[(K("Avg Impact"), "Ascending")], font=10)
    p.add("prepost", "clusteredColumnChart", x0 + L + GUT, by + th + GUT, R, bh - th - GUT,
          query={"Category": [P((C("Store", "Format"), "Format"))], "Y": [P((K("Avg Pre Sales"), "Before")), P((K("Avg Post Sales"), "After"))]},
          objects={"dataPoint": [fill_for(K("Avg Pre Sales"), S.grey), fill_for(K("Avg Post Sales"), S.primary)],
                   "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(False))},
          vco=_title("Before and after, by format", "Average daily sales of affected stores, EUR"), alt="Sales before and after competitor opening")
    finish(r, p, sl)

    # ------------------------------------------------------------------ 5 Calendar & staffing
    p, sl = page_frame(r, "calendar", "Calendar & staffing", "Footfall swings 30% across the week: staff to the peak, not the average", "Transactions per open store-day by weekday and month  ·  school holidays add 4% to daily sales", 5, [YEAR(), STATE, FORMAT])
    kpi_band(p, "kpi", [(K("Customers per Store Day"), "Transactions per store-day"), (K("Peak to Trough Weekday"), "Busiest vs quietest day"),
                        (K("School Holiday Uplift"), "School-holiday effect"), (K("Avg Ticket"), "Average ticket"), (K("Sunday Store Days"), "Sunday trading days")])
    matrix(p, "heat", x0, by, L, th, [(C("Date", "Weekday"), "Weekday")], [(K("Customers per Store Day"), "Transactions")],
           "Transactions per store-day: weekday by month", "Darker orange = busier: the staffing plan in one view  ·  Monday to Saturday",
           columns=[(C("Date", "Month"), "Month")], gradients=[(K("Customers per Store Day"), S.panel, S.primary)], font=11,
           filters=[D.in_filter("Date", "Weekday Number", [1, 2, 3, 4, 5, 6], "heat")])
    bar(p, "wdc", x0 + L + GUT, by, R, th, (C("Date", "Weekday"), "Weekday"), (K("Customers per Store Day"), "Transactions"), "Footfall by weekday",
        "Transactions per open store-day, Mon-Sat (only 33 stores trade on Sundays)", horizontal=False, sort_desc=False, sort_cat=C("Date", "Weekday Number"),
        filters=[D.in_filter("Date", "Weekday Number", [1, 2, 3, 4, 5, 6], "wdc")])
    p.add("season", "lineChart", x0, by + th + GUT, L, bh - th - GUT,
          query={"Category": [P((C("Date", "Month"), "Month"))], "Y": [P((K("Sales per Store Day"), "Sales / store-day"))], "Series": [P(C("Date", "Year"))]},
          objects={"lineStyles": props(showMarker=b(True), strokeWidth=n(2.5))},
          vco=_title("Seasonality, year by year", "Average daily sales of an open store, EUR  ·  December peak, January trough"),
          sort=[(C("Date", "Month Number"), "Ascending")], alt="Seasonality by year")
    p.visuals[-1]["visual"]["query"]["sortDefinition"] = {"sort": [{"field": C("Date", "Month"), "direction": "Ascending"}], "isDefaultSort": True}
    bar(p, "sch", x0 + L + GUT, by + th + GUT, R, bh - th - GUT, (C("Store", "Format"), "Format"), (K("School Holiday Uplift"), "Uplift"),
        "School holidays by format", "Sales per store-day in school holidays vs term time")
    finish(r, p, sl)

    # ------------------------------------------------------------------ 6 Sales forecast (ML)
    p, sl = page_frame(r, "forecast", "Sales forecast", "The forecast cuts error by 28% vs a naive baseline: 7.9% MAPE on six unseen weeks", "Gradient boosting on 27 features  ·  trained to 18 Jun 2015, tested 19 Jun - 31 Jul  ·  network bias -0.6%", 6, [STATE, FORMAT, ASSORT])
    kpi_band(p, "kpi", [(K("Forecast MAPE"), "Forecast error (MAPE)"), (K("Baseline MAPE"), "Naive baseline error"), (K("Error Reduction"), "Error reduction"),
                        (K("Forecast RMSPE"), "RMSPE (Kaggle metric)"), (K("Forecast Bias"), "Network bias"), (K("FC Actual"), "Actual sales, 6 weeks", 1000000)])
    fh = 400
    line(p, "fcline", x0, by, L, fh, (C("Date", "Date"), "Day"), [(K("FC Actual"), "Actual"), (K("FC Forecast"), "Forecast"), (K("FC Baseline"), "Naive baseline")],
         "Six weeks the model never saw: actual vs forecast", "Network sales per trading day (Mon-Sat), 19 Jun - 31 Jul 2015  ·  gradient boosting vs same-weekday median",
         colors=(S.accent, S.primary, S.grey), sort_cat=C("Date", "Date"))
    p.visuals[-1]["filterConfig"] = {"filters": [D.in_filter("Date", "Weekday Number", [1, 2, 3, 4, 5, 6], "fcline")]}
    p.add("mapefmt", "clusteredBarChart", x0 + L + GUT, by, R, fh,
          query={"Category": [P((C("Store", "Format"), "Format"))], "Y": [P((K("Forecast MAPE"), "Model")), P((K("Baseline MAPE"), "Baseline"))]},
          objects={"dataPoint": [fill_for(K("Forecast MAPE"), S.primary), fill_for(K("Baseline MAPE"), S.grey)],
                   "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(True), fontSize=n(10), color=color(S.ink))},
          vco=_title("Error by store format", "Mean absolute % error per store-day"), alt="Forecast error by format")
    table(p, "worst", x0, by + fh + GUT, L, bh - fh - GUT,
          [(C("Store", "StoreName"), "Store"), (C("Store", "StateName"), "State"), (C("Store", "Format"), "Format"), (K("FC Actual"), "Actual"),
           (K("FC Forecast"), "Forecast"), (K("Forecast Bias"), "Bias"), (K("Forecast MAPE"), "MAPE"), (K("Baseline MAPE"), "Baseline MAPE")],
          "Stores to review: highest forecast error", "Shaded by error  ·  large bias usually means a local event the model cannot see",
          gradients=[(K("Forecast MAPE"), S.panel, S.soft_neg)], sort=[(K("Forecast MAPE"), "Descending")], font=10)
    bar(p, "mapest", x0 + L + GUT, by + fh + GUT, R, bh - fh - GUT, (C("Store", "StateName"), "State"), (K("Forecast MAPE"), "MAPE"),
        "Error by state", "Mean absolute % error per store-day", gradient=(S.light, S.primary))
    finish(r, p, sl)

    # ------------------------------------------------------------------ 7 About
    p = r.page("about", "About & method", bg=S.page)
    chrome(p, TITLE, "How this report is built", "Questions, data, model and definitions behind every number", f"7 / {NP}", SRC, title_w=TW)
    about_page(p, [
        ("Business questions", ["Is the network growing like-for-like, and where?", "Which stores lead and lag, and is it traffic or basket?",
                                "Do promotions pay, and does a new competitor hurt?", "When are stores busiest, and can we forecast the next six weeks?"]),
        ("Data & ML", ["Rossmann Store Sales: 1,017,209 store-days, 844,338 open with sales; store attributes and states joined.",
                       "180 stores have no data Jul-Dec 2014 (source gap); like-for-like compares only store-months present in both years.",
                       "Forecast: gradient boosting on log sales, 27 features, trained to 18 Jun 2015 and tested on the next 6 weeks."]),
        ("Model & UX", ["Star schema: Store Day fact with Store and Date; Competition Event and Forecast facts share the Store dimension.",
                        "Every page has a slide-in filter panel (button + bookmarks) with page-specific slicers, and a filter breadcrumb.",
                        "Measures documented in display folders; LFL and weekday-matched promo uplift written in DAX."]),
    ], [("Sales per store-day", "sales / open store-days: the productivity measure behind every comparison."),
        ("Like-for-like (LFL)", "same stores, same months, this year vs last year."),
        ("Promo uplift", "sales per store-day on promo days vs regular days, matched by weekday (Mon-Fri)."),
        ("Competition impact", "store's sales change minus the network's change over the same window (difference in differences)."),
        ("MAPE / RMSPE", "mean absolute / root mean square percentage error per store-day.")],
        "Built by Mohamed Ayman, Senior Data Scientist  ·  github.com/ayman-stat  ·  mohamedaymanstat.vercel.app")

    r.write()
    write_pbip(os.path.join(ROOT, f"{NAME}.pbip"), f"{NAME}.Report")


if __name__ == "__main__":
    build()
    print("report written")
