# App.py
import io, csv, math, time, traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

st.set_page_config(page_title="Aggressive but Balanced – 1M Tracker", layout="wide")
st.write("✅ App booted…")

# -------------------- CONFIG --------------------
TICKERS = ["OKLO", "SHLD", "ARKG", "CIBR", "NVDA", "TSLA", "URA"]
DEFAULT_WEIGHTS = {"OKLO":0.16, "SHLD":0.16, "ARKG":0.16, "CIBR":0.16, "NVDA":0.16, "TSLA":0.08, "URA":0.12}
DEFAULT_NIS = 10000.0
DEFAULT_FX  = 0.27  # NIS→USD

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://finance.yahoo.com",
    "Referer": "https://finance.yahoo.com/",
    "Connection": "keep-alive",
}

# -------------------- SIDEBAR --------------------
st.sidebar.title("Settings")
nis = st.sidebar.number_input("Portfolio size (₪ NIS)", value=DEFAULT_NIS, step=100.0, min_value=0.0)
fx  = st.sidebar.number_input("NIS → USD", value=DEFAULT_FX, step=0.01, min_value=0.0)
total_usd = nis * fx
st.sidebar.write(f"≈ **${total_usd:,.2f} USD**")

st.sidebar.markdown("---")
st.sidebar.write("**Weights** (sum to 1.00)")
weights, ssum = {}, 0.0
for t in TICKERS:
    w = st.sidebar.number_input(f"{t} weight", value=float(DEFAULT_WEIGHTS[t]), step=0.01,
                                min_value=0.0, max_value=1.0, key=f"w_{t}")
    weights[t] = float(w)
    ssum += float(w)
if abs(ssum - 1.0) > 1e-6:
    st.sidebar.error(f"Weights sum to {ssum:.2f}. They must sum to 1.00.")
    st.stop()

# -------------------- TIMEFRAME (MAIN) --------------------
st.title("Aggressive but Balanced – 1‑Month Portfolio Tracker")
st.caption("Time‑windowed holdings by default. Switch to live snapshot if needed. Not investment advice.")

st.markdown("### Timeframe")
c1, c2, c3 = st.columns([1, 1, 2])
today_local = datetime.now(ZoneInfo("Asia/Jerusalem")).date()
preset = c1.selectbox("Preset", ["1W", "1M", "3M", "6M", "1Y", "Custom"], index=2)

if preset == "1W":
    start_date = today_local - timedelta(days=7)
    end_date   = today_local
elif preset == "1M":
    start_date = today_local - timedelta(days=30)
    end_date   = today_local
elif preset == "3M":
    start_date = today_local - timedelta(days=90)
    end_date   = today_local
elif preset == "6M":
    start_date = today_local - timedelta(days=182)
    end_date   = today_local
elif preset == "1Y":
    start_date = today_local - timedelta(days=365)
    end_date   = today_local
else:
    with c2:
        start_date = st.date_input("Start date", today_local - timedelta(days=90), key="bt_start")
    with c3:
        end_date   = st.date_input("End date", today_local, key="bt_end")

if start_date > end_date:
    st.error("Start date must be on/before End date.")
    st.stop()

# -------------------- FETCHERS --------------------
@st.cache_data(ttl=60)
def yahoo_quotes(symbols: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    r = requests.get(url, params={"symbols": ",".join(symbols)}, headers=UA, timeout=10)
    r.raise_for_status()
    out: dict[str, tuple[float, float]] = {}
    for d in r.json().get("quoteResponse", {}).get("result", []):
        sym = d.get("symbol")
        price = d.get("regularMarketPrice")
        prev  = d.get("regularMarketPreviousClose", price)
        if sym is not None:
            out[sym] = (
                float(price) if price is not None else float("nan"),
                float(prev)  if prev  is not None else float("nan"),
            )
    return out

def stooq_symbols(symbols: list[str]) -> dict[str, str]:
    m = {}
    for s in symbols:
        if "." in s:
            m[s] = s.replace(".", "-").upper()
        else:
            m[s] = f"{s}.US"
    return m

@st.cache_data(ttl=300)
def stooq_quotes(symbols: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    mapped = stooq_symbols(list(symbols))
    url = "https://stooq.com/q/l/"
    params = {"s": ",".join(mapped.values()), "f": "sd2t2ohlcv", "h": "", "e": "csv"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    rev = {v: k for k, v in mapped.items()}
    out: dict[str, tuple[float, float]] = {}
    for row in reader:
        sym = row.get("Symbol")
        orig = rev.get(sym, sym)
        try:
            close = float(row.get("Close") or "nan")
            prev  = float(row.get("Open") or "nan")  # proxy for prev close
        except ValueError:
            close, prev = float("nan"), float("nan")
        out[orig.replace("-", ".")] = (close, prev)
    return out

def fetch_quotes(symbols: list[str]) -> dict[str, tuple[float, float]]:
    syms = tuple(symbols)
    yahoo_out: dict[str, tuple[float, float]] = {}
    try:
        for i in range(2):
            try:
                yahoo_out = yahoo_quotes(syms)
                break
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code in (401, 403):
                    time.sleep(1.0 + i)
                    continue
                raise
    except Exception:
        yahoo_out = {}

    missing = [s for s in symbols if s not in yahoo_out or math.isnan(yahoo_out[s][0])]
    stq_out: dict[str, tuple[float, float]] = {}
    if missing:
        try:
            stq_out = stooq_quotes(tuple(missing))
        except Exception:
            pass

    out = {}
    for s in symbols:
        if s in yahoo_out and not math.isnan(yahoo_out[s][0]):
            out[s] = yahoo_out[s]
        else:
            out[s] = stq_out.get(s, (float("nan"), float("nan")))
    return out

@st.cache_data(ttl=600)
def yahoo_history(symbol: str, start_dt: datetime, end_dt: datetime, interval: str = "1d") -> pd.Series:
    """Daily close series for [start_dt, end_dt] inclusive (UTC)."""
    p1 = int(start_dt.replace(tzinfo=ZoneInfo("UTC")).timestamp())
    p2 = int((end_dt + timedelta(days=1)).replace(tzinfo=ZoneInfo("UTC")).timestamp())  # inclusive end
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": p1, "period2": p2, "interval": interval, "events": "div,splits"}
    r = requests.get(url, params=params, headers=UA, timeout=10)
    r.raise_for_status()
    j = r.json()
    result = j.get("chart", {}).get("result", [])
    if not result:
        return pd.Series(dtype="float64")
    res = result[0]
    closes = res.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    ts = res.get("timestamp", [])
    if not closes or not ts:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(ts, unit="s", utc=True).date
    s = pd.Series(closes, index=pd.Index(idx, name="date"), dtype="float64")
    return s.dropna()

# -------------------- BUILD HISTORY (first) --------------------
hist_map: dict[str, pd.Series] = {}
for t in TICKERS:
    try:
        s = yahoo_history(
            t,
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date,   datetime.min.time()),
        )
        if not s.empty:
            hist_map[t] = s
    except Exception:
        # continue silently; we will handle missing later
        pass

prices = pd.DataFrame(hist_map).dropna(how="any")  # aligned overlapping dates only

# -------------------- LIVE QUOTES (optional live table) --------------------
try:
    live_quotes = fetch_quotes(TICKERS)
except Exception as e:
    live_quotes = {}
    st.warning(f"Live quotes unavailable right now: {e}")

# -------------------- HOLDINGS TABLE (timeframe by default) --------------------
table_mode = st.radio("Holdings table mode", ["Timeframe (start→end)", "Live snapshot (now)"],
                      horizontal=True, index=0)

def render_table_timeframe() -> pd.DataFrame | None:
    if prices.empty:
        st.warning("No overlapping dates for selected window.")
        return None
    p0 = prices.iloc[0]  # start prices
    p1 = prices.iloc[-1] # end prices
    rows = []
    for t in TICKERS:
        if t in prices.columns:
            sp0 = float(p0[t]) if pd.notna(p0[t]) else float("nan")
            sp1 = float(p1[t]) if pd.notna(p1[t]) else float("nan")
            if not (math.isnan(sp0) or sp0 <= 0 or math.isnan(sp1)):
                alloc = total_usd * weights[t]
                units = alloc / sp0
                start_v = alloc
                end_v   = units * sp1
                rows.append({
                    "Ticker": t,
                    "Start Price ($)": round(sp0, 2),
                    "End Price ($)":   round(sp1, 2),
                    "Units":           round(units, 4),
                    "Start Value $":   round(start_v, 2),
                    "End Value $":     round(end_v, 2),
                    "Δ $":             round(end_v - start_v, 2),
                    "Δ %":             round((end_v / start_v - 1.0) * 100.0, 2),
                })
    df = pd.DataFrame(rows)
    st.subheader("Holdings (timeframe‑based)")
    st.dataframe(df, use_container_width=True)

    tot_start = float(df["Start Value $"].sum()) if not df.empty else 0.0
    tot_end   = float(df["End Value $"].sum()) if not df.empty else 0.0
    tot_pl    = tot_end - tot_start
    tot_pct   = (tot_end / tot_start - 1.0) * 100.0 if tot_start > 0 else 0.0
    a, b, c = st.columns(3)
    a.metric("Start (window)", f"${tot_start:,.2f}")
    b.metric("End (window)",   f"${tot_end:,.2f}")
    c.metric("P/L (window)",   f"${tot_pl:,.2f}", f"{tot_pct:.2f}%")
    return df

def render_table_live() -> pd.DataFrame:
    rows = []
    for t in TICKERS:
        price, prev = live_quotes.get(t, (float("nan"), float("nan")))
        if math.isnan(prev) or prev == 0:
            prev = price
        day_pct = (price / prev - 1.0) * 100.0 if (price and prev and not math.isnan(price) and not math.isnan(prev)) else 0.0
        alloc = total_usd * weights[t]
        units = alloc / price if (price and not math.isnan(price) and price > 0) else 0.0
        value = units * price if not math.isnan(price) else 0.0
        rows.append({
            "Ticker": t,
            "Price ($)": round(0.0 if math.isnan(price) else price, 2),
            "Day %": round(day_pct, 2),
            "Alloc $": round(alloc, 2),
            "Units": round(units, 4),
            "Position Value $": round(value, 2),
        })
    df = pd.DataFrame(rows)
    df["Δ Value $"] = (df["Position Value $"] - df["Alloc $"]).round(2)
    df["Δ %"] = ((df["Position Value $"] / df["Alloc $"] - 1.0) * 100.0).replace([np.inf, -np.inf], 0).fillna(0).round(2)
    st.subheader("Holdings (live snapshot)")
    st.dataframe(df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    total_val = float(df["Position Value $"].sum())
    pl_val    = float(df["Δ Value $"].sum())
    pl_pct    = (pl_val / total_usd * 100.0) if total_usd > 0 else 0.0
    c1.metric("Total Value ($)", f"{total_val:,.2f}")
    c2.metric("Daily P/L ($)",   f"{pl_val:,.2f}")
    c3.metric("Daily P/L (%)",   f"{pl_pct:.2f}%")
    return df

try:
    shown_tbl = render_table_timeframe() if table_mode.startswith("Timeframe") else render_table_live()
except Exception as e:
    st.error("Error building holdings table.")
    with st.expander("Show error details"):
        st.code("".join(traceback.format_exc()))
    shown_tbl = None

# -------------------- PORTFOLIO VALUE OVER TIME --------------------
st.subheader("Portfolio value over time")
if prices.empty:
    st.warning("No historical data available for the selected window.")
else:
    # Units fixed at window start
    first_row = prices.iloc[0]
    units = {}
    for t in TICKERS:
        if t in prices.columns:
            p0 = float(first_row[t]) if pd.notna(first_row[t]) else float("nan")
            units[t] = (total_usd * weights[t]) / p0 if (not math.isnan(p0) and p0 > 0) else 0.0
        else:
            units[t] = 0.0

    port_val = (prices * pd.Series(units)).sum(axis=1)
    port_val.name = "Portfolio Value ($)"

    start_val = float(port_val.iloc[0])
    if start_val > 0:
        port_pct = (port_val / start_val - 1.0) * 100.0
    else:
        port_pct = port_val * 0.0
    port_pct.name = "Return (%)"

    # Max Drawdown ($/%)
    cum_max = port_val.cummax()
    drawdown_pct = (port_val / cum_max - 1.0) * 100.0
    max_dd_pct = float(drawdown_pct.min())
    trough_idx = drawdown_pct.idxmin()
    peak_up_to_trough = cum_max.loc[:trough_idx].idxmax()
    mdd_usd = float(port_val.loc[trough_idx] - port_val.loc[peak_up_to_trough])

    end_val = float(port_val.iloc[-1])
    ret_usd = end_val - start_val
    ret_pct = (end_val / start_val - 1.0) * 100.0 if start_val > 0 else 0.0

    # Charts (Altair with explicit axes)
    val_df = port_val.reset_index()
    val_df.columns = ["date", "value_usd"]
    chart_val = (
        alt.Chart(val_df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value_usd:Q", title="Portfolio Value ($)", axis=alt.Axis(format="$,.0f")),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("value_usd:Q", format="$,.2f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart_val, use_container_width=True)

    pct_df = port_pct.reset_index()
    pct_df.columns = ["date", "ret_pct"]
    chart_pct = (
        alt.Chart(pct_df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("ret_pct:Q", title="Return (%)", axis=alt.Axis(format=".1f")),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("ret_pct:Q", format=".2f")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart_pct, use_container_width=True)

    # Metrics
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Start value", f"${start_val:,.2f}")
    d2.metric("End value", f"${end_val:,.2f}", f"{ret_pct:.2f}%")
    d3.metric("Period P&L ($)", f"{ret_usd:,.2f}")
    d4.metric("Max drawdown (%)", f"{max_dd_pct:.2f}%")
    d5.metric("Max drawdown ($)", f"{mdd_usd:,.2f}")

    # CSV download
    csv_bytes = pd.concat([port_val, port_pct], axis=1).to_csv(index=True).encode()
    st.download_button("Download portfolio history (CSV)", csv_bytes, "portfolio_history.csv", "text/csv")

    # -------------------- FORECAST (Return → Probability) --------------------
    st.markdown("### Forecast (probability view)")

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    horizon_days = fc1.number_input("Horizon (trading days)", min_value=5, max_value=60, value=21, step=1)
    n_sims = fc2.number_input("Simulations", min_value=500, max_value=10000, value=4000, step=500)
    use_window = fc3.selectbox("Return source", ["This window", "Last 3M"], index=0)

    # Build daily returns for bootstrap
    if use_window == "This window":
        ret_series = port_val.pct_change().dropna()
    else:
        hist3m = {}
        s3_start = today_local - timedelta(days=90)
        s3_end = today_local
        for t in TICKERS:
            try:
                s3 = yahoo_history(
                    t,
                    datetime.combine(s3_start, datetime.min.time()),
                    datetime.combine(s3_end, datetime.min.time()),
                )
                if not s3.empty:
                    hist3m[t] = s3
            except Exception:
                pass
        if hist3m:
            p3 = pd.DataFrame(hist3m).dropna(how="any")
            if not p3.empty:
                f3 = p3.iloc[0]
                u3 = {}
                for tt in TICKERS:
                    if tt in p3.columns and pd.notna(f3[tt]) and f3[tt] > 0:
                        u3[tt] = (total_usd * weights[tt]) / float(f3[tt])
                    else:
                        u3[tt] = 0.0
                pv3 = (p3 * pd.Series(u3)).sum(axis=1)
                ret_series = pv3.pct_change().dropna()
            else:
                ret_series = port_val.pct_change().dropna()
        else:
            ret_series = port_val.pct_change().dropna()

    if ret_series.empty:
        st.info("Not enough data to simulate.")
    else:
        try:
            last_val = float(port_val.iloc[-1])
            rets = ret_series.values

            sims_end = []
            sims_ret = []
            for _ in range(int(n_sims)):
                samp = np.random.choice(rets, size=int(horizon_days), replace=True)
                growth = np.prod(1.0 + samp)
                sims_end.append(last_val * growth)
                sims_ret.append((growth - 1.0) * 100.0)

            sims_end = np.array(sims_end)
            sims_ret = np.array(sims_ret)

            # Probability curve: P(final return ≥ r)
            r_grid = np.linspace(np.percentile(sims_ret, 1), np.percentile(sims_ret, 99), 200)
            prob_ge = [(sims_ret >= r).mean() * 100.0 for r in r_grid]
            curve_df = pd.DataFrame({"Return %": r_grid, "Probability ≥ r": prob_ge})

            # Stats & tables
            p5, p50, p95 = np.percentile(sims_end, [5, 50, 95])
            ret5, ret50, ret95 = np.percentile(sims_ret, [5, 50, 95])
            mean_end = sims_end.mean()
            mean_ret = sims_ret.mean()
            prob_profit = (sims_end > last_val).mean() * 100.0
            var5_ret = np.percentile(sims_ret, 5)
            cvar5_ret = sims_ret[sims_ret <= var5_ret].mean() if np.any(sims_ret <= var5_ret) else var5_ret

            levels = np.array([-5, -3, 0, 3, 5], dtype=float)
            probs_levels = [(sims_ret >= L).mean() * 100.0 for L in levels]

            chart_curve = (
                alt.Chart(curve_df)
                .mark_line()
                .encode(
                    x=alt.X("Return %:Q", title="Return over horizon (%)"),
                    y=alt.Y("Probability ≥ r:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
                    tooltip=[
                        alt.Tooltip("Return %:Q", format=".1f"),
                        alt.Tooltip("Probability ≥ r:Q", format=".1f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart_curve, use_container_width=True)

            summary = pd.DataFrame(
                {
                    "Metric": [
                        "Horizon (trading days)",
                        "Start (current $)",
                        "Mean end ($)",
                        "Median end ($)",
                        "5% end ($)",
                        "95% end ($)",
                        "Mean return (%)",
                        "Median return (%)",
                        "5% return (%)",
                        "95% return (%)",
                        "VaR 5% (return %)",
                        "CVaR 5% (return %)",
                        "Prob(final > current) %",
                    ],
                    "Value": [
                        int(horizon_days),
                        f"${last_val:,.2f}",
                        f"${mean_end:,.2f}",
                        f"${p50:,.2f}",
                        f"${p5:,.2f}",
                        f"${p95:,.2f}",
                        f"{mean_ret:.2f}%",
                        f"{ret50:.2f}%",
                        f"{ret5:.2f}%",
                        f"{ret95:.2f}%",
                        f"{var5_ret:.2f}%",
                        f"{cvar5_ret:.2f}%",
                        f"{prob_profit:.1f}%",
                    ],
                }
            )
            thresh = pd.DataFrame(
                {"Return threshold (%)": levels, "Prob( ≥ threshold ) %": np.round(probs_levels, 1)}
            )

            st.markdown("**Risk / Return summary**")
            st.dataframe(summary, use_container_width=True)
            st.markdown("**Threshold probabilities**")
            st.dataframe(thresh, use_container_width=True)

            # Downloads
            dl1 = pd.DataFrame({"Sim end ($)": sims_end, "Sim return (%)": sims_ret}).to_csv(index=False).encode()
            dl2 = curve_df.to_csv(index=False).encode()
            cdl1, cdl2 = st.columns(2)
            cdl1.download_button("Download simulations (CSV)", dl1, "simulations.csv", "text/csv")
            cdl2.download_button("Download probability curve (CSV)", dl2, "prob_curve.csv", "text/csv")

        except Exception:
            st.error("Forecast failed.")
            with st.expander("Show error details"):
                st.code("".join(traceback.format_exc()))
