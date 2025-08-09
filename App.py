# App.py
import math, time, io, csv, requests
import numpy as np, pandas as pd, streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import altair as alt

st.set_page_config(page_title="Aggressive but Balanced – 1M Tracker", layout="wide")
st.write("✅ App booted… loading quotes")

# ---- CONFIG ----
TICKERS = ["OKLO","SHLD","ARKG","CIBR","NVDA","TSLA","URA"]
DEFAULT_WEIGHTS = {"OKLO":0.16,"SHLD":0.16,"ARKG":0.16,"CIBR":0.16,"NVDA":0.16,"TSLA":0.08,"URA":0.12}
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

# ---------- Sidebar ----------
st.sidebar.title("Settings")
nis = st.sidebar.number_input("Portfolio size (₪ NIS)", value=DEFAULT_NIS, step=100.0, min_value=0.0)
fx  = st.sidebar.number_input("NIS → USD", value=DEFAULT_FX, step=0.01, min_value=0.0)
total_usd = nis * fx
st.sidebar.write(f"≈ **${total_usd:,.2f} USD**")
st.sidebar.markdown("---")
st.sidebar.write("**Weights** (sum to 1.00)")
weights, s = {}, 0.0
for t in TICKERS:
    w = st.sidebar.number_input(f"{t} weight", value=float(DEFAULT_WEIGHTS[t]), step=0.01,
                                min_value=0.0, max_value=1.0, key=f"w_{t}")
    weights[t] = float(w); s += float(w)
if abs(s - 1.0) > 1e-6:
    st.sidebar.error(f"Weights sum to {s:.2f}. They must sum to 1.00.")
    st.stop()

st.title("Aggressive but Balanced – 1‑Month Portfolio Tracker")
st.caption("Live prices (Yahoo first, Stooq fallback), history, drawdown, and a simple forecast. Info only — not investment advice.")

# ---------- Live quote fetchers ----------
@st.cache_data(ttl=60)
def yahoo_quotes(symbols: tuple[str, ...]) -> dict[str, tuple[float,float]]:
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    r = requests.get(url, params={"symbols": ",".join(symbols)}, headers=UA, timeout=10)
    r.raise_for_status()
    out = {}
    for d in r.json().get("quoteResponse", {}).get("result", []):
        sym = d.get("symbol")
        price = d.get("regularMarketPrice")
        prev  = d.get("regularMarketPreviousClose", price)
        if sym is not None:
            out[sym] = (
                float(price) if price is not None else float("nan"),
                float(prev)  if prev  is not None else float("nan")
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
def stooq_quotes(symbols: tuple[str, ...]) -> dict[str, tuple[float,float]]:
    mapped = stooq_symbols(list(symbols))
    url = "https://stooq.com/q/l/"
    params = {"s": ",".join(mapped.values()), "f": "sd2t2ohlcv", "h": "", "e": "csv"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    reader = csv.DictReader(io.StringIO(r.text))
    rev = {v: k for k, v in mapped.items()}
    out = {}
    for row in reader:
        sym = row.get("Symbol")
        orig = rev.get(sym, sym)
        try:
            close = float(row.get("Close") or "nan")
            prev  = float(row.get("Open") or "nan")
        except ValueError:
            close, prev = float("nan"), float("nan")
        out[orig.replace("-", ".")] = (close, prev)
    return out

def fetch_quotes(symbols: list[str]) -> dict[str, tuple[float,float]]:
    syms_tuple = tuple(symbols)
    yahoo_out = {}
    try:
        for i in range(2):
            try:
                yahoo_out = yahoo_quotes(syms_tuple)
                break
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code in (401,403):
                    time.sleep(1.0 + i)
                    continue
                raise
    except Exception:
        yahoo_out = {}
    missing = [s for s in symbols if s not in yahoo_out or math.isnan(yahoo_out[s][0])]
    stq_out = {}
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

# ---- Yahoo history (for backtest) ----
@st.cache_data(ttl=600)
def yahoo_history(symbol: str, start_dt: datetime, end_dt: datetime, interval: str = "1d") -> pd.Series:
    p1 = int(start_dt.replace(tzinfo=ZoneInfo("UTC")).timestamp())
    p2 = int((end_dt + timedelta(days=1)).replace(tzinfo=ZoneInfo("UTC")).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"period1": p1, "period2": p2, "interval": interval, "events": "div,splits"},
                     headers=UA, timeout=10)
    r.raise_for_status()
    j = r.json()
    res = j.get("chart", {}).get("result", [])
    if not res:
        return pd.Series(dtype="float64")
    res = res[0]
    closes = res.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    ts = res.get("timestamp", [])
    if not closes or not ts:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(ts, unit="s", utc=True).date
    return pd.Series(closes, index=pd.Index(idx, name="date"), dtype="float64").dropna()

# ---------- Live snapshot ----------
try:
    quotes = fetch_quotes(TICKERS)
except Exception as e:
    st.error(f"Price download failed. Please refresh in a minute. Details: {e}")
    st.stop()

rows = []
for t in TICKERS:
    price, prev = quotes.get(t, (float("nan"), float("nan")))
    if math.isnan(prev) or prev == 0:
        prev = price
    day_pct = (price/prev - 1.0) * 100.0 if (price and prev and not math.isnan(price) and not math.isnan(prev)) else 0.0
    alloc = total_usd * weights[t]
    units = alloc / price if (price and not math.isnan(price) and price > 0) else 0.0
    value = units * price if not math.isnan(price) else 0.0
    rows.append({
        "Ticker": t,
        "Price ($)": round(0.0 if math.isnan(price) else price, 2),
        "Day %": round(day_pct, 2),
        "Alloc $": round(alloc, 2),
        "Units": round(units, 4),
        "Position Value $": round(value, 2)
    })

tbl = pd.DataFrame(rows)
tbl["Δ Value $"] = (tbl["Position Value $"] - tbl["Alloc $"]).round(2)
tbl["Δ %"] = ((tbl["Position Value $"] / tbl["Alloc $"] - 1.0) * 100.0).replace([np.inf, -np.inf], 0).fillna(0).round(2)

st.subheader("Holdings (snapshot)")
st.dataframe(tbl, use_container_width=True)

c1, c2, c3 = st.columns(3)
total_val = float(tbl["Position Value $"].sum())
pl_val    = float(tbl["Δ Value $"].sum())
pl_pct    = (pl_val / total_usd * 100.0) if total_usd > 0 else 0.0
c1.metric("Total Value ($)", f"{total_val:,.2f}")
c2.metric("Daily P/L ($)",   f"{pl_val:,.2f}")
c3.metric("Daily P/L (%)",   f"{pl_pct:.2f}%")

# ---------- Timeframe controls (MAIN AREA) ----------
st.markdown("### Timeframe")
c1, c2, c3 = st.columns([1,1,2])
today_local = datetime.now(ZoneInfo("Asia/Jerusalem")).date()
preset = c1.selectbox("Preset", ["1W","1M","3M","6M","1Y","Custom"], index=2)
if preset == "1W":
    start_date = today_local - timedelta(days=7);  end_date = today_local
elif preset == "1M":
    start_date = today_local - timedelta(days=30); end_date = today_local
elif preset == "3M":
    start_date = today_local - timedelta(days=90); end_date = today_local
elif preset == "6M":
    start_date = today_local - timedelta(days=182); end_date = today_local
elif preset == "1Y":
    start_date = today_local - timedelta(days=365); end_date = today_local
else:
    with c2:
        start_date = st.date_input("Start date", today_local - timedelta(days=90), key="bt_start")
    with c3:
        end_date   = st.date_input("End date", today_local, key="bt_end")

if start_date > end_date:
    st.error("Start date must be on/before End date.")
    st.stop()

# ---------- Portfolio value over time (no rebalancing) ----------
st.subheader("Portfolio value over time")

hist_map = {}
for t in TICKERS:
    try:
        s = yahoo_history(
            t,
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date,   datetime.min.time())
        )
        if not s.empty:
            hist_map[t] = s
    except Exception:
        pass

if not hist_map:
    st.warning("No historical data available for the selected window.")
else:
    prices = pd.DataFrame(hist_map).dropna(how="any")
    if prices.empty:
        st.warning("No overlapping dates across tickers in the selected range.")
    else:
        first_row = prices.iloc[0]
        units = {}
        for t in TICKERS:
            if t in prices.columns and first_row[t] and not math.isnan(first_row[t]) and first_row[t] > 0:
                units[t] = (total_usd * weights[t]) / first_row[t]
            else:
                units[t] = 0.0

        port_val = (prices * pd.Series(units)).sum(axis=1)
        port_val.name = "Portfolio Value ($)"

        start_val = float(port_val.iloc[0])
        port_pct = (port_val / start_val - 1.0) * 100.0 if start_val > 0 else port_val*0
        port_pct.name = "Return (%)"

        # --- Max Drawdown ($ and %) ---
        cum_max = port_val.cummax()
        drawdown_pct = (port_val / cum_max - 1.0) * 100.0
        max_dd_pct = float(drawdown_pct.min())

        trough_idx = drawdown_pct.idxmin()
        peak_up_to_trough = cum_max.loc[:trough_idx].idxmax()
        mdd_usd = float(port_val.loc[trough_idx] - port_val.loc[peak_up_to_trough])

        # Period P&L ($/%)
        end_val   = float(port_val.iloc[-1])
        ret_usd   = end_val - start_val
        ret_pct   = (end_val/start_val - 1.0) * 100.0 if start_val > 0 else 0.0

        # --- Charts with explicit axes (Altair) ---
        val_df = port_val.reset_index()
        val_df.columns = ["date","value_usd"]
        chart_val = (
            alt.Chart(val_df)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value_usd:Q", title="Portfolio Value ($)", axis=alt.Axis(format="$,.0f")),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("value_usd:Q", format="$,.2f")]
            )
            .properties(height=260)
        )
        st.altair_chart(chart_val, use_container_width=True)

        pct_df = port_pct.reset_index()
        pct_df.columns = ["date","ret_pct"]
        chart_pct = (
            alt.Chart(pct_df)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("ret_pct:Q", title="Return (%)", axis=alt.Axis(format=".1f")),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("ret_pct:Q", format=".2f")]
            )
            .properties(height=220)
        )
        st.altair_chart(chart_pct, use_container_width=True)

        # --- Metrics ---
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Start value", f"${start_val:,.2f}")
        d2.metric("End value",   f"${end_val:,.2f}", f"{ret_pct:.2f}%")
        d3.metric("Period P&L ($)", f"{ret_usd:,.2f}")
        d4.metric("Max drawdown (%)", f"{max_dd_pct:.2f}%")
        d5.metric("Max drawdown ($)", f"{mdd_usd:,.2f}")

        # CSV download
        csv_bytes = pd.concat([port_val, port_pct], axis=1).to_csv(index=True).encode()
        st.download_button("Download portfolio history (CSV)", csv_bytes,
                           file_name="portfolio_history.csv", mime="text/csv")

        # ---------- Forecast (bootstrap; USD) ----------
        st.markdown("### Forecast (experimental)")
        fc_col1, fc_col2, fc_col3 = st.columns([1,1,2])
        horizon_days = fc_col1.number_input("Horizon (trading days)", min_value=5, max_value=60, value=21, step=1)
        n_sims       = fc_col2.number_input("Simulations", min_value=200, max_value=5000, value=2000, step=100)
        use_window   = fc_col3.selectbox("Return source", ["This window", "Last 3M"], index=0)

        if use_window == "This window":
            ret_series = port_val.pct_change().dropna()
        else:
            hist3m = {}
            s3_start = today_local - timedelta(days=90)
            s3_end   = today_local
            for t in TICKERS:
                try:
                    s3 = yahoo_history(
                        t,
                        datetime.combine(s3_start, datetime.min.time()),
                        datetime.combine(s3_end,   datetime.min.time())
                    )
                    if not s3.empty:
                        hist3m[t] = s3
                except Exception:
                    pass
            if hist3m:
                p3 = pd.DataFrame(hist3m).dropna(how="any")
                if not p3.empty:
                    f3 = p3.iloc[0]
                    u3 = {tt: (total_usd * weights[tt]) / f3[tt]
                          if tt in p3.columns and f3[tt] and not math.isnan(f3[tt]) and f3[tt] > 0 else 0.0
                          for tt in TICKERS}
                    pv3 = (p3 * pd.Series(u3)).sum(axis=1)
                    ret_series = pv3.pct_change().dropna()
                else:
                    ret_series = port_val.pct_change().dropna()
            else:
                ret_series = port_val.pct_change().dropna()

        if ret_series.empty or start_val <= 0:
            st.info("Not enough data to simulate.")
        else:
            rets = ret_series.values
            last_val = float(port_val.iloc[-1])

            sims_end = []
            for _ in range(int(n_sims)):
                samp = np.random.choice(rets, size=int(horizon_days), replace=True)
                growth = np.prod(1.0 + samp)
                sims_end.append(last_val * growth)
            sims_end = np.array(sims_end)

            p5, p50, p95 = np.percentile(sims_end, [5, 50, 95])
            st.write(f"**Projected end value (median)**: ${p50:,.2f}  |  **5%**: ${p5:,.2f}  |  **95%**: ${p95:,.2f}")
            prob_prof = float((sims_end > last_val).mean() * 100.0)
            st.write(f"**Probability of profit over {int(horizon_days)} days:** {prob_prof:.1f}%")

            sim_df = pd.DataFrame({"end_value": sims_end})
            hist = (
                alt.Chart(sim_df)
                .mark_bar()
                .encode(
                    x=alt.X("end_value:Q", bin=alt.Bin(maxbins=30), title="Simulated end value ($)", axis=alt.Axis(format="$,.0f")),
                    y=alt.Y("count():Q", title="Frequency"),
                    tooltip=[alt.Tooltip("count():Q"), alt.Tooltip("end_value:Q", bin=True, format="$,.0f")]
                )
                .properties(height=260)
            )
            rules = alt.Chart(pd.DataFrame({
                "label": ["Current", "Median", "5th %", "95th %"],
                "x":     [last_val, p50, p5, p95]
            })).mark_rule(color="red").encode(x="x:Q")
            labels = alt.Chart(pd.DataFrame({
                "text":  ["Current", "Median", "5th %", "95th %"],
                "x":     [last_val, p50, p5, p95],
                "y":     [0, 0, 0, 0]
            })).mark_text(angle=90, dy=-10, color="red").encode(x="x:Q", text="text:N")
            st.altair_chart(hist + rules + labels, use_container_width=True)

st.caption("Quotes cached ~60s; history cached ~10 min. Yahoo first; Stooq fallback for gaps.")
