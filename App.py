import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Aggressive but Balanced – 1M Tracker", layout="wide")

# ---- CONFIG ----
TICKERS = ["OKLO", "SHLD", "ARKG", "CIBR", "NVDA", "TSLA", "URA"]
DEFAULT_WEIGHTS = {"OKLO":0.16,"SHLD":0.16,"ARKG":0.16,"CIBR":0.16,"NVDA":0.16,"TSLA":0.08,"URA":0.12}
DEFAULT_NIS = 10000.0
DEFAULT_FX  = 0.27  # NIS→USD

# ---------- Sidebar ----------
st.sidebar.title("Settings")
nis = st.sidebar.number_input("Portfolio size (₪ NIS)", value=DEFAULT_NIS, step=100.0, min_value=0.0)
fx  = st.sidebar.number_input("NIS → USD", value=DEFAULT_FX, step=0.01, min_value=0.0)
total_usd = nis * fx
st.sidebar.write(f"≈ **${total_usd:,.2f} USD**")
st.sidebar.markdown("---")
st.sidebar.write("**Weights** (sum to 1.00)")

weights = {}
sum_w = 0.0
for t in TICKERS:
    w = st.sidebar.number_input(f"{t} weight", value=float(DEFAULT_WEIGHTS[t]), step=0.01,
                                min_value=0.0, max_value=1.0, key=f"w_{t}")
    weights[t] = float(w)
    sum_w += float(w)

if abs(sum_w - 1.0) > 1e-6:
    st.sidebar.error(f"Weights sum to {sum_w:.2f}. They must sum to 1.00.")
    st.stop()

# ---------- Title ----------
st.title("Aggressive but Balanced – 1-Month Portfolio Tracker")
st.caption("Live prices, P/L and allocations. Info only, not investment advice.")

# ---------- Data fetchers (requests) ----------
@st.cache_data(ttl=60)
def fetch_quotes(symbols: list[str]) -> dict[str, tuple[float, float]]:
    """
    Returns dict: symbol -> (last_price, previous_close).
    Uses Yahoo Finance quote API (no key).
    """
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    r = requests.get(url, params={"symbols": ",".join(symbols)}, timeout=10)
    r.raise_for_status()
    result = r.json().get("quoteResponse", {}).get("result", [])
    out = {}
    for d in result:
        sym = d.get("symbol")
        price = d.get("regularMarketPrice")
        prev  = d.get("regularMarketPreviousClose", price)
        if sym:
            p = float(price) if price is not None else float("nan")
            pc = float(prev)  if prev  is not None else float("nan")
            out[sym] = (p, pc)
    # ensure all requested symbols present (even if missing)
    for s in symbols:
        out.setdefault(s, (float("nan"), float("nan")))
    return out

@st.cache_data(ttl=120)
def fetch_spark(symbols: list[str], range_code: str = "5d", interval: str = "1d") -> pd.DataFrame:
    """
    Tiny history for sparkline chart. Returns DataFrame with columns per symbol.
    """
    # Yahoo "spark" endpoint (undocumented but stable for small use)
    url = "https://query1.finance.yahoo.com/v8/finance/spark"
    r = requests.get(url, params={"symbols": ",".join(symbols),
                                  "range": range_code, "interval": interval}, timeout=10)
    r.raise_for_status()
    data = r.json().get("spark", {}).get("result", [])
    frames = {}
    for item in data:
        sym = item.get("symbol")
        series = item.get("response", [{}])[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        frames[sym] = pd.Series(series, dtype="float64")
    # Align length
    df = pd.DataFrame(frames)
    return df

# ---------- Fetch quotes ----------
try:
    quotes = fetch_quotes(TICKERS)
except Exception as e:
    st.error(f"Price download failed. Try again in a minute. ({e})")
    st.stop()

# ---------- Build table ----------
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

# ---------- Display ----------
st.subheader("Holdings")
st.dataframe(tbl, use_container_width=True)

c1, c2, c3 = st.columns(3)
total_val = float(tbl["Position Value $"].sum())
pl_val    = float(tbl["Δ Value $"].sum())
pl_pct    = (pl_val / total_usd * 100.0) if total_usd > 0 else 0.0
c1.metric("Total Value ($)", f"{total_val:,.2f}")
c2.metric("Daily P/L ($)",   f"{pl_val:,.2f}")
c3.metric("Daily P/L (%)",   f"{pl_pct:.2f}%")

# ---------- Spark chart (last 5 sessions) ----------
try:
    spark = fetch_spark(TICKERS, range_code="5d", interval="1d").pct_change().dropna()
    if not spark.empty:
        st.subheader("Last 5 sessions – daily returns")
        st.line_chart(spark)
except Exception:
    # Silently skip chart if spark endpoint hiccups
    pass

# Highlight movers
movers = tbl[ tbl["Day %"].abs() > 3.0 ]
if len(movers) > 0:
    st.warning("Movers > 3%: " + ", ".join([f"{r.Ticker} ({r['Day %']}%)" for _, r in movers.iterrows()]))

st.caption("Quotes cached ~60s. Weights must sum to 1.00.")

