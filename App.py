import math, time, requests
import numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Aggressive but Balanced – 1M Tracker", layout="wide")
st.write("✅ App booted… loading quotes")

# ---- CONFIG ----
TICKERS = ["OKLO","SHLD","ARKG","CIBR","NVDA","TSLA","URA"]
DEFAULT_WEIGHTS = {"OKLO":0.16,"SHLD":0.16,"ARKG":0.16,"CIBR":0.16,"NVDA":0.16,"TSLA":0.08,"URA":0.12}
DEFAULT_NIS = 10000.0
DEFAULT_FX  = 0.27  # NIS→USD

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

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

st.title("Aggressive but Balanced – 1-Month Portfolio Tracker")
st.caption("Live prices, P/L and allocations. Info only, not investment advice.")

# ---------- Fetchers ----------
def _quote_call(symbols: list[str]) -> dict:
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    r = requests.get(url, params={"symbols": ",".join(symbols)}, headers=UA, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_quotes(symbols: list[str]) -> dict[str, tuple[float, float]]:
    # retry up to 3 times with small backoff
    last_err = None
    for i in range(3):
        try:
            data = _quote_call(symbols)
            result = data.get("quoteResponse", {}).get("result", [])
            out = {}
            for d in result:
                sym = d.get("symbol")
                price = d.get("regularMarketPrice")
                prev  = d.get("regularMarketPreviousClose", price)
                if sym:
                    p = float(price) if price is not None else float("nan")
                    pc = float(prev)  if prev  is not None else float("nan")
                    out[sym] = (p, pc)
            # ensure keys for all symbols
            for s in symbols:
                out.setdefault(s, (float("nan"), float("nan")))
            return out
        except Exception as e:
            last_err = e
            time.sleep(0.8 * (i + 1))
    raise last_err

# ---------- Quotes ----------
try:
    quotes = fetch_quotes(TICKERS)
except Exception as e:
    st.error(f"Price download failed. Please refresh in a minute. Details: {e}")
    st.stop()

# ---------- Table ----------
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

st.subheader("Holdings")
st.dataframe(tbl, use_container_width=True)

c1, c2, c3 = st.columns(3)
total_val = float(tbl["Position Value $"].sum())
pl_val    = float(tbl["Δ Value $"].sum())
pl_pct    = (pl_val / total_usd * 100.0) if total_usd > 0 else 0.0
c1.metric("Total Value ($)", f"{total_val:,.2f}")
c2.metric("Daily P/L ($)",   f"{pl_val:,.2f}")
c3.metric("Daily P/L (%)",   f"{pl_pct:.2f}%")

# Optional: spark chart can be added later; focusing on stable quotes first.
st.caption("Quotes cached ~60s. If prices don't appear, tap the ↻ in the top-right menu to rerun.")
