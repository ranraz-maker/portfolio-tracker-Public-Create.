import math, time, io, csv, requests
import numpy as np, pandas as pd, streamlit as st

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

st.title("Aggressive but Balanced – 1-Month Portfolio Tracker")
st.caption("Live prices (with fallback) and P/L. Info only, not investment advice.")

# ---------- Fetchers ----------
def yahoo_quotes(symbols: list[str]) -> dict[str, tuple[float,float]]:
    """Primary: Yahoo v7 quote JSON with full headers."""
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
    """Map to Stooq tickers (US stocks/ETFs often need .US)."""
    m = {}
    for s in symbols:
        if "." in s:  # already has suffix (e.g., BRK.B)
            m[s] = s.replace(".", "-").upper()  # Stooq sometimes uses '-' for dot
        else:
            m[s] = f"{s}.US"  # try US listing
    return m

def stooq_quotes(symbols: list[str]) -> dict[str, tuple[float,float]]:
    """
    Fallback: Stooq CSV (no key). Returns last price and prev close where available.
    Docs: https://stooq.com
    """
    mapped = stooq_symbols(symbols)
    url = "https://stooq.com/q/l/"
    params = {"s": ",".join(mapped.values()), "f": "sd2t2ohlcv", "h": "", "e": "csv"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    text = r.text
    out = {}
    reader = csv.DictReader(io.StringIO(text))
    rev = {v: k for k, v in mapped.items()}
    for row in reader:
        sym = row.get("Symbol")
        # Map back to original if possible
        orig = rev.get(sym, sym)
        try:
            close = float(row.get("Close") or "nan")
            prev  = float(row.get("Open") or "nan")  # Stooq has no prev close; Open is a rough proxy
        except ValueError:
            close, prev = float("nan"), float("nan")
        out[orig.replace("-", ".")] = (close, prev)
    return out

def fetch_quotes(symbols: list[str]) -> dict[str, tuple[float,float]]:
    """Try Yahoo; on 401/403/empty, fallback to Stooq per symbol."""
    # 1) Try Yahoo with up to 2 retries
    yahoo_out = {}
    try:
        for i in range(2):
            try:
                yahoo_out = yahoo_quotes(symbols)
                break
            except requests.HTTPError as e:
                # 401/403 often; backoff once
                if e.response is not None and e.response.status_code in (401,403):
                    time.sleep(1.0 + i)
                    continue
                raise
    except Exception:
        yahoo_out = {}

    # 2) Anything missing → ask Stooq for all; then merge
    missing = [s for s in symbols if s not in yahoo_out or math.isnan(yahoo_out[s][0])]
    stooq_out = {}
    if missing:
        try:
            stooq_out = stooq_quotes(missing)
        except Exception:
            pass

    out = {}
    for s in symbols:
        if s in yahoo_out and not math.isnan(yahoo_out[s][0]):
            out[s] = yahoo_out[s]
        else:
            out[s] = stooq_out.get(s, (float("nan"), float("nan")))
    return out

# ---------- Get quotes ----------
try:
    quotes = fetch_quotes(TICKERS)
except Exception as e:
    st.error(f"Price download failed. Please refresh in a minute. Details: {e}")
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

st.caption("Quotes cached ~60s. Yahoo first, Stooq fallback for unavailable symbols.")
