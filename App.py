import pandas as pd, numpy as np, yfinance as yf, streamlit as st
st.set_page_config(page_title="Aggressive but Balanced – 1M Tracker", layout="wide")
TICKERS=["OKLO","SHLD","ARKG","CIBR","NVDA","TSLA","URA"]
WEIGHTS={"OKLO":0.16,"SHLD":0.16,"ARKG":0.16,"CIBR":0.16,"NVDA":0.16,"TSLA":0.08,"URA":0.12}
st.sidebar.title("Settings")
nis=st.sidebar.number_input("Portfolio size (₪ NIS)",value=10000.0,step=100.0,min_value=0.0)
fx=st.sidebar.number_input("NIS → USD",value=0.27,step=0.01,min_value=0.0)
total_usd=nis*fx; st.sidebar.write(f"≈ **${total_usd:,.2f} USD**")
st.sidebar.markdown("---"); st.sidebar.write("**Weights** (sum to 1.00)")
weights={}; s=0.0
for t in TICKERS:
    w=st.sidebar.number_input(f"{t} weight",value=float(WEIGHTS[t]),step=0.01,min_value=0.0,max_value=1.0,key=f"w_{t}")
    weights[t]=float(w); s+=float(w)
if abs(s-1.0)>1e-6: st.sidebar.error(f"Weights sum to {s:.2f}"); st.stop()
st.title("Aggressive but Balanced – 1‑Month Portfolio Tracker")
st.caption("Live prices, daily P/L, and allocations. (Info only, not investment advice)")
data=yf.download(" ".join(TICKERS),period="5d",interval="1d",auto_adjust=True,progress=False)
if data.empty: st.error("Price download failed. Try again in a minute."); st.stop()
close=data["Close"] if "Close" in data else data
if isinstance(close,pd.Series): close=close.to_frame()
latest=close.iloc[-1]; prev=close.iloc[-2] if len(close)>=2 else latest
rows=[]
for t in TICKERS:
    price=float(latest.get(t,np.nan)); prevc=float(prev.get(t,price))
    day=((price/prevc)-1.0)*100.0 if prevc>0 else 0.0
    alloc=total_usd*weights[t]; units=alloc/price if price and price>0 else 0.0; val=units*price
    rows.append({"Ticker":t,"Price ($)":round(price,2),"Day %":round(day,2),"Alloc $":round(alloc,2),
                 "Units":round(units,4),"Position Value $":round(val,2)})
tbl=pd.DataFrame(rows); tbl["Δ Value $"]=(tbl["Position Value $"]-tbl["Alloc $"]).round(2)
tbl["Δ %"]=((tbl["Position Value $"]/tbl["Alloc $"]-1.0)*100.0).replace([np.inf,-np.inf],0).fillna(0).round(2)
st.subheader("Holdings"); st.dataframe(tbl, use_container_width=True)
c1,c2,c3=st.columns(3)
tot=float(tbl["Position Value $"].sum()); pl=float(tbl["Δ Value $"].sum())
plpct=(pl/total_usd*100.0) if total_usd>0 else 0.0
c1.metric("Total Value ($)",f"{tot:,.2f}"); c2.metric("Daily P/L ($)",f"{pl:,.2f}"); c3.metric("Daily P/L (%)",f"{plpct:.2f}%")
movers=tbl[tbl["Day %"].abs()>3.0]
if len(movers)>0: st.warning("Movers > 3%: "+", ".join([f"{r.Ticker} ({r['Day %']}%)" for _,r in movers.iterrows()]))
st.subheader("Last 5 sessions – daily returns"); st.line_chart(close.pct_change().dropna())
