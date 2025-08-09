# ---------- Portfolio value over time (no rebalancing) ----------
st.subheader("Portfolio value over time")

# Fetch aligned history
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
        # Units = starting USD allocation / price at first date
        first_row = prices.iloc[0]
        units = {}
        for t in TICKERS:
            if t in prices.columns and first_row[t] and not math.isnan(first_row[t]) and first_row[t] > 0:
                units[t] = (total_usd * weights[t]) / first_row[t]
            else:
                units[t] = 0.0

        # Value series in USD
        port_val = (prices * pd.Series(units)).sum(axis=1)
        port_val.name = "Portfolio Value ($)"

        # % series vs start
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

        # --- Charts (two synced views) ---
        st.markdown("**USD value**")
        st.line_chart(port_val)

        st.markdown("**% change from start**")
        st.line_chart(port_pct)

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
