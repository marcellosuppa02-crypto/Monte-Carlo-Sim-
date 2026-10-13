import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Monte Carlo Signal", layout="wide")
st.title("Monte Carlo Signal Tool")
st.caption("Educational tool only. Not financial advice. Past volatility does not predict future direction.")

# ---- Sidebar settings ----
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker (e.g. AAPL, TSLA, BTC-USD)", "AAPL").strip().upper()
history = st.sidebar.selectbox("History to learn from", ["1y", "2y", "5y"], index=1)
horizon = st.sidebar.slider("Days ahead to simulate", 5, 60, 20)
n_sims = st.sidebar.select_slider("Number of simulations", [1000, 5000, 10000, 50000], value=10000)
drift_weight = st.sidebar.slider("Trust past trend (0 = ignore, 1 = fully)", 0.0, 1.0, 0.0, 0.1)
long_above = st.sidebar.slider("LONG if chance of rising above", 0.50, 0.70, 0.55, 0.01)
short_below = st.sidebar.slider("SHORT if chance of rising below", 0.30, 0.50, 0.45, 0.01)


@st.cache_data(ttl=3600)
def load(t, period):
    return yf.download(t, period=period, auto_adjust=True, progress=False)


if st.button("Run simulation", type="primary"):
    data = load(ticker, history)
    if data.empty:
        st.error("No data found for that ticker.")
        st.stop()

    prices = data["Close"].squeeze().dropna()
    returns = np.log(prices / prices.shift(1)).dropna().values
    mu = returns.mean() * drift_weight
    sigma = returns.std()

    shocks = np.random.standard_normal((horizon, n_sims))
    daily = (mu - 0.5 * sigma**2) + sigma * shocks
    start = float(prices.iloc[-1])
    paths = start * np.exp(np.cumsum(daily, axis=0))

    final = paths[-1]
    prob_up = float((final > start).mean())
    expected = float(final.mean() / start - 1)
    p5, p50, p95 = np.percentile(final, [5, 50, 95])

    if prob_up > long_above:
        call = "LONG"
    elif prob_up < short_below:
        call = "SHORT"
    else:
        call = "NO CLEAR EDGE"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last price", f"{start:.2f}")
    c2.metric(f"Chance higher in {horizon}d", f"{prob_up:.1%}")
    c3.metric("Average expected change", f"{expected:+.2%}")
    c4.metric("Suggestion", call)
    st.write(f"Bad case (5th percentile): **{p5:.2f}** | Middle: **{p50:.2f}** | Good case (95th percentile): **{p95:.2f}**")

    full = np.vstack([np.full(n_sims, start), paths])
    x = np.arange(0, horizon + 1)
    lo, mid, hi = np.percentile(full, [5, 50, 95], axis=1)
    hist = prices.iloc[-120:]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(range(-len(hist) + 1, 1), hist.values, color="black", label="Past price")
    ax.plot(x, full[:, :200], color="steelblue", alpha=0.05)
    ax.fill_between(x, lo, hi, color="steelblue", alpha=0.2, label="90% range")
    ax.plot(x, mid, color="red", label="Median")
    ax.set_xlabel("Days (0 = today)")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker}: {n_sims:,} simulated futures")
    ax.legend()
    st.pyplot(fig)

    st.info("This reflects only past volatility, not news or real market direction. Use it to understand risk, not as a guarantee.")
else:
    st.write("Pick your settings on the left, then press **Run simulation**.")
