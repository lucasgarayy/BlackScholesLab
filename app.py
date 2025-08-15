import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import yfinance as yf

from core.market import list_expiries, get_enriched_chain
from core.bsm import bsm_price

st.set_page_config(page_title="Options Explorer")

st.title("Options Explorer")

# Sidebar inputs
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()

expiries = list_expiries(ticker) if ticker else []
expiry = st.sidebar.selectbox("Expiration", expiries) if expiries else None

if ticker and expiry:
    try:
        df = get_enriched_chain(ticker, expiry)
    except Exception as e:
        st.error(f"Failed to fetch option data: {e}")
    else:
        if df.empty:
            st.warning("No option data for this expiry.")
        else:
            spot = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
            strikes = np.sort(df["strike"].unique())
            default_idx = int(np.argmin(np.abs(strikes - spot)))
            strike = st.sidebar.selectbox("Strike", strikes, index=default_idx)
            T = (pd.to_datetime(expiry) - pd.Timestamp.today().normalize()) / pd.Timedelta(days=365)

            # Grids for spot and volatility
            spot_vals = np.linspace(0.5 * spot, 1.5 * spot, 50)
            iv_vals = np.linspace(0.05, 0.8, 50)
            S_grid, V_grid = np.meshgrid(spot_vals, iv_vals)

            call_prices = bsm_price(S=S_grid, K=strike, r=0.05, q=0.0, sigma=V_grid, T=T, option_type="call")
            put_prices = bsm_price(S=S_grid, K=strike, r=0.05, q=0.0, sigma=V_grid, T=T, option_type="put")

            st.subheader("Call price heatmap")
            fig_call = px.imshow(call_prices,
                                 x=spot_vals,
                                 y=iv_vals,
                                 origin="lower",
                                 labels={"x": "Spot price", "y": "Implied volatility", "color": "Price"})
            st.plotly_chart(fig_call, use_container_width=True)

            st.subheader("Put price heatmap")
            fig_put = px.imshow(put_prices,
                                x=spot_vals,
                                y=iv_vals,
                                origin="lower",
                                labels={"x": "Spot price", "y": "Implied volatility", "color": "Price"})
            st.plotly_chart(fig_put, use_container_width=True)

            st.subheader("Volatility smile")
            smile_df = df[["strike", "implied_vol", "option_type"]].dropna()
            if smile_df.empty:
                st.info("No implied volatilities to display.")
            else:
                fig_smile = px.scatter(smile_df,
                                       x="strike",
                                       y="implied_vol",
                                       color="option_type",
                                       labels={"implied_vol": "Implied Volatility", "strike": "Strike"})
                st.plotly_chart(fig_smile, use_container_width=True)
else:
    st.info("Enter a ticker and select an expiration to view data.")
