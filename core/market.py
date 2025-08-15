# core/market.py

import yfinance as yf
import pandas as pd
import numpy as np

from core.iv import implied_vol
from core.bsm import bsm_greeks

def list_expiries(ticker):
    return yf.Ticker(ticker).options

def get_options_chain(ticker, expiry):
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    
    calls = chain.calls.assign(option_type="call")
    puts = chain.puts.assign(option_type="put")
    df = pd.concat([calls, puts], ignore_index=True)
    df["expiry"] = expiry
    df["ticker"] = ticker
    return df

def enrich_chain(df, spot_price, r=0.05, q=0.0):
    T = (pd.to_datetime(df["expiry"]) - pd.Timestamp.today().normalize()) / pd.Timedelta(days=365)
    ivs = implied_vol(
        price=df["lastPrice"].values,
        S=np.full(len(df), spot_price),
        K=df["strike"].values,
        r=np.full(len(df), r),
        q=np.full(len(df), q),
        T=T.values,
        option_type=df["option_type"].values
    )

    df["implied_vol"] = ivs
    df["time_to_expiry"] = T

    greeks_list = []
    for i, row in df.iterrows():
        if np.isnan(row["implied_vol"]) or row["time_to_expiry"] <= 0:
            greeks_list.append({"delta": np.nan, "gamma": np.nan,
                                "theta": np.nan, "vega": np.nan, "rho": np.nan})
            continue
        greeks = bsm_greeks(
            S=spot_price,
            K=row["strike"],
            r=r,
            q=q,
            sigma=row["implied_vol"],
            T=row["time_to_expiry"],
            option_type=row["option_type"]
        )
        greeks_list.append(greeks)

    greeks_df = pd.DataFrame(greeks_list)
    return pd.concat([df.reset_index(drop=True), greeks_df], axis=1)


def get_enriched_chain(ticker, expiry, r=0.05, q=0.0):
    """Fetch and enrich an options chain in one call."""
    df = get_options_chain(ticker, expiry)
    spot_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    return enrich_chain(df, spot_price, r, q)
