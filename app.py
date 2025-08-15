# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

from core.market import list_expiries, get_enriched_chain
from core.bsm import bsm_price

# --------------------
# Page config
# --------------------
st.set_page_config(page_title="Options IV Explorer", layout="wide")

# --------------------
# Helpers
# --------------------
def _moving_avg(y: np.ndarray, w: int = 5) -> np.ndarray:
    """Simple moving average with edge handling."""
    w = max(1, int(w))
    if y.size < w:
        return y
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    out = np.convolve(ypad, kernel, mode="valid")
    if out.size > y.size:
        out = out[:y.size]
    return out

def _atm_strike(strikes: np.ndarray, spot: float) -> float:
    idx = int(np.argmin(np.abs(strikes - spot)))
    return float(strikes[idx])

# --------------------
# Data fetchers (cached)
# --------------------
@st.cache_data(show_spinner=False, ttl=600)
def fetch_spot(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="1d")
    return float(hist["Close"].iloc[-1])

@st.cache_data(show_spinner=True, ttl=600)
def fetch_enriched(ticker: str, expiry: str, r: float, q: float) -> pd.DataFrame:
    df = get_enriched_chain(ticker, expiry, r=r, q=q)
    if {"bid", "ask"}.issubset(df.columns):
        valid = (~df["bid"].isna()) & (~df["ask"].isna()) & (df["bid"] > 0) & (df["ask"] > 0)
        mid = np.where(valid, 0.5 * (df["bid"].values + df["ask"].values), df["lastPrice"].values)
        df.insert(df.columns.get_loc("lastPrice") + 1, "midPrice", mid)
    return df

# --------------------
# Sidebar controls
# --------------------
st.sidebar.title("Controls")
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()
r = st.sidebar.number_input("Risk-free rate (cont., yearly)", value=0.05, step=0.005, format="%.3f")
q = st.sidebar.number_input("Dividend yield (cont., yearly)", value=0.00, step=0.005, format="%.3f")

expiries = []
if ticker:
    try:
        expiries = list_expiries(ticker)
    except Exception as e:
        st.sidebar.error(f"Could not fetch expiries for {ticker}: {e}")

expiry = st.sidebar.selectbox("Expiry", options=expiries or ["—"], index=0)
st.sidebar.button("Refresh")

# --- in the sidebar: Call price heatmap (σ × K) controls ---
with st.sidebar.expander("Call price heatmap (σ × K)", expanded=False):
    sigma_min, sigma_max = st.slider("σ range", 0.01, 2.00, (0.10, 0.60), 0.01)
    n_sigmas = st.slider("σ steps", 10, 200, 60, 10)
    n_strikes = st.slider("K steps", 20, 400, 160, 10)
    show_contours_ck = st.checkbox("Show contour lines", value=True, key="ck_contours")

# --- in the sidebar: Price heatmap (Spot × σ) controls ---
with st.sidebar.expander("Price heatmap (Spot × σ)", expanded=False):
    opt_type = st.selectbox("Option type", ["call", "put"], index=0, key="sv_opt_type")
    spot_lo_mult, spot_hi_mult = st.slider("Spot range (× spot)", 0.5, 1.5, (0.85, 1.15), 0.01)
    n_spots = st.slider("Spot steps", 5, 60, 21, 1)
    vol_min, vol_max = st.slider("σ range", 0.05, 1.00, (0.10, 0.40), 0.01)
    n_vols = st.slider("σ steps", 5, 60, 21, 1)
    show_values_sv = st.checkbox("Annotate cell values", value=False, key="sv_values")
    show_contours_sv = st.checkbox("Show contour lines", value=True, key="sv_contours")

# --------------------
# Top info
# --------------------
st.title("Options IV Explorer")
st.caption("Browse real chains, compute implied vols & Greeks, and visualize the smile.")

if not expiries:
    st.info("Enter a valid ticker to load expiries.")
    st.stop()

spot = fetch_spot(ticker)
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Ticker", ticker)
kpi2.metric("Spot", f"{spot:,.2f}")
kpi3.metric("Expiry", expiry)

# --------------------
# Load chain
# --------------------
df = fetch_enriched(ticker, expiry, r=r, q=q) if expiry and expiry != "—" else pd.DataFrame()
if df.empty:
    st.warning("No data for this expiry.")
    st.stop()

df = df.copy()
df["moneyness"] = df["strike"] / spot
df["px_for_iv"] = df["midPrice"] if "midPrice" in df.columns else df["lastPrice"]

# --------------------
# Layout tabs
# --------------------
tab_chain, tab_smile, tab_heat1, tab_heat2, tab_custom = st.tabs(
    ["📋 Chain", "📈 Smile", "🔥 Call heatmap (σ × K)", "🔥 Price heatmap (Spot × σ)", "🧮 Custom BSM Surface"]
)

# --------------------
# Chain table
# --------------------
with tab_chain:
    st.subheader("Options chain (enriched)")
    cols = [
        "contractSymbol", "option_type", "strike", "moneyness", "time_to_expiry",
        "lastPrice", "bid", "ask"
    ]
    if "midPrice" in df.columns:
        cols.append("midPrice")
    cols += ["implied_vol","delta","gamma","theta","vega","rho"]
    show_cols = [c for c in cols if c in df.columns]
    df_show = df[show_cols].sort_values(["option_type","strike"]).reset_index(drop=True)
    st.dataframe(df_show, use_container_width=True)
    csv = df_show.to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv, file_name=f"{ticker}_{expiry}_enriched.csv", mime="text/csv")

# --------------------
# Smile
# --------------------
with tab_smile:
    st.subheader("Volatility smile")
    plot_df = df[["strike","implied_vol","option_type","moneyness"]].dropna()
    if plot_df.empty:
        st.info("No implied vols available to plot (quotes may be stale for this expiry). Try another one.")
    else:
        fig = go.Figure()
        for typ, sym in [("call", "Calls"), ("put", "Puts")]:
            sub = plot_df[plot_df["option_type"] == typ].sort_values("strike")
            if sub.empty:
                continue
            x = sub["strike"].to_numpy(float)
            y = sub["implied_vol"].to_numpy(float)
            y_sm = _moving_avg(y, w=max(3, len(y)//15))
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=f"{sym} (pts)",
                hovertemplate="K=%{x:.2f}<br>IV=%{y:.4f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y_sm, mode="lines", name=f"{sym} (smooth)", hoverinfo="skip", line=dict(width=2)
            ))
        k_atm = _atm_strike(plot_df["strike"].unique(), spot)
        fig.add_vline(x=k_atm, line_dash="dot", opacity=0.5)
        fig.update_layout(
            title=f"{ticker} {expiry} — IV Smile",
            xaxis_title="Strike (K)", yaxis_title="Implied Volatility (σ)",
            legend_orientation="h", legend_y=-0.2, margin=dict(l=10,r=10,t=60,b=10),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show IV vs Moneyness (K/S)"):
            fig2 = px.scatter(
                plot_df, x="moneyness", y="implied_vol", color="option_type",
                labels={"moneyness":"K/S", "implied_vol":"σ"}, title="IV vs Moneyness"
            )
            for typ in ["call","put"]:
                sub = plot_df[plot_df["option_type"] == typ].sort_values("moneyness")
                if sub.empty: continue
                xg = np.linspace(sub["moneyness"].min(), sub["moneyness"].max(), 100)
                yg = _moving_avg(np.interp(xg, sub["moneyness"], sub["implied_vol"]), w=7)
                fig2.add_traces(go.Scatter(x=xg, y=yg, mode="lines", name=f"{typ} smooth", showlegend=True))
            fig2.update_layout(margin=dict(l=10,r=10,t=60,b=10), legend_orientation="h", legend_y=-0.25)
            st.plotly_chart(fig2, use_container_width=True)

# --------------------
# Heatmap 1: Call price over (σ × K)
# --------------------
with tab_heat1:
    st.subheader("Call price heatmap (BSM) — σ × K")
    k_min = float(df["strike"].min()); k_max = float(df["strike"].max())
    K_grid = np.linspace(k_min, k_max, int(n_strikes))
    sig_grid = np.linspace(float(sigma_min), float(sigma_max), int(n_sigmas))
    T_val = float(df["time_to_expiry"].iloc[0])
    Z = np.empty((sig_grid.size, K_grid.size), dtype=float)
    for i, sig in enumerate(sig_grid):
        Z[i, :] = bsm_price(S=spot, K=K_grid, r=r, q=q, sigma=sig, T=T_val, option_type="call")
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=Z, x=K_grid, y=sig_grid, colorbar=dict(title="Price"), colorscale="Turbo",
        hovertemplate="K=%{x:.2f}<br>σ=%{y:.3f}<br>Price=%{z:.4f}<extra></extra>"
    ))
    if show_contours_ck:
        fig.add_trace(go.Contour(z=Z, x=K_grid, y=sig_grid,
                                 contours=dict(showlines=True, coloring="none"),
                                 showscale=False, line_width=1))
    fig.add_vline(x=_atm_strike(K_grid, spot), line_dash="dot", opacity=0.5)
    iv_med = float(np.nanmedian(df["implied_vol"])) if "implied_vol" in df.columns else None
    if iv_med and np.isfinite(iv_med):
        fig.add_hline(y=iv_med, line_dash="dot", opacity=0.5)
    fig.update_layout(
        xaxis_title="Strike (K)", yaxis_title="Volatility (σ, annualized)",
        title=f"{ticker} {expiry} — Call price heatmap (BSM)", margin=dict(l=10,r=10,t=60,b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Heatmap 2: Option price over (Spot × σ) @ fixed K (chain-driven)
# --------------------
with tab_heat2:
    st.subheader(f"Option price heatmap (BSM) — Spot × σ @ fixed K ({opt_type.upper()})")
    strikes_sorted = np.sort(df["strike"].unique())
    default_strike = float(strikes_sorted[np.argmin(np.abs(strikes_sorted - spot))])
    K = st.number_input("Strike (K)", value=default_strike, step=1.0, format="%.2f", key="heat2_K")
    S_grid = np.linspace(spot * float(spot_lo_mult), spot * float(spot_hi_mult), int(n_spots))
    sig_grid2 = np.linspace(float(vol_min), float(vol_max), int(n_vols))
    T_val2 = float(df["time_to_expiry"].iloc[0])
    Z2 = np.empty((sig_grid2.size, S_grid.size), dtype=float)
    for i, sig in enumerate(sig_grid2):
        Z2[i, :] = bsm_price(S=S_grid, K=K, r=r, q=q, sigma=sig, T=T_val2, option_type=opt_type)
    fig2 = go.Figure()
    fig2.add_trace(go.Heatmap(
        z=Z2, x=S_grid, y=sig_grid2, colorbar=dict(title="Price"), colorscale="Turbo",
        hovertemplate="S=%{x:.2f}<br>σ=%{y:.3f}<br>Price=%{z:.4f}<extra></extra>"
    ))
    if show_contours_sv:
        fig2.add_trace(go.Contour(
            z=Z2, x=S_grid, y=sig_grid2, contours=dict(showlines=True, coloring="none"),
            showscale=False, line_width=1
        ))
    fig2.add_vline(x=spot, line_dash="dot", opacity=0.5, annotation_text="Spot", annotation_position="top")
    fig2.update_layout(
        xaxis_title="Spot (S)", yaxis_title="Volatility (σ, annualized)",
        title=f"{ticker} {expiry} — {opt_type.upper()} price heatmap at K={K:.2f}",
        margin=dict(l=10,r=10,t=60,b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

# --------------------
# NEW: Custom BSM Surface — independent inputs; heat = BS price over (S × σ)
# --------------------
with tab_custom:
    st.subheader("Custom BSM Surface — Spot × σ (heat = price)")

    # Controls (local, independent of chain)
    c1, c2, c3 = st.columns(3)
    with c1:
        custom_opt = st.selectbox("Option type", ["call", "put"], key="custom_opt")
        S0 = st.number_input("Asset price S₀", value=float(spot), min_value=0.0, step=1.0, format="%.4f", key="custom_S0")
        Kc = st.number_input("Strike K", value=float(_atm_strike(np.sort(df['strike'].unique()), spot)), min_value=0.0, step=1.0, format="%.4f", key="custom_K")
    with c2:
        rc = st.number_input("Risk-free r (cont., yearly)", value=float(r), step=0.005, format="%.4f", key="custom_r")
        qc = st.number_input("Dividend yield q (cont., yearly)", value=float(q), step=0.005, format="%.4f", key="custom_q")
        T_years = st.number_input("Time to maturity T (years)", value=float(df["time_to_expiry"].iloc[0]), min_value=0.0001, step=0.05, format="%.4f", key="custom_T")
    with c3:
        sigma_center = st.number_input("Center σ*", value=float(np.nanmedian(df["implied_vol"])) if "implied_vol" in df.columns and np.isfinite(np.nanmedian(df["implied_vol"])) else 0.25,
                                       min_value=0.001, step=0.01, format="%.4f", key="custom_sigma_center")
        s_mult_lo, s_mult_hi = st.slider("Spot range (× S₀)", 0.1, 3.0, (0.75, 1.25), 0.01, key="custom_S_range")
        v_span = st.slider("σ range around σ*", 0.01, 1.50, 0.30, 0.01, key="custom_sigma_span")

    # Resolution and rendering options
    r1, r2, r3 = st.columns(3)
    with r1:
        nS = st.slider("Spot steps", 5, 200, 101, 1, key="custom_nS")
    with r2:
        nV = st.slider("σ steps", 5, 200, 81, 1, key="custom_nV")
    with r3:
        show_contours_custom = st.checkbox("Show contour lines", value=True, key="custom_contours")

    # Build grids from independent inputs
    S_grid_c = np.linspace(S0 * float(s_mult_lo), S0 * float(s_mult_hi), int(nS))
    sig_min_c = max(1e-6, float(sigma_center - v_span/2))
    sig_max_c = float(sigma_center + v_span/2)
    sig_grid_c = np.linspace(sig_min_c, sig_max_c, int(nV))

    # Compute price surface Zc: rows = vol, cols = spot
    Zc = np.empty((sig_grid_c.size, S_grid_c.size), dtype=float)
    for i, sig in enumerate(sig_grid_c):
        Zc[i, :] = bsm_price(S=S_grid_c, K=Kc, r=rc, q=qc, sigma=sig, T=T_years, option_type=custom_opt)

    # Heatmap
    figc = go.Figure()
    figc.add_trace(go.Heatmap(
        z=Zc, x=S_grid_c, y=sig_grid_c, colorscale="Turbo", colorbar=dict(title="Price"),
        hovertemplate="S=%{x:.4f}<br>σ=%{y:.4f}<br>Price=%{z:.6f}<extra></extra>"
    ))
    if show_contours_custom:
        figc.add_trace(go.Contour(
            z=Zc, x=S_grid_c, y=sig_grid_c, contours=dict(showlines=True, coloring="none"),
            showscale=False, line_width=1
        ))

    # Visual guides at S₀ and σ*
    figc.add_vline(x=S0, line_dash="dot", opacity=0.6)
    figc.add_hline(y=sigma_center, line_dash="dot", opacity=0.6)

    figc.update_layout(
        title=f"BSM {custom_opt.upper()} price over Spot × σ at K={Kc:.4f}, T={T_years:.4f}, r={rc:.4f}, q={qc:.4f}",
        xaxis_title="Spot (S)",
        yaxis_title="Volatility (σ, annualized)",
        margin=dict(l=10,r=10,t=60,b=10)
    )
    st.plotly_chart(figc, use_container_width=True)

    # Optional: quick snapshot table at (S₀, σ*)
    with st.expander("Show price at (S₀, σ*) and small neighborhood"):
        S_snap = np.array([S0 * m for m in [0.95, 1.0, 1.05]])
        V_snap = np.array([max(1e-6, sigma_center * m) for m in [0.9, 1.0, 1.1]])
        snap = []
        for vv in V_snap:
            prices = bsm_price(S=S_snap, K=Kc, r=rc, q=qc, sigma=vv, T=T_years, option_type=custom_opt)
            for s_i, p in zip(S_snap, prices if hasattr(prices, "__len__") else [prices]):
                snap.append({"S": s_i, "σ": vv, "Price": p})
        st.dataframe(pd.DataFrame(snap), use_container_width=True)
