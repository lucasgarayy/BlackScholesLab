import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from core.iv import implied_vol_from_price
from core.bsm import bsm_price, bsm_greeks
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go  # Add this import

st.set_page_config(page_title="Options Pricing Lab", layout="wide")
st.title("Options Pricing Lab — Trader & Researcher View")
# -------------------------------
# Sidebar — App Mode
# -------------------------------
st.sidebar.header("Mode")
mode = st.sidebar.selectbox("Select view", ["Trader", "Researcher"])

# Common defaults
DEFAULTS = dict(
    S=100.0,
    K=100.0,
    T=1.0,
    sigma=0.2,     # 20%
    r=0.015,       # 1.5%
    q=0.0          # 0% dividend yield
)

# -------------------------------
# Sidebar — Inputs (vary by mode)
# -------------------------------
if mode == "Trader":
    st.sidebar.header("Inputs")
    S = st.sidebar.number_input("Spot Price (S)", min_value=0.0, max_value=99999.0, value=DEFAULTS["S"])
    K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, max_value=99999.0, value=DEFAULTS["K"])
    T = st.sidebar.number_input("Time to Expiry (Years, T)", min_value=0.0, value=DEFAULTS["T"])
    sigma = st.sidebar.number_input("Volatility (σ, decimal)", min_value=0.0, value=DEFAULTS["sigma"])
    r = st.sidebar.number_input("Risk-free Rate (r, decimal)", min_value=0.0, value=DEFAULTS["r"])
    q = st.sidebar.number_input("Dividend Yield (q, decimal)", min_value=0.0, value=DEFAULTS["q"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Scenarios")
    scen_cols = st.sidebar.columns(2)
    with scen_cols[0]:
        scen_S_down = st.button("S −10%")
        scen_vol_down = st.button("σ −5 pts")
    with scen_cols[1]:
        scen_S_up = st.button("S +10%")
        scen_vol_up = st.button("σ +5 pts")

    S_disp = S * (0.9 if st.session_state.get("scen_S_down", False) else (1.1 if st.session_state.get("scen_S_up", False) else 1.0))
    sigma_disp = sigma + (-0.05 if st.session_state.get("scen_vol_down", False) else (0.05 if st.session_state.get("scen_vol_up", False) else 0.0))

    if scen_S_down:
        st.session_state["scen_S_down"] = True
    if scen_S_up:
        st.session_state["scen_S_up"] = True
    if scen_vol_down:
        st.session_state["scen_vol_down"] = True
    if scen_vol_up:
        st.session_state["scen_vol_up"] = True

    def _clear_scen_flags():
        for k in ["scen_S_down", "scen_S_up", "scen_vol_down", "scen_vol_up"]:
            if k in st.session_state:
                st.session_state[k] = False

    st.subheader("Trader Dashboard")

    # Remove option type selection
    call_price = bsm_price(S=S, K=K, r=r, q=q, sigma=sigma, T=T, option_type="call")
    put_price = bsm_price(S=S, K=K, r=r, q=q, sigma=sigma, T=T, option_type="put")

    # Add boxes for call and put price above greeks
    price_cols = st.columns(2)
    price_cols[0].markdown(
        f'<div style="background-color:#2563eb;padding:16px;border-radius:6px;color:white;font-weight:normal;">Call Price: {call_price:.4f}</div>',
        unsafe_allow_html=True
    )
    price_cols[1].markdown(
        f'<div style="background-color:#ef4444;padding:16px;border-radius:6px;color:white;font-weight:normal;">Put Price: {put_price:.4f}</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("Greek values")
    # Greeks for call by default
    greeks = bsm_greeks(S=S, K=K, r=r, q=q, sigma=sigma, T=T, option_type="call")

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Delta", f"{greeks.get('delta', np.nan):.4f}")
    kpi_cols[1].metric("Gamma", f"{greeks.get('gamma', np.nan):.6f}")
    kpi_cols[2].metric("Vega", f"{greeks.get('vega', np.nan):.4f}")
    kpi_cols[3].metric("Theta", f"{greeks.get('theta', np.nan):.4f}")
    kpi_cols[4].metric("Rho", f"{greeks.get('rho', np.nan):.4f}")

    st.markdown("---")

    xS = np.linspace(max(0.01, S * 0.5), S * 1.5 if S > 0 else 1.0, 200)
    y_call = [bsm_price(S=s, K=K, r=r, q=q, sigma=sigma, T=T, option_type="call") for s in xS]
    y_put = [bsm_price(S=s, K=K, r=r, q=q, sigma=sigma, T=T, option_type="put") for s in xS]

    # Plotly chart with secondary y-axis for put price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xS, y=y_call, mode='lines', name='Call Price', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=xS, y=y_put, mode='lines', name='Put Price', line=dict(color='firebrick'), yaxis='y2'))

    fig.update_layout(
        xaxis_title="Spot S",
        yaxis=dict(
            title=dict(text="Call Price", font=dict(color="royalblue")),
            tickfont=dict(color="royalblue")
        ),
        yaxis2=dict(
            title=dict(text="Put Price", font=dict(color="firebrick")),
            tickfont=dict(color="firebrick"),
            overlaying='y',
            side='right'
        ),
        title="Call & Put Price vs Spot (today)",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    _clear_scen_flags()

else:
    st.sidebar.header("Inputs")
    S = st.sidebar.number_input("Spot Price (S)", min_value=0.0, max_value=99999.0, value=DEFAULTS["S"])
    K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, max_value=99999.0, value=DEFAULTS["K"])
    T = st.sidebar.number_input("Time to Expiry (Years, T)", min_value=0.0, value=DEFAULTS["T"])
    sigma = st.sidebar.number_input("Volatility (σ, decimal)", min_value=0.0, value=DEFAULTS["sigma"])
    r = st.sidebar.number_input("Risk-free Rate (r, decimal)", min_value=0.0, value=DEFAULTS["r"])
    q = st.sidebar.number_input("Dividend Yield (q, decimal)", min_value=0.0, value=DEFAULTS["q"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Heatmap Axes Controls")
    min_vol, max_vol = st.sidebar.slider(
        "Volatility range (σ): min, max",
        min_value=0.0,
        max_value=3.0,
        value=(0.1, 0.5),
        step=0.01,
    )

    min_S = st.sidebar.number_input("Minimum Spot Price", min_value=0.0, max_value=99999.0, value=75.0)
    max_S = st.sidebar.number_input("Maximum Spot Price", min_value=0.0, max_value=99999.0, value=125.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Resolution")
    nS = st.sidebar.slider("# Spot points", 10, 200, 60, step=10)
    nV = st.sidebar.slider("# Vol points", 10, 200, 60, step=10)

    st.subheader("Researcher Lab")

    tab_price, tab_greeks, tab_smile = st.tabs(["Price Heatmap", "Greeks Heatmap", "Smile / Term Structure"])

    min_S = float(min_S)
    max_S = float(max_S)
    min_vol = float(min_vol)
    max_vol = float(max_vol)
    if max_S <= min_S:
        st.error("Maximum Spot must be greater than Minimum Spot.")
    if max_vol <= min_vol:
        st.error("Maximum volatility must be greater than Minimum volatility.")

    S_grid = np.linspace(max(1e-9, min_S), max_S, nS)
    V_grid = np.linspace(max(1e-9, min_vol), max_vol, nV)

    with tab_price:
        option_type_heat = st.radio("Option type", ["call", "put"], horizontal=True, key="opt_heat")

        call_price_box = bsm_price(S=S, K=K, r=r, q=q, sigma=sigma, T=T, option_type="call")
        st.info(f"**Call Price (current inputs):** {call_price_box:.4f}")

        perspective = st.radio("Participant perspective", ["Buyer (lower price is good)", "Seller (higher price is good)"], horizontal=False)

        S_edges = np.linspace(S_grid[0], S_grid[-1], 11)
        V_edges = np.linspace(V_grid[0], V_grid[-1], 11)
        S_centers = 0.5 * (S_edges[:-1] + S_edges[1:])
        V_centers = 0.5 * (V_edges[:-1] + V_edges[1:])

        Z = np.zeros((10, 10))
        for i, v in enumerate(V_centers):
            for j, s in enumerate(S_centers):
                Z[i, j] = bsm_price(S=s, K=K, r=r, q=q, sigma=v, T=T, option_type=option_type_heat)

        base_price = bsm_price(S=S, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type_heat)
        # Prepare text and color arrays
        text = np.empty_like(Z, dtype=object)
        text_color = np.empty_like(Z, dtype=object)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                price = Z[i, j]
                text[i, j] = f"{price:.2f}"
                if price < base_price:
                    text_color[i, j] = "green"
                else:
                    text_color[i, j] = "red"

        # Plotly heatmap with text labels
        fig = go.Figure(data=go.Heatmap(
            z=Z,
            x=S_centers,
            y=V_centers,
            colorscale='Viridis',
            colorbar=dict(title="Option Price"),
            zmin=np.min(Z),
            zmax=np.max(Z),
            hovertemplate="Spot: %{x:.2f}<br>Vol: %{y:.2f}<br>Price: %{z:.2f}<extra></extra>"
        ))

        # Add text annotations
        for i, v in enumerate(V_centers):
            for j, s in enumerate(S_centers):
                fig.add_annotation(
                    x=s, y=v,
                    text=text[i, j],
                    showarrow=False,
                    font=dict(color=text_color[i, j], size=12),
                    xanchor="center", yanchor="middle"
                )

        fig.update_layout(
            xaxis_title="Spot S",
            yaxis_title="Volatility σ",
            title=f"{option_type_heat.capitalize()} Option Price Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_greeks:
        greek_name = st.selectbox("Greek", ["delta", "gamma", "vega", "theta", "rho"])
        option_type_g = st.radio("Option type", ["call", "put"], horizontal=True, key="opt_greek")

        greek_val_box = bsm_greeks(S=S, K=K, r=r, q=q, sigma=sigma, T=T, option_type=option_type_g).get(greek_name, np.nan)
        st.info(f"**{greek_name.capitalize()} (current inputs):** {greek_val_box:.6f}")

        S_edges = np.linspace(S_grid[0], S_grid[-1], 11)
        V_edges = np.linspace(V_grid[0], V_grid[-1], 11)
        S_centers = 0.5 * (S_edges[:-1] + S_edges[1:])
        V_centers = 0.5 * (V_edges[:-1] + V_edges[1:])

        Zg = np.zeros((10, 10))
        for i, v in enumerate(V_centers):
            for j, s in enumerate(S_centers):
                g = bsm_greeks(S=s, K=K, r=r, q=q, sigma=v, T=T, option_type=option_type_g)
                Zg[i, j] = g.get(greek_name, np.nan)

        # Prepare text and color arrays for Greeks
        text_g = np.empty_like(Zg, dtype=object)
        text_color_g = np.empty_like(Zg, dtype=object)
        for i in range(Zg.shape[0]):
            for j in range(Zg.shape[1]):
                val = Zg[i, j]
                text_g[i, j] = f"{val:.2f}"
                if val < greek_val_box:
                    text_color_g[i, j] = "green"
                else:
                    text_color_g[i, j] = "red"

        # Plotly heatmap for Greeks with text labels
        figg = go.Figure(data=go.Heatmap(
            z=Zg,
            x=S_centers,
            y=V_centers,
            colorscale='Plasma',
            colorbar=dict(title=f"{greek_name.capitalize()}"),
            zmin=np.min(Zg),
            zmax=np.max(Zg),
            hovertemplate="Spot: %{x:.2f}<br>Vol: %{y:.2f}<br>"+f"{greek_name.capitalize()}: "+"%{z:.2f}<extra></extra>"
        ))
        # Add text annotations
        for i, v in enumerate(V_centers):
            for j, s in enumerate(S_centers):
                figg.add_annotation(
                    x=s, y=v,
                    text=text_g[i, j],
                    showarrow=False,
                    font=dict(color=text_color_g[i, j], size=12),
                    xanchor="center", yanchor="middle"
                )
        figg.update_layout(
            xaxis_title="Spot S",
            yaxis_title="Volatility σ",
            title=f"{greek_name.capitalize()} Heatmap — Discrete 10×10 ({option_type_g})"
        )
        st.plotly_chart(figg, use_container_width=True)

    with tab_smile:
        st.markdown("### Implied Volatility Smile")
        src = st.radio("Data source", ["Synthetic", "Upload CSV"], horizontal=True)

        st.info(
            "**CSV format:**\n"
            "- Columns: `K`, `price`, `option_type`\n"
            "- `K`: Strike price (float)\n"
            "- `price`: Option price (float)\n"
            "- `option_type`: 'call' or 'put' (case-insensitive)"
        )

        if src == "Synthetic":
            st.info("No market data? Generate one or two parametric smiles.")
            # Smile 1 params as a list
            smile1 = [
                st.slider("T₁ (years)", 0.01, 5.0, float(T), 0.01),
                st.slider("ATM vol σ₁", 0.01, 2.0, float(sigma), 0.01),
                st.slider("Skew₁ (vol pts per log-moneyness)", -1.0, 1.0, -0.20, 0.01),
                st.slider("Curvature₁", 0.0, 2.0, 0.20, 0.01)
            ]
            # Set initial value to False so second smile is off by default
            add_second = st.checkbox("Add second smile (T₂, σ₂, skew₂, curvature₂)", value=False)
            smile2 = None
            if add_second:
                smile2 = [
                    st.slider("T₂ (years)", 0.01, 5.0, max(0.25, float(T)), 0.01, key="T2_slider"),
                    st.slider("ATM vol σ₂", 0.01, 2.0, max(0.01, float(sigma) * 0.9), 0.01, key="base_sigma2_slider"),
                    st.slider("Skew₂", -1.0, 1.0, -0.10, 0.01, key="skew2_slider"),
                    st.slider("Curvature₂", 0.0, 2.0, 0.15, 0.01, key="curvature2_slider")
                ]

            nK = st.slider("# Strikes", 5, 60, 21)
            # Smile 1
            T1, base_sigma1, skew1, curvature1 = smile1
            F1 = S * np.exp((r - q) * T1)
            K_grid1 = np.linspace(max(1e-6, 0.5 * F1), 1.5 * F1 if F1 > 0 else 1.0, nK)
            x1 = np.log(K_grid1 / F1)
            iv1 = np.clip(base_sigma1 + skew1 * x1 + curvature1 * x1**2, 1e-4, 5.0)

            fig3, ax3 = plt.subplots()
            ax3.plot(K_grid1, iv1, label=f"Smile 1 (T={T1:.2f}y)")

            # Optional Smile 2
            if smile2 is not None:
                T2, base_sigma2, skew2, curvature2 = smile2
                F2 = S * np.exp((r - q) * T2)
                K_grid2 = np.linspace(max(1e-6, 0.5 * F2), 1.5 * F2 if F2 > 0 else 1.0, nK)
                x2 = np.log(K_grid2 / F2)
                iv2 = np.clip(base_sigma2 + skew2 * x2 + curvature2 * x2**2, 1e-4, 5.0)
                ax3.plot(K_grid2, iv2, linestyle="--", label=f"Smile 2 (T={T2:.2f}y)")

            ax3.set_xlabel("Strike K")
            ax3.set_ylabel("Implied Volatility σ")
            ax3.set_title("Synthetic IV Smiles")
            ax3.legend()
            st.pyplot(fig3, use_container_width=True)

        else:
            st.write("Upload a CSV with columns: K, price, option_type (call/put).")
            file = st.file_uploader("CSV file", type=["csv"])
            if file is not None:
                try:
                    df = pd.read_csv(file)
                    required = {"K", "price", "option_type"}
                    if not required.issubset(set(df.columns)):
                        st.error("CSV must contain columns: K, price, option_type")
                    else:
                        strikes = df["K"].to_numpy(dtype=float)
                        prices = df["price"].to_numpy(dtype=float)
                        types = df["option_type"].astype(str).str.lower().to_list()

                        ivs = []
                        for K_i, p_i, t_i in zip(strikes, prices, types):
                            iv_i = implied_vol_from_price(
                                p_i, S=S, K=K_i, r=r, q=q, T=T, option_type=t_i
                            )
                            ivs.append(iv_i)
                        df_out = pd.DataFrame({"K": strikes, "IV": ivs, "type": types})

                        # Plot calls and puts as two curves/series
                        fig4, ax4 = plt.subplots()
                        calls = df_out[df_out["type"] == "call"].dropna(subset=["IV"])
                        puts = df_out[df_out["type"] == "put"].dropna(subset=["IV"])

                        # If there are multiple strikes, sort them for a nice curve
                        if not calls.empty:
                            calls = calls.sort_values("K")
                            ax4.plot(calls["K"], calls["IV"], marker="o", label="Calls")
                        if not puts.empty:
                            puts = puts.sort_values("K")
                            ax4.plot(puts["K"], puts["IV"], marker="x", linestyle="--", label="Puts")

                        # Fallback: if types missing/mixed, still show scatter
                        if calls.empty and puts.empty:
                            ax4.scatter(df_out["K"], df_out["IV"], s=24, label="All options")

                        ax4.set_xlabel("Strike K")
                        ax4.set_ylabel("Implied Volatility σ")
                        ax4.set_title("Implied Vol Smiles (from CSV)")
                        ax4.legend()
                        st.pyplot(fig4, use_container_width=True)

                        st.dataframe(df_out)
                except Exception as e:
                    st.error(f"Failed to parse/plot CSV: {e}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '''
    <div style="text-align:center; color:gray; font-size:16px; margin-top:24px;">
        Made by Lucas Sánchez Garay<br>
        <a href="https://www.linkedin.com/in/lucas-s%C3%A1nchez-garay-9a3592198/" target="_blank" style="color:#0A66C2;text-decoration:none;">
            <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" alt="LinkedIn" style="height:18px;vertical-align:middle;margin-right:6px;">
            LinkedIn
        </a>
    </div>
    ''',
    unsafe_allow_html=True
)
