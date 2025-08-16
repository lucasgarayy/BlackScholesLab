# Options Pricing Lab — Trader & Researcher View

[Launch App](https://blackscholeslab.streamlit.app/)  

An interactive app for pricing European options with Black–Scholes–Merton, exploring Greeks, and visualizing implied-volatility smiles/term structures. Designed with two perspectives:

- **Trader View** — quick pricing, Greeks, and scenario testing  
- **Researcher View** — heatmaps, implied volatility smiles, and CSV upload support

---

## ✨ Features

### Trader mode
- Inputs: $S, K, T, \sigma, r, q$  
- One-click **Quick Scenarios**: $S \pm 10\%$, $\sigma \pm 5$ vol pts  
- Live **Call/Put price** tiles  
- **Greeks** (call by default): Delta, Gamma, Vega, Theta, Rho  
- **Plotly** chart: Call & Put price vs Spot (dual y-axes)  

### Researcher mode
- Controls for S/σ ranges and resolution  
- **Price Heatmap** (call/put) with in-cell labels and buyer/seller perspective coloring  
- **Greeks Heatmap** (delta/gamma/vega/theta/rho) for call/put  
- **Implied Volatility Smiles / Term Structure**  
  - **Synthetic** parametric smiles with skew & curvature  
  - **CSV upload** → invert to IV with `implied_vol_from_price` and plot calls/puts  

---

## 🗂️ Repository Structure

```
.
├── app.py                       # Streamlit app (main file)
├── core/
│   ├── bsm.py                   # bsm_price, bsm_greeks
│   └── iv.py                    # implied_vol_from_price
└── README.md
```

---

## 🔧 Core Functions

These live in `core/` and are imported by the app:

- **`bsm_price(S, K, r, q, sigma, T, option_type)`**  
  Returns the European option price under Black–Scholes–Merton.  
  `option_type`: `"call"` or `"put"`.  

- **`bsm_greeks(S, K, r, q, sigma, T, option_type)`**  
  Returns a dict with keys: `delta`, `gamma`, `vega`, `theta`, `rho`.  

- **`implied_vol_from_price(price, S, K, r, q, T, option_type)`**  
  Numerically inverts BSM to recover implied volatility from an option price.  

---

## 📈 CSV Upload Format (IV Smile Inversion)

When selecting **Upload CSV** in the “Implied Volatility Smile” tab, provide a file with:

- **Columns**: `K`, `price`, `option_type`  
  - `K`: float (strike)  
  - `price`: float (option price)  
  - `option_type`: `"call"` or `"put"` (case-insensitive)  

Example:

```csv
K,price,option_type
80,22.15,call
90,14.72,call
100,9.12,call
110,5.51,call
120,3.24,call
80,1.80,put
90,3.05,put
100,5.32,put
110,9.86,put
120,16.90,put
```

The app computes IV per row and overlays separate curves for calls and puts.

---

## 🧭 Usage Notes

- **Quick Scenarios** set temporary state flags; the app clears them automatically after plotting.  
- **Ranges & resolution** (Researcher mode) control computational grids for speed vs detail.  
- **Synthetic smiles**:  

Forward  

$$
F = S e^{(r - q)T}
$$

Log-moneyness  

$$
x = \log \frac{K}{F}
$$

Parametric IV  

$$
\sigma(x) = \sigma_{\text{ATM}} + \text{skew} \cdot x + \text{curvature} \cdot x^2
$$

clipped to  

$$
[10^{-4}, 5.0]
$$

