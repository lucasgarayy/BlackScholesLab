# core/iv.py
import numpy as np
from scipy.optimize import brentq
from core.bsm import bsm_price

EPS = 1e-12  # numerical slack

def _price_bounds(S, K, r, q, T, option_type):
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if option_type == "call":
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:  # put
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    return lower, upper

def _bsm_vega(S, K, r, q, sigma, T):
    """Analytic Vega under BSM (per 1.0 change in sigma)."""
    if sigma <= 0.0 or T <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    nprime = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    return np.exp(-q * T) * S * nprime * np.sqrt(T)

def _initial_guess(price, S, K, r, q, T, typ):
    """
    Forward-scaled Brenner–Subrahmanyam seed with sensible clipping.
    Works well near-ATM and behaves in wings.
    """
    if T <= 0.0:
        return 0.0
    DF = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    # Use call on forward (undiscounted)
    if typ == "put":
        c_undisc = price / DF + (F - K)  # from put-call parity
    else:
        c_undisc = price / DF
    intrinsic = max(F - K, 0.0)
    extr = max(c_undisc - intrinsic, 0.0)
    phi = extr / max(max(F, K), EPS)     # dimensionless extrinsic
    guess = np.sqrt(2.0 * np.pi) * phi / np.sqrt(T)

    m = np.log(max(F, EPS) / max(K, EPS))
    lo = 1e-4
    hi = 3.0 if abs(m) > 0.5 or phi < 1e-4 else 1.5
    return float(np.clip(guess, lo, hi))

def implied_vol(price, S, K, r, q, T, option_type="call", tol=1e-6, max_iter=100):
    """
    Fast, robust implied volatility:
      - Validates no-arbitrage bounds under BSM with dividends (q)
      - Near lower bound -> returns 0.0
      - Uses safeguarded Newton–Raphson with a dynamic sigma bracket
      - Falls back to Brent if Newton fails
      - Returns np.nan if price is out of bounds or no solution found

    Signature preserved for compatibility.
    """
    # Broadcast all inputs to the shape of price
    price = np.asarray(price)
    S, K, r, q, T, price = np.broadcast_arrays(S, K, r, q, T, price)

    # option_type can be scalar or array-like
    if np.isscalar(option_type):
        option_type = np.full(price.shape, option_type, dtype=object)
    else:
        option_type = np.asarray(option_type)
        if option_type.shape != price.shape:
            option_type = np.broadcast_to(option_type, price.shape)

    # Map max_iter to reasonable Newton/Brent budgets
    max_newton = max(4, min(12, max_iter // 2))   # quick convergence expected
    max_brent  = max_iter                         # keep your previous knob

    iv = np.full(price.shape, np.nan, dtype=float)

    for idx, p in np.ndenumerate(price):
        S_i, K_i, r_i, q_i, T_i = float(S[idx]), float(K[idx]), float(r[idx]), float(q[idx]), float(T[idx])
        typ = str(option_type[idx]).lower()

        if typ not in ("call", "put") or S_i <= 0.0 or K_i <= 0.0 or T_i <= 0.0:
            continue

        # No-arbitrage bounds (model-consistent)
        lower, upper = _price_bounds(S_i, K_i, r_i, q_i, T_i, typ)
        if p < lower - 1e-10 or p > upper + 1e-10:
            # Price inconsistent with BSM -> no implied vol
            continue
        if p <= lower + 1e-10:
            iv[idx] = 0.0
            continue
        # If you prefer np.inf at the upper bound, change the next line to: iv[idx] = np.inf
        if p >= upper - 1e-10:
            iv[idx] = np.nan
            continue

        # Define pricing error
        def f(sig):
            return bsm_price(S_i, K_i, r_i, q_i, sig, T_i, typ) - p

        # Build a dynamic bracket [a, b] containing a sign change for Brent and to safeguard Newton
        a, b = 1e-6, 1.5
        fa, fb = f(a), f(b)
        widen = 0
        while np.isfinite(fa) and np.isfinite(fb) and fa * fb > 0.0 and b < 10.0 and widen < 8:
            b *= 2.0
            fb = f(b)
            widen += 1
        if not (np.isfinite(fa) and np.isfinite(fb)):
            a, b = 1e-6, 10.0  # pathological: ensure a wide bracket

        # Newton–Raphson with safeguarding to the live bracket
        sigma = _initial_guess(p, S_i, K_i, r_i, q_i, T_i, typ)
        sigma = float(np.clip(sigma, a, b))
        success = False

        for _ in range(max_newton):
            diff = f(sigma)
            if abs(diff) < tol:
                success = True
                break

            vega = _bsm_vega(S_i, K_i, r_i, q_i, sigma, T_i)
            if vega <= 0.0 or not np.isfinite(vega):
                break  # unsafe for Newton

            step = diff / vega
            sigma_new = sigma - step

            # Safeguard: keep inside (a, b); if we step out, bisect
            if not np.isfinite(sigma_new) or sigma_new <= a or sigma_new >= b:
                # Bisect toward a sign flip
                if diff > 0.0:
                    a = max(a, 0.5 * (a + sigma))
                else:
                    b = min(b, 0.5 * (b + sigma))
                sigma_new = 0.5 * (a + b)
            else:
                # Maintain bracket using the sign of f(sigma_new)
                if f(sigma_new) > 0.0:
                    a = sigma_new
                else:
                    b = sigma_new

            if abs(sigma_new - sigma) < 1e-12:
                sigma = sigma_new
                success = True
                break

            sigma = sigma_new

        if success and np.isfinite(sigma):
            iv[idx] = max(sigma, 0.0)
            continue

        # Brent fallback (robust)
        try:
            iv[idx] = brentq(f, max(a, 1e-12), min(b, 10.0), xtol=tol, maxiter=max_brent)
        except Exception:
            iv[idx] = np.nan

    return iv
