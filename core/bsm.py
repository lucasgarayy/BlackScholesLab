# core/bsm.py

import numpy as np
from scipy.stats import norm

def _d1(S, K, r, q, sigma, T):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def _d2(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

def bsm_price(S, K, r, q, sigma, T, option_type="call"):
    S, K, r, q, sigma, T = map(np.asarray, (S, K, r, q, sigma, T))
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)

    if option_type.lower() == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def bsm_greeks(S, K, r, q, sigma, T, option_type="call"):
    S, K, r, q, sigma, T = map(np.asarray, (S, K, r, q, sigma, T))
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)
    pdf_d1 = norm.pdf(d1)

    if option_type.lower() == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
    elif option_type.lower() == "put":
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        theta = (-S * pdf_d1 * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega / 100,
        "rho": rho / 100
    }
