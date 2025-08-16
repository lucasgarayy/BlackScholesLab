# core/iv.py
import numpy as np
from core.bsm import bsm_price

def implied_vol_from_price(price, S, K, r, q, T, option_type, tol=1e-8, max_iter=100, lo=1e-6, hi=5.0):
    # Bisection: find sigma such that BSM price matches observed price
    plo = bsm_price(S=S, K=K, r=r, q=q, sigma=lo, T=T, option_type=option_type)
    phi = bsm_price(S=S, K=K, r=r, q=q, sigma=hi, T=T, option_type=option_type)
    if not (plo <= price <= phi):
        return np.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pm = bsm_price(S=S, K=K, r=r, q=q, sigma=mid, T=T, option_type=option_type)
        if abs(pm - price) < tol:
            return mid
        if pm > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
