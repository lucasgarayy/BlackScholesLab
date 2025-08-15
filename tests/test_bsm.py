# tests/test_bsm.py

import numpy as np
from core.bsm import bsm_price, bsm_greeks

def test_bsm_prices_and_greeks():
    S = 100
    K = 100
    r = 0.05
    q = 0.0
    sigma = 0.2
    T = 1

    call_price = bsm_price(S, K, r, q, sigma, T, option_type="call")
    put_price = bsm_price(S, K, r, q, sigma, T, option_type="put")

    # Expected from textbook Black-Scholes
    assert np.isclose(call_price, 10.4506, atol=1e-4)
    assert np.isclose(put_price, 5.5735, atol=1e-4)

    call_greeks = bsm_greeks(S, K, r, q, sigma, T, option_type="call")
    put_greeks = bsm_greeks(S, K, r, q, sigma, T, option_type="put")

    # Check a couple of Greeks for sanity
    assert np.isclose(call_greeks["delta"], 0.6368, atol=1e-4)
    assert np.isclose(put_greeks["delta"], -0.3632, atol=1e-4)
    assert np.isclose(call_greeks["gamma"], 0.0188, atol=1e-4)
    assert np.isclose(call_greeks["vega"] * 100, 37.5240, atol=1e-4)  # vega per 1.0 = 37.52

if __name__ == "__main__":
    test_bsm_prices_and_greeks()
    print("All tests passed.")
