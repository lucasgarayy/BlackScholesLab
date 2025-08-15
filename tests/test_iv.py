# tests/test_iv.py

import numpy as np
from core.bsm import bsm_price
from core.iv import implied_vol

def test_implied_vol():
    S, K, r, q, sigma, T = 100, 100, 0.05, 0.0, 0.2, 1.0
    option_type = "call"

    # Get a price from known sigma
    price = bsm_price(S, K, r, q, sigma, T, option_type)

    # Feed price back into solver
    iv_est = implied_vol(price, S, K, r, q, T, option_type)

    assert np.isclose(iv_est, sigma, atol=1e-6)

if __name__ == "__main__":
    test_implied_vol()
    print("IV test passed.")
