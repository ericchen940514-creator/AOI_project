"""
LHS parameter sampling — generates w/h/p combinations.

Run standalone:
    python data_generation/params.py

Output files go to data_generation/output/.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import os

OUTPUT_DIR = "data_generation/output"

BOUNDS = {"w": (0.05, 0.25), "h": (0.10, 0.50), "p": (0.20, 0.60)}
FILL_FACTOR_RANGE = (0.30, 0.70)


def generate_params(total: int, test_ratio: float = 500 / 20500, seed: int = 42):
    """
    Draw *total* valid (w, h, p) samples via LHS, filtered by fill-factor.
    Returns (df_train, df_test).
    """
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    raw = sampler.random(n=total * 3)   # oversample to survive FF filter

    l_bounds = [BOUNDS[k][0] for k in ("w", "h", "p")]
    u_bounds = [BOUNDS[k][1] for k in ("w", "h", "p")]
    scaled = qmc.scale(raw, l_bounds, u_bounds)

    df = pd.DataFrame(scaled, columns=["w", "h", "p"])
    ff = df["w"] / df["p"]
    df = df[(ff >= FILL_FACTOR_RANGE[0]) & (ff <= FILL_FACTOR_RANGE[1])].reset_index(drop=True)

    if len(df) < total:
        raise ValueError(f"Only {len(df)} valid samples after FF filter; need {total}. "
                         "Increase oversampling multiplier.")

    df = df.head(total)
    n_test = int(total * test_ratio)
    return df.iloc[n_test:].reset_index(drop=True), df.iloc[:n_test].reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_train, df_test = generate_params(20500)
    train_path = f"{OUTPUT_DIR}/rcwa_params_train.csv"
    test_path  = f"{OUTPUT_DIR}/rcwa_params_test.csv"
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path,   index=False)
    print(f"Train params: {len(df_train):,} rows -> {train_path}")
    print(f"Test  params: {len(df_test):,} rows  -> {test_path}")
