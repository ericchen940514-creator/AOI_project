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

# SiO2 grating bounds (nm)
BOUNDS = {"period": (600, 760), "pillar": (150, 500), "height": (100, 400)}
FILL_FACTOR_RANGE = (0.20, 0.75)  # pillar/period


def generate_params(total: int, seed: int = 42):
    """
    Draw *total* valid (period, pillar, height) samples via LHS, filtered by fill-factor.
    Returns DataFrame with columns: period_nm, pillar_nm, height_nm.
    """
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    raw = sampler.random(n=total * 3)

    l_bounds = [BOUNDS[k][0] for k in ("period", "pillar", "height")]
    u_bounds = [BOUNDS[k][1] for k in ("period", "pillar", "height")]
    scaled = qmc.scale(raw, l_bounds, u_bounds)

    df = pd.DataFrame(scaled, columns=["period_nm", "pillar_nm", "height_nm"])
    ff = df["pillar_nm"] / df["period_nm"]
    df = df[(ff >= FILL_FACTOR_RANGE[0]) & (ff <= FILL_FACTOR_RANGE[1])].reset_index(drop=True)

    if len(df) < total:
        raise ValueError(f"Only {len(df)} valid samples after FF filter; need {total}.")

    return df.head(total).reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=5000)
    parser.add_argument("--out",   default=f"{OUTPUT_DIR}/sio2_params.csv")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate_params(args.nrows)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df):,} params -> {args.out}")
    print(df.describe().to_string())
