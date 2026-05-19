"""
RCWA broadband simulation — generates reflectance spectra for a grating.

Change the grating material by setting GRATING_MATERIAL to any key in
data_generation/materials.py::MATERIALS, e.g.:
    GRATING_MATERIAL = "Si"
    GRATING_MATERIAL = "TiO2"
    GRATING_MATERIAL = "GaN"

Run:
    python data_generation/generate.py                  # full run
    python data_generation/generate.py --test           # first 100 rows only
    python data_generation/generate.py --material TiO2  # override material
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import grcwa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────────────────────
GRATING_MATERIAL = "Si"        # ← change material here

PARAMS_CSV  = "data_generation/output/rcwa_params_train.csv"
OUTPUT_DIR  = "data_generation/output"

WL_MIN, WL_MAX = 0.4, 0.7
NUM_WL         = 100
EPSILON_AIR    = 1.0
NX, NY         = 50, 50        # grid resolution inside grating layer
NG             = 11            # number of Fourier orders

wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WL)

# ── Material epsilon (loaded once per worker via module-level cache) ───────────

def _get_epsilon(material: str, wl_um: float) -> complex:
    from data_generation.materials import get_epsilon
    return get_epsilon(material, wl_um)


# ── Worker function (must be top-level for multiprocessing) ──────────────────

def _worker(args: tuple) -> list:
    material, idx, w, h, p = args
    grcwa.set_backend("numpy")
    L1, L2 = [p, 0], [0, p]
    reflectances = []

    for wl in wavelengths:
        freq       = 1.0 / wl
        ep_grating = _get_epsilon(material, wl)

        obj = grcwa.obj(NG, L1, L2, freq, theta=0, phi=0, verbose=0)
        obj.Add_LayerUniform(0, EPSILON_AIR)
        obj.Add_LayerGrid(h, NX, NY)
        obj.Add_LayerUniform(0, ep_grating)   # substrate = same material
        obj.Init_Setup()

        ep_grid = np.ones((NX, NY), dtype=complex) * EPSILON_AIR
        si_cols = int(NX * (w / p))
        ep_grid[:, :si_cols] = ep_grating
        obj.GridLayer_geteps(ep_grid.flatten())

        obj.MakeExcitationPlanewave(p_amp=0, p_phase=0, s_amp=1, s_phase=0, order=0)
        R, _ = obj.RT_Solve(normalize=1)
        reflectances.append(R)

    return [idx, w, h, p] + reflectances


# ── Main ──────────────────────────────────────────────────────────────────────

def run(material: str, params_csv: str, test_mode: bool = False) -> None:
    if not os.path.exists(params_csv):
        raise FileNotFoundError(
            f"Parameter file not found: {params_csv}\n"
            "Run  python data_generation/params.py  first."
        )

    df = pd.read_csv(params_csv)
    if test_mode:
        df = df.head(100)
        print("[TEST MODE] using first 100 rows only")

    tasks = [(material, row.Index, row.w, row.h, row.p) for row in df.itertuples()]
    cores = max(1, mp.cpu_count() - 1)
    print(f"Material : {material}")
    print(f"Rows     : {len(tasks):,}")
    print(f"Cores    : {cores}")

    results = []
    with mp.Pool(processes=cores) as pool:
        for res in tqdm(pool.imap(_worker, tasks), total=len(tasks), desc="RCWA"):
            results.append(res)

    cols = ["id", "w", "h", "p"] + [f"R_wl_{i}" for i in range(NUM_WL)]
    df_out = pd.DataFrame(results, columns=cols).drop(columns=["id"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag    = "test" if test_mode else "train"
    outfile = f"{OUTPUT_DIR}/rcwa_{material}_{tag}.csv"
    df_out.to_csv(outfile, index=False)
    print(f"Saved {len(df_out):,} rows -> {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default=GRATING_MATERIAL,
                        help="Grating material key (see materials.py)")
    parser.add_argument("--test", action="store_true",
                        help="Run only first 100 rows")
    parser.add_argument("--params", default=PARAMS_CSV,
                        help="Path to params CSV")
    args = parser.parse_args()

    run(material=args.material, params_csv=args.params, test_mode=args.test)
