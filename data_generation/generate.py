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
import io
import multiprocessing as mp
import os

import grcwa
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────────────────────
GRATING_MATERIAL = "Si"        # ← change material here

PARAMS_CSV  = "data_generation/output/rcwa_params_train.csv"
OUTPUT_DIR  = "data_generation/output"

WL_MIN, WL_MAX = 0.4, 0.7
NUM_WL         = 100
EPSILON_AIR    = 1.0
NX, NY         = 50, 50        # grid resolution inside grating layer
NG             = 51            # number of Fourier orders (higher = more accurate, slower)

wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WL)

MATERIALS = {
    "Si":    ("main", "Si",    "nk/Aspnes"),
    "SiO2":  ("main", "SiO2",  "nk/Franta"),
    "Si3N4": ("main", "Si3N4", "nk/Beliaev"),
    "TiO2":  ("main", "TiO2",  "nk/Franta"),
    "GaN":   ("main", "GaN",   "nk/Kawashima"),
    "Ge":    ("main", "Ge",    "nk/Aspnes"),
}


def _precompute_epsilon(material: str) -> np.ndarray:
    """Compute complex epsilon at each wavelength in the main process (scipy safe here)."""
    if material not in MATERIALS:
        raise ValueError(f"Unknown material '{material}'. Available: {list(MATERIALS.keys())}")

    shelf, book, page = MATERIALS[material]
    db_root = os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data")
    path = os.path.join(db_root, shelf, book, page + ".yml")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    wl_n, n_arr, k_arr = None, None, None
    for block in data["DATA"]:
        dtype = block["type"].strip()
        rows = np.array([
            [float(v) for v in line.split()]
            for line in io.StringIO(block["data"]).readlines() if line.strip()
        ])
        if dtype == "tabulated nk":
            wl_n, n_arr, k_arr = rows[:, 0], rows[:, 1], rows[:, 2]
        elif dtype == "tabulated n":
            wl_n, n_arr = rows[:, 0], rows[:, 1]
        elif dtype == "tabulated k":
            k_arr = rows[:, 1]

    n_fn = interp1d(wl_n, n_arr, kind="cubic", fill_value="extrapolate")
    k_fn = (interp1d(wl_n, k_arr, kind="cubic", fill_value="extrapolate")
            if k_arr is not None else lambda wl: np.zeros_like(wl))

    n_vals = n_fn(wavelengths)
    k_vals = k_fn(wavelengths)
    return (n_vals + 1j * k_vals) ** 2   # shape (NUM_WL,)


# Module-level array filled in run() before spawning workers.
# Workers inherit it via fork/spawn copy — no scipy needed in subprocesses.
_EP_GRATING: np.ndarray | None = None


def _worker(args: tuple) -> list:
    idx, w, h, p = args
    grcwa.set_backend("numpy")
    L1, L2 = [p, 0], [0, p]
    reflectances = []

    for i, wl in enumerate(wavelengths):
        freq       = 1.0 / wl
        ep_grating = _EP_GRATING[i]

        obj = grcwa.obj(NG, L1, L2, freq, theta=0, phi=0, verbose=0)
        obj.Add_LayerUniform(0, EPSILON_AIR)
        obj.Add_LayerGrid(h, NX, NY)
        obj.Add_LayerUniform(0, ep_grating)
        obj.Init_Setup()

        ep_grid = np.ones((NX, NY), dtype=complex) * EPSILON_AIR
        si_cols = int(NX * (w / p))
        ep_grid[:, :si_cols] = ep_grating
        obj.GridLayer_geteps(ep_grid.flatten())

        obj.MakeExcitationPlanewave(p_amp=0, p_phase=0, s_amp=1, s_phase=0, order=0)
        R, _ = obj.RT_Solve(normalize=1)
        reflectances.append(R)

    return [idx, w, h, p] + reflectances


def _init_worker(ep_array: np.ndarray) -> None:
    """Initializer: inject precomputed epsilon into each worker process."""
    global _EP_GRATING
    _EP_GRATING = ep_array


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

    ep_array = _precompute_epsilon(material)

    tasks = [(row.Index, row.w, row.h, row.p) for row in df.itertuples()]
    cores = max(1, min(mp.cpu_count() - 1, 8))  # NG=51 is memory-heavy; cap at 8
    print(f"Material : {material}")
    print(f"Rows     : {len(tasks):,}")
    print(f"Cores    : {cores}")

    results = []
    with mp.Pool(processes=cores, initializer=_init_worker, initargs=(ep_array,)) as pool:
        for res in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc="RCWA"):
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
