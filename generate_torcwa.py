"""
Generate RCWA dataset using torcwa (GPU).
Run on 4070 Super machine AFTER validate_torcwa.py passes.

Usage:
    python generate_torcwa.py                    # 5000 rows
    python generate_torcwa.py --nrows 20000      # 20000 rows
    python generate_torcwa.py --nrows 100 --test # quick test
"""
import io, os, sys, time, argparse
import numpy as np
import yaml
import torch
import torcwa
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────────────────────
WL_MIN, WL_MAX, NUM_WL = 0.4, 0.7, 100
NG   = 101
NX   = 210  # must be ≥ 2*NG+1 = 203 for torcwa FFT indexing
CHUNK = 100   # rows per CSV write

wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WL)
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_dtype   = torch.complex128

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
PARAMS  = os.path.join(BASE, "data_generation/output/rcwa_params_train.csv")
OUTFILE = os.path.join(BASE, "data_generation/output/rcwa_Si_train_torcwa.csv")

_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)
if DB is None:
    raise RuntimeError("refractiveindex database not found. Copy it to ~/.refractiveindex.info-database/")


# ── Load Si epsilon ───────────────────────────────────────────────────────────
def load_eps_si():
    with open(f"{DB}/main/Si/nk/Aspnes.yml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        if blk["type"].strip() == "tabulated nk":
            rows = np.array([[float(v) for v in ln.split()]
                             for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
            wl_db, n_db, k_db = rows[:, 0], rows[:, 1], rows[:, 2]
    n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
    return (n_fn(wavelengths) + 1j * k_fn(wavelengths)) ** 2


# ── Single geometry simulation ─────────────────────────────────────────────────
def simulate_one(w, h, p, eps_si):
    reflectances = []
    for i, wl in enumerate(wavelengths):
        ep = complex(eps_si[i])

        # 1D grating varies in y (ep_grid[:, :si_cols]) → only 1 x-order needed
        sim = torcwa.rcwa(
            freq=1.0 / wl,
            order=[1, NG],
            L=[p, p],
            dtype=sim_dtype,
            device=device,
        )
        sim.add_input_layer(eps=torch.tensor(1.0, dtype=sim_dtype, device=device))
        sim.add_output_layer(eps=torch.tensor(ep, dtype=sim_dtype, device=device))
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)  # must be before add_layer

        ep_grid = torch.ones(NX, NX, dtype=sim_dtype, device=device)
        si_cols = int(NX * (w / p))
        ep_grid[:, :si_cols] = ep
        sim.add_layer(thickness=h, eps=ep_grid)

        sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
        sim.solve_global_smatrix()

        R_s = sim.S_parameters(
            orders=[0, 0],
            direction="forward",
            port="reflection",
            polarization="xx",
            ref_order=[0, 0],
        )
        reflectances.append(float(abs(R_s) ** 2))
    return reflectances


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows",  type=int, default=5000)
    parser.add_argument("--test",   action="store_true", help="Quick 100-row test")
    parser.add_argument("--out",    default=OUTFILE)
    args = parser.parse_args()

    nrows   = 100 if args.test else args.nrows
    outfile = args.out

    print(f"Device : {device}")
    print(f"NG={NG}  NX={NX}  rows={nrows}")

    eps_si = load_eps_si()
    cols   = ["w", "h", "p"] + [f"R_wl_{i}" for i in range(NUM_WL)]

    # Resume support
    if os.path.exists(outfile):
        done = sum(1 for _ in open(outfile)) - 1
        done = max(done, 0)
        print(f"Resuming from row {done}")
    else:
        done = 0
        pd.DataFrame(columns=cols).to_csv(outfile, index=False)

    df_params = pd.read_csv(PARAMS).head(nrows).iloc[done:]
    if df_params.empty:
        print("Already complete.")
        return

    buffer = []
    t0 = time.time()

    with tqdm(total=len(df_params), desc="torcwa") as pbar:
        for row in df_params.itertuples():
            R = simulate_one(row.w, row.h, row.p, eps_si)
            buffer.append([row.w, row.h, row.p] + R)
            pbar.update(1)

            if len(buffer) >= CHUNK:
                pd.DataFrame(buffer, columns=cols).to_csv(
                    outfile, mode="a", header=False, index=False)
                buffer.clear()

    if buffer:
        pd.DataFrame(buffer, columns=cols).to_csv(
            outfile, mode="a", header=False, index=False)

    dt = (time.time() - t0) / 60
    total = done + len(df_params)
    print(f"\nDone: {total} rows in {dt:.1f} min → {outfile}")
    rows_per_min = len(df_params) / max(dt, 0.01)
    print(f"Speed: {rows_per_min:.1f} rows/min")


if __name__ == "__main__":
    main()
