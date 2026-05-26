"""
High-precision torcwa simulation for Mattila 2024 Sample A and B.

This runs only the two measured SiO2 grating samples, so it uses a higher
RCWA order than the training-data generator.

Usage:
    python simulate_mattila_torcwa_high_precision.py
    python simulate_mattila_torcwa_high_precision.py --ng 51
"""
import argparse
import io
import os
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torcwa
import yaml
from scipy.interpolate import interp1d


BASE = os.path.dirname(os.path.abspath(__file__))
VALID_DIR = os.path.join(BASE, "validation")
H5_A = os.path.join(VALID_DIR, "v1_A_256.h5")
H5_B = os.path.join(VALID_DIR, "v1_B_256.h5")

DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
    r"C:\Users\user\.refractiveindex.info-database\data",
]


def find_database():
    for path in DB_CANDIDATES:
        if os.path.isdir(path):
            return path
    raise RuntimeError("refractiveindex.info database not found.")


def load_eps_sio2(db_root, wavelengths_nm):
    path = os.path.join(db_root, "main", "SiO2", "nk", "Franta.yml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    wl_db = n_db = k_db = None
    for block in data["DATA"]:
        rows = np.array([
            [float(v) for v in line.split()]
            for line in io.StringIO(block["data"]).readlines()
            if line.strip()
        ])
        dtype = block["type"].strip()
        if dtype == "tabulated nk":
            wl_db, n_db, k_db = rows[:, 0] * 1000.0, rows[:, 1], rows[:, 2]
        elif dtype == "tabulated n":
            wl_db, n_db = rows[:, 0] * 1000.0, rows[:, 1]
            k_db = np.zeros_like(n_db)

    if wl_db is None:
        raise RuntimeError(f"No tabulated SiO2 data found in {path}")

    n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
    return (n_fn(wavelengths_nm) + 1j * k_fn(wavelengths_nm)) ** 2


def load_samples():
    samples = []
    if os.path.exists(H5_A):
        with h5py.File(H5_A, "r") as f:
            wl_nm = f["lambda"][:].flatten()
            params = f["params"]["d680_el300"][0]
            eta_r = f["eta_r"]["d680_el300"][0]
        period, height, groove = params[0], params[1], params[2]
        samples.append({
            "label": "Sample A",
            "key": "d680_el300",
            "period": period,
            "pillar": period - groove,
            "height": height,
            "wl_nm": wl_nm,
            "measured": eta_r,
        })

    if os.path.exists(H5_B):
        with h5py.File(H5_B, "r") as f:
            wl_nm = f["lambda"][:].flatten()
            best_key, best_err = None, float("inf")
            for key in f["eta_r"].keys():
                params = f["params"][key][0]
                period, height, groove = params[0], params[1], params[2]
                pillar = period - groove
                err = abs(period - 675.0) + abs(pillar - 351.0) + abs(height - 283.0)
                if err < best_err:
                    best_key, best_err = key, err

            params = f["params"][best_key][0]
            eta_r = f["eta_r"][best_key][0]
        period, height, groove = params[0], params[1], params[2]
        samples.append({
            "label": "Sample B",
            "key": best_key,
            "period": period,
            "pillar": period - groove,
            "height": height,
            "wl_nm": wl_nm,
            "measured": eta_r,
        })

    if not samples:
        raise RuntimeError("No validation H5 files found.")
    return samples


def simulate_torcwa(sample, eps_sio2, ng, nx, device):
    dtype = torch.complex128
    p = sample["period"] / 1000.0
    w = sample["pillar"] / 1000.0
    h = sample["height"] / 1000.0
    wl_um = sample["wl_nm"] / 1000.0
    sio2_cols = max(1, min(int(nx * w / p), nx - 1))

    results = []
    for i, wl in enumerate(wl_um):
        ep = complex(eps_sio2[i])
        try:
            sim = torcwa.rcwa(
                freq=1.0 / wl,
                order=[1, ng],
                L=[p, p],
                dtype=dtype,
                device=device,
            )
            sim.add_input_layer(eps=torch.tensor(1.0, dtype=dtype, device=device))
            sim.add_output_layer(eps=torch.tensor(ep, dtype=dtype, device=device))
            sim.set_incident_angle(inc_ang=1e-6, azi_ang=0.0)

            eps_grid = torch.ones(nx, nx, dtype=dtype, device=device)
            eps_grid[:, :sio2_cols] = ep
            sim.add_layer(thickness=h, eps=eps_grid)
            sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
            sim.solve_global_smatrix()

            r0 = sim.S_parameters(
                orders=[0, 0],
                direction="forward",
                port="reflection",
                polarization="xx",
                ref_order=[0, 0],
            )
            results.append(float(abs(r0) ** 2))
        except Exception:
            results.append(float("nan"))
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    arr = np.array(results, dtype=np.float64)
    nans = np.isnan(arr)
    if nans.any():
        x = np.arange(arr.size)
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ng", type=int, default=51, help="RCWA Fourier order in grating direction")
    parser.add_argument("--nx", type=int, default=None, help="Real-space grid size; default is 2*ng+3")
    args = parser.parse_args()

    nx = args.nx or (2 * args.ng + 3)
    if nx < 2 * args.ng + 1:
        raise ValueError("nx must be at least 2*ng+1 for torcwa FFT indexing.")

    db_root = find_database()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_samples()

    print(f"Device : {device}")
    print(f"torcwa : {torcwa.__version__}")
    print(f"NG={args.ng}  NX={nx}  dtype=complex128")
    print(f"DB     : {db_root}")

    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for ax_pair, sample in zip(axes, samples):
        eps_sio2 = load_eps_sio2(db_root, sample["wl_nm"])
        print(
            f"\n{sample['label']} [{sample['key']}]: "
            f"p={sample['period']:.1f} nm, "
            f"ridge={sample['pillar']:.1f} nm, "
            f"h={sample['height']:.1f} nm"
        )
        t0 = time.perf_counter()
        simulated = simulate_torcwa(sample, eps_sio2, args.ng, nx, device)
        dt = time.perf_counter() - t0
        diff = simulated - sample["measured"]
        mae = np.mean(np.abs(diff)) * 100.0
        rmse = np.sqrt(np.mean(diff ** 2)) * 100.0
        print(f"  MAE={mae:.4f}%  RMSE={rmse:.4f}%  time={dt:.1f}s")

        ax0, ax1 = ax_pair
        ax0.plot(sample["wl_nm"], sample["measured"] * 100, "k-", lw=2, label="Measured")
        ax0.plot(sample["wl_nm"], simulated * 100, "C0--", lw=1.8, label=f"torcwa NG={args.ng}")
        ax0.set_title(
            f"{sample['label']} [{sample['key']}]\n"
            f"p={sample['period']:.0f}, ridge={sample['pillar']:.1f}, h={sample['height']:.1f} nm"
        )
        ax0.set_xlabel("Wavelength (nm)")
        ax0.set_ylabel("Reflectance (%)")
        ax0.grid(True, ls="--", alpha=0.35)
        ax0.legend()

        ax1.plot(sample["wl_nm"], diff * 100, "C3-", lw=1.4)
        ax1.axhline(0, color="k", lw=0.8)
        ax1.set_title(f"Error: MAE={mae:.4f}%, RMSE={rmse:.4f}%")
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("torcwa - measured (%)")
        ax1.grid(True, ls="--", alpha=0.35)

    plt.tight_layout()
    out = os.path.join(BASE, f"mattila_torcwa_high_precision_ng{args.ng}.png")
    fig.savefig(out, dpi=160)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
