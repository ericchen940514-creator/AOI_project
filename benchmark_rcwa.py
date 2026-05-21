"""
Benchmark grcwa vs torcwa: accuracy vs tidy3d FDTD, and speed for 20k-scale dataset.
W=150nm, H=300nm, P=400nm, 100 wavelengths, 400-700nm.
"""

import io
import time
import warnings
import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d

W, H, P = 0.150, 0.300, 0.400
WL_MIN, WL_MAX, NUM_WL = 0.4, 0.7, 100
wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WL)
NX, NY = 50, 50

TIDY3D_CSV = "tidy3d/output/tidy3d_w150_h300_p400.csv"
DB = "C:/Users/user/.refractiveindex.info-database/data"


def load_si_nk():
    with open(f"{DB}/main/Si/nk/Aspnes.yml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        if blk["type"].strip() == "tabulated nk":
            rows = np.array([[float(v) for v in ln.split()]
                             for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
            wl, n, k = rows[:, 0], rows[:, 1], rows[:, 2]
            n_fn = interp1d(wl, n, kind="cubic", fill_value="extrapolate")
            k_fn = interp1d(wl, k, kind="cubic", fill_value="extrapolate")
            return n_fn, k_fn
    raise RuntimeError("Aspnes nk not found")

n_fn, k_fn = load_si_nk()
eps_si = (n_fn(wavelengths) + 1j * k_fn(wavelengths)) ** 2


# ── grcwa ─────────────────────────────────────────────────────────────────────
def run_grcwa(w, h, p, ng):
    import grcwa
    grcwa.set_backend("numpy")
    L1, L2 = [p, 0], [0, p]
    si_cols = int(NX * (w / p))
    ep_grid = np.ones((NX, NY), dtype=complex)
    ep_grid[:, :si_cols] = 1.0  # will be overwritten per wavelength

    results = []
    for i, wl in enumerate(wavelengths):
        freq = 1.0 / wl
        ep = eps_si[i]

        obj = grcwa.obj(ng, L1, L2, freq, theta=0, phi=0, verbose=0)
        obj.Add_LayerUniform(0, 1.0)
        obj.Add_LayerGrid(h, NX, NY)
        obj.Add_LayerUniform(0, ep)
        obj.Init_Setup()

        ep_grid[:] = 1.0
        ep_grid[:, :si_cols] = ep
        obj.GridLayer_geteps(ep_grid.flatten())

        obj.MakeExcitationPlanewave(p_amp=0, p_phase=0, s_amp=1, s_phase=0, order=0)
        R, _ = obj.RT_Solve(normalize=1)
        results.append(float(R))
    return np.array(results)


# ── torcwa ────────────────────────────────────────────────────────────────────
def run_torcwa(w, h, p, ng):
    import torch
    import torcwa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    si_cols = int(NX * (w / p))
    results = []

    for i, wl in enumerate(wavelengths):
        ep = complex(eps_si[i])

        geo = torch.ones((NX, NY), dtype=torch.complex128, device=device)
        geo[:, :si_cols] = ep

        sim = torcwa.rcwa(
            freq=1.0 / wl,
            order=[ng, ng],
            L=[p, p],
            dtype=torch.complex128,
            device=device,
        )
        sim.add_input_layer(eps=1.0)
        sim.add_layer(thickness=h, eps=geo)
        sim.add_output_layer(eps=ep)
        sim.solve_global_smatrix()

        # s-polarization: amplitude=[Ex, Ey]=[0,1] in xy notation
        sim.source_planewave(amplitude=[0., 1.], direction='forward', notation='xy')

        # Reflectance = sum of power in all reflected orders (ss polarization)
        all_orders = [[i, j] for i in range(-ng, ng+1) for j in range(-ng, ng+1)]
        R_ss = sim.S_parameters(orders=all_orders, direction='forward',
                                 port='reflection', polarization='yy',
                                 power_norm=True)
        R_total = float(R_ss.abs().sum().real.cpu())
        results.append(min(R_total, 1.0))
    return np.array(results)


# ── Load tidy3d reference ──────────────────────────────────────────────────────
df_t = pd.read_csv(TIDY3D_CSV)
wl_t = df_t["wavelength_nm"].values / 1000.0
R_ref = interp1d(wl_t, df_t["reflectance_fdtd"].values, kind="cubic",
                 fill_value="extrapolate")(wavelengths)


# ── Run benchmarks ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"W={W*1e3:.0f}nm  H={H*1e3:.0f}nm  P={P*1e3:.0f}nm  100 wavelengths")
print("=" * 60)

for ng in [21, 51, 101]:
    t0 = time.perf_counter()
    R_g = run_grcwa(W, H, P, ng)
    dt_g = time.perf_counter() - t0
    mae_g = np.abs(R_g - R_ref).mean() * 100
    print(f"grcwa  NG={ng:3d}: MAE={mae_g:.2f}%  time={dt_g:.1f}s")

print()

for ng in [5, 10, 15]:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        R_t = run_torcwa(W, H, P, ng)
    dt_t = time.perf_counter() - t0
    mae_t = np.abs(R_t - R_ref).mean() * 100
    print(f"torcwa NG={ng:3d}: MAE={mae_t:.2f}%  time={dt_t:.1f}s")

print()

# Speed extrapolation for 20k dataset
print("── 20k dataset extrapolation (single point * 20000) ──")
t0 = time.perf_counter()
run_grcwa(W, H, P, 51)
dt_one_g = time.perf_counter() - t0
print(f"grcwa  NG=51  per sample: {dt_one_g:.2f}s → 20k = {dt_one_g*20000/3600:.1f} hr (1 core)")

t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    run_torcwa(W, H, P, 10)
dt_one_t = time.perf_counter() - t0
print(f"torcwa NG=10  per sample: {dt_one_t:.2f}s → 20k = {dt_one_t*20000/3600:.1f} hr (1 core)")
