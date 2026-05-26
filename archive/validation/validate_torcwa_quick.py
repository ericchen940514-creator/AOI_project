"""
Quick torcwa smoke test — NG=11, NX=25, 20 wavelengths (~5s)
Confirms torcwa works on this machine before running full validation.
"""
import io, os, sys, time
import numpy as np
import yaml
import torch
import torcwa
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W, H, P = 0.200, 0.200, 0.500
WL_MIN, WL_MAX, NUM_WL = 0.4, 0.7, 20
NG = 11
NX = 25

wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WL)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_dtype = torch.complex128
geo_dtype = torch.float64

print(f"Device:          {device}")
print(f"torcwa version:  {torcwa.__version__}")
print(f"NG={NG}  NX={NX}  NUM_WL={NUM_WL}")

# ── DB path ───────────────────────────────────────────────────────────────────
_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "database", "data"),
]
DB = next(p for p in _DB_CANDIDATES if os.path.isdir(p))
print(f"DB:              {DB}\n")

with open(f"{DB}/main/Si/nk/Aspnes.yml", encoding="utf-8") as f:
    data = yaml.safe_load(f)
for blk in data["DATA"]:
    if blk["type"].strip() == "tabulated nk":
        rows = np.array([[float(v) for v in ln.split()]
                         for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
        wl_db, n_db, k_db = rows[:, 0], rows[:, 1], rows[:, 2]

n_fn   = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
k_fn   = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
eps_si = (n_fn(wavelengths) + 1j * k_fn(wavelengths)) ** 2


def run_torcwa(w, h, p):
    results = []
    for i, wl in enumerate(wavelengths):
        freq = 1.0 / wl
        ep   = complex(eps_si[i])

        sim = torcwa.rcwa(
            freq=freq,
            order=[1, NG],
            L=[p, p],
            dtype=sim_dtype,
            device=device,
        )
        sim.add_input_layer(eps=torch.tensor(1.0, dtype=sim_dtype, device=device))
        sim.add_output_layer(eps=torch.tensor(ep, dtype=sim_dtype, device=device))
        # FIX: 1e-6 instead of 0.0 to avoid singular matrix at exact normal incidence
        sim.set_incident_angle(inc_ang=1e-6, azi_ang=0.0)

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
        results.append(float(abs(R_s) ** 2))
    return np.array(results)


def run_grcwa(w, h, p):
    import grcwa
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)

    L1, L2 = [p, 0], [0, p]
    results = []
    for i, wl in enumerate(wavelengths):
        freq       = 1.0 / wl
        ep_grating = eps_si[i]

        # FIX: theta=1e-6 instead of 0 to avoid singular matrix
        obj = grcwa.obj(NG, L1, L2, freq, theta=1e-6, phi=0, verbose=0)
        obj.Add_LayerUniform(0, 1.0)
        obj.Add_LayerGrid(h, NX, NX)
        obj.Add_LayerUniform(0, ep_grating)
        obj.Init_Setup()

        ep_grid = np.ones((NX, NX), dtype=complex)
        si_cols = int(NX * (w / p))
        ep_grid[:, :si_cols] = ep_grating
        obj.GridLayer_geteps(ep_grid.flatten())

        obj.MakeExcitationPlanewave(p_amp=0, p_phase=0, s_amp=1, s_phase=0, order=0)
        R, _ = obj.RT_Solve(normalize=1)
        results.append(R)
    return np.array(results)


print(f"Geometry: w={W*1e3:.0f} h={H*1e3:.0f} p={P*1e3:.0f} nm")

t0 = time.perf_counter()
R_torcwa = run_torcwa(W, H, P)
dt_torcwa = time.perf_counter() - t0
print(f"torcwa:  {dt_torcwa:.2f}s  avg_R={R_torcwa.mean()*100:.2f}%")

t0 = time.perf_counter()
R_grcwa = run_grcwa(W, H, P)
dt_grcwa = time.perf_counter() - t0
print(f"grcwa:   {dt_grcwa:.2f}s  avg_R={R_grcwa.mean()*100:.2f}%")

mae = np.abs(R_torcwa - R_grcwa).mean() * 100
print(f"\nMAE torcwa vs grcwa: {mae:.4f}%")
print(f"Speedup: {dt_grcwa/dt_torcwa:.1f}x")
print("\n✓ PASS" if mae < 1.0 else "\n✗ FAIL — check NG or API mapping")
