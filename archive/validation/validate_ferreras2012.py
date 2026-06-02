"""
Validation against Ferreras Paz et al., Light: Science & Applications 1, e36 (2012).
"Solving the inverse grating problem by white light interference Fourier scatterometry"

Structure : Si 1D grating on Si substrate, square lattice
Geometry  : period=400nm, CD(linewidth)=200nm, height=75nm
            fill fraction = 0.50
Incidence : normal (theta=0), s-polarization
Wavelength: 400-700nm (white light)

── How to get reference data ────────────────────────────────────────────────
1. Open paper: https://www.nature.com/articles/lsa201236
2. Find the measured reflectance vs wavelength figure
3. Open https://automeris.io/WebPlotDigitizer/
4. Digitize the measured curve
5. Export CSV → save as  ref_ferreras2012.csv
   Columns: wavelength_nm, reflectance   (reflectance 0–1, not percent)
─────────────────────────────────────────────────────────────────────────────

Run on 4070 Super:
    python validate_ferreras2012.py
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

# ── Geometry (Ferreras 2012) ──────────────────────────────────────────────────
P  = 0.400   # µm — period
CD = 0.200   # µm — linewidth (fill fraction = 0.50)
H  = 0.075   # µm — grating height

# ── Simulation settings ───────────────────────────────────────────────────────
WL_MIN, WL_MAX, NUM_WL = 0.4, 0.7, 100
NG = 79   # max for NX=160; NX must be ≥ 2*NG+1 for torcwa FFT
NX = 160

wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WL)
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_dtype   = torch.complex128

print(f"Device : {device}")
print(f"torcwa : {torcwa.__version__}")
print(f"Geometry: p={P*1e3:.0f}nm  CD={CD*1e3:.0f}nm  h={H*1e3:.0f}nm  ff={CD/P:.2f}")

# ── Load Si n,k ───────────────────────────────────────────────────────────────
_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)
if DB is None:
    raise RuntimeError("refractiveindex database not found.")

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

# ── torcwa simulation ─────────────────────────────────────────────────────────
def run_torcwa(p, cd, h):
    results = []
    for i, wl in enumerate(wavelengths):
        ep = complex(eps_si[i])

        # 1D grating varies in y → order=[1, NG] is equivalent to [NG, NG] but ~67x less memory
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

        # 1D grating: Si lines in x-direction, uniform in y
        ep_grid = torch.ones(NX, NX, dtype=sim_dtype, device=device)
        si_cols = int(NX * (cd / p))
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

# ── Reference data ────────────────────────────────────────────────────────────
REF_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ref_ferreras2012.csv")
if os.path.exists(REF_CSV):
    import pandas as pd
    df_ref  = pd.read_csv(REF_CSV)
    ref_fn  = interp1d(df_ref["wavelength_nm"] / 1000, df_ref["reflectance"],
                       kind="cubic", bounds_error=False, fill_value="extrapolate")
    R_ref   = ref_fn(wavelengths)
    REF_LABEL = "Ferreras 2012 (measured)"
    print(f"Loaded reference: {REF_CSV}")
else:
    R_ref     = None
    REF_LABEL = None
    print("No ref_ferreras2012.csv — run WebPlotDigitizer on paper figure first")

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\nRunning torcwa  {NUM_WL} wavelengths  NG={NG}  NX={NX}...")
t0 = time.perf_counter()
R_sim = run_torcwa(P, CD, H)
dt = time.perf_counter() - t0
print(f"Done in {dt:.1f}s  avg_R={R_sim.mean()*100:.2f}%")

if R_ref is not None:
    mae = np.abs(R_sim - R_ref).mean() * 100
    print(f"MAE vs experiment: {mae:.4f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
wl_nm = wavelengths * 1000
fig, axes = plt.subplots(1, 2 if R_ref is not None else 1, figsize=(12 if R_ref is not None else 6, 4))
ax = axes[0] if R_ref is not None else axes

ax.plot(wl_nm, R_sim * 100, "C0-", lw=2,
        label=f"torcwa  avg={R_sim.mean()*100:.1f}%")
if R_ref is not None:
    ax.plot(wl_nm, R_ref * 100, "C1-", lw=2, label=REF_LABEL)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance (%)")
ax.set_title(f"Ferreras 2012 — Si 1D grating  p={P*1e3:.0f}/CD={CD*1e3:.0f}/h={H*1e3:.0f} nm")
ax.legend()
ax.grid(True, ls="--", alpha=0.4)

if R_ref is not None:
    diff = (R_sim - R_ref) * 100
    axes[1].plot(wl_nm, diff, "C2-", lw=1.5)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].fill_between(wl_nm, diff, alpha=0.3, color="C2")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("torcwa − experiment (%)")
    axes[1].set_title(f"MAE = {mae:.3f}%")
    axes[1].grid(True, ls="--", alpha=0.4)

plt.tight_layout()
out = "validate_ferreras2012.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

if R_ref is not None:
    print("\n✓ PASS" if mae < 2.0 else "\n✗ CHECK — MAE > 2%, review NG or geometry")
