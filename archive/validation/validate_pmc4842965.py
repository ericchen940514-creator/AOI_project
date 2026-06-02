"""
Validation against PMC4842965 (Brissinger et al., Sci. Rep. 6:24947, 2016).
"Optimized 2D array of thin silicon pillars for efficient antireflective coatings"

Structure : Si 2D square pillar array on Si substrate, square lattice
Geometry  : w=140nm, h=110nm, p=240nm  (optimized design from paper)
Incidence : normal (theta=0), s-polarization
Expected  : avg R ~ 4% in 400-900nm (paper Fig 3c)

── How to get reference data ────────────────────────────────────────────────
1. Open paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC4842965/
2. Find Figure 3c  (measured reflectance vs wavelength, s-pol, normal incidence)
3. Open https://automeris.io/WebPlotDigitizer/ → digitize the measured curve
4. Export CSV → save as  ref_pmc4842965_fig3c.csv
   Columns: wavelength_nm, reflectance   (reflectance 0–1, NOT percent)
─────────────────────────────────────────────────────────────────────────────

Run on 4070 Super:
    python validate_pmc4842965.py
"""
import io, os, time
import numpy as np
import yaml
import torch
import torcwa
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Geometry (PMC4842965 optimized design) ────────────────────────────────────
W = 0.140   # µm — pillar width
H = 0.110   # µm — pillar height
P = 0.240   # µm — period

# ── Simulation settings ───────────────────────────────────────────────────────
WL_MIN, WL_MAX, NUM_WL = 0.4, 0.7, 100
NG = 101
NX = 160

wavelengths = np.linspace(WL_MIN, WL_MAX, NUM_WL)
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_dtype   = torch.complex128

print(f"Device : {device}")
print(f"torcwa : {torcwa.__version__}")
print(f"Geometry: w={W*1e3:.0f}nm  h={H*1e3:.0f}nm  p={P*1e3:.0f}nm")

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

# ── torcwa: 2D square pillar ──────────────────────────────────────────────────
def run_torcwa(w, h, p):
    results = []
    for i, wl in enumerate(wavelengths):
        ep = complex(eps_si[i])

        sim = torcwa.rcwa(
            freq=1.0 / wl,
            order=[NG, NG],
            L=[p, p],
            dtype=sim_dtype,
            device=device,
        )
        sim.add_input_layer(eps=torch.tensor(1.0, dtype=sim_dtype, device=device))
        sim.add_output_layer(eps=torch.tensor(ep, dtype=sim_dtype, device=device))
        sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)  # must be before add_layer

        # 2D square pillar centered in unit cell
        ep_grid = torch.ones(NX, NX, dtype=sim_dtype, device=device)
        half    = int(NX * (w / p) / 2)
        cx      = NX // 2
        ep_grid[cx - half : cx + half, cx - half : cx + half] = ep
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
                       "ref_pmc4842965_fig3c.csv")
if os.path.exists(REF_CSV):
    df_ref    = pd.read_csv(REF_CSV)
    ref_fn    = interp1d(df_ref["wavelength_nm"] / 1000, df_ref["reflectance"],
                         kind="cubic", bounds_error=False, fill_value="extrapolate")
    R_ref     = ref_fn(wavelengths)
    REF_LABEL = "Brissinger 2016 (measured)"
    print(f"Loaded reference: {REF_CSV}")
else:
    R_ref     = None
    REF_LABEL = None
    print("No ref_pmc4842965_fig3c.csv — digitize paper Fig 3c first")

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\nRunning torcwa  {NUM_WL} wavelengths  NG={NG}  NX={NX} ...")
t0 = time.perf_counter()
R_sim = run_torcwa(W, H, P)
dt = time.perf_counter() - t0
print(f"Done in {dt:.1f}s  avg_R={R_sim.mean()*100:.2f}%")

if R_ref is not None:
    mae = np.abs(R_sim - R_ref).mean() * 100
    print(f"MAE vs experiment : {mae:.4f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
wl_nm   = wavelengths * 1000
ncols   = 2 if R_ref is not None else 1
fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
ax0 = axes[0] if R_ref is not None else axes

ax0.plot(wl_nm, R_sim * 100, "C0-", lw=2,
         label=f"torcwa NG={NG}  avg={R_sim.mean()*100:.1f}%")
if R_ref is not None:
    ax0.plot(wl_nm, R_ref * 100, "C1-", lw=2, label=REF_LABEL)
ax0.set_xlabel("Wavelength (nm)")
ax0.set_ylabel("Reflectance (%)")
ax0.set_title(f"PMC4842965 — Si 2D pillar  w={W*1e3:.0f}/h={H*1e3:.0f}/p={P*1e3:.0f} nm")
ax0.legend()
ax0.grid(True, ls="--", alpha=0.4)

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
out = "validate_pmc4842965.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

if R_ref is not None:
    if mae < 2.0:
        print(f"\n✓ PASS  MAE={mae:.3f}% < 2% — torcwa validated against experiment")
    else:
        print(f"\n✗ CHECK  MAE={mae:.3f}% > 2% — review NG or geometry")
else:
    print(f"\ntorcwa avg R = {R_sim.mean()*100:.2f}%  (paper claims ~4%)")
    print("Add ref_pmc4842965_fig3c.csv to get quantitative MAE")
