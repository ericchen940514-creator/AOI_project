"""
Validation against Mattila et al., Meas. Sci. Technol. 35, 085025 (2024).
"Artificial neural network assisted spectral scatterometry for grating quality control"

Structure : SiO2 1D grating on SiO2 substrate
Measurement: spectral reflectance scatterometry, TE (s-pol), normal incidence, 400-750nm
Data       : openly available at https://doi.org/10.23729/7a9bbd75-7b8a-43d5-8c55-489f96e718b7

NOTE on conventions:
  h5 params = [period, height, groove_width, swa_L, swa_R, chm_L, chm_R]
  paper "line width" = period - groove_width  (pillar/ridge width)

Figure 6 samples validated here:
  Sample A: d680_el300 → period=680, pillar=309.1nm, height=209.7nm
  Sample B: d675_el310 in v1_B_256.h5 → period=675, pillar=351nm, height=283nm (if available)

Run on 4070 Super:
    python validate_mattila2024.py
"""
import io, os, sys, time
import numpy as np
import yaml
import torch
import torcwa
import h5py
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────────────────────
NG = 31   # SiO2 low contrast: NG=31 converges to within 0.008% of NG=101
NX = 65   # must be ≥ 2*NG+1 = 63 for torcwa FFT indexing

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_dtype = torch.complex128

print(f"Device : {device}")
print(f"torcwa : {torcwa.__version__}")

# ── Load SiO2 n,k ─────────────────────────────────────────────────────────────
_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)
if DB is None:
    raise RuntimeError("refractiveindex database not found.")

def load_eps(material_path, wavelengths_nm):
    with open(os.path.join(DB, material_path), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        dtype = blk["type"].strip()
        rows = np.array([[float(v) for v in ln.split()]
                         for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
        if dtype == "tabulated n":
            wl_n, n_arr = rows[:, 0] * 1000, rows[:, 1]  # µm→nm
            k_arr = np.zeros_like(n_arr)
        elif dtype == "tabulated nk":
            wl_n, n_arr, k_arr = rows[:, 0] * 1000, rows[:, 1], rows[:, 2]
        elif dtype == "tabulated k":
            wl_k, k_arr_k = rows[:, 0] * 1000, rows[:, 1]

    n_fn = interp1d(wl_n, n_arr, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_n, k_arr, kind="cubic", fill_value="extrapolate")
    n_v = n_fn(wavelengths_nm)
    k_v = k_fn(wavelengths_nm)
    return (n_v + 1j * k_v) ** 2


# ── torcwa simulation: SiO2 1D grating ────────────────────────────────────────
def simulate(period_nm, pillar_nm, height_nm, wavelengths_nm, eps_sio2):
    """
    period_nm  : grating period (nm)
    pillar_nm  : pillar/ridge width (nm) — NOT groove width
    height_nm  : grating height (nm)
    """
    p = period_nm / 1000   # nm → µm
    w = pillar_nm / 1000
    h = height_nm / 1000
    wl_um = wavelengths_nm / 1000

    results = []
    for i, wl in enumerate(wl_um):
        ep = complex(eps_sio2[i])
        try:
            # 1D grating varies in y (ep_grid[:, :si_cols]) → need NG y-orders, only 1 x-order
            # order=[1, NG] is equivalent to [NG, NG] for this geometry but ~67x less memory
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

            # 1D SiO2 pillar grating — groove is air (eps=1), ridge is SiO2
            ep_grid = torch.ones(NX, NX, dtype=sim_dtype, device=device)
            sio2_cols = int(NX * (w / p))
            ep_grid[:, :sio2_cols] = ep  # SiO2 ridge occupies left fraction of unit cell

            sim.add_layer(thickness=h, eps=ep_grid)
            sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
            sim.solve_global_smatrix()

            R_s = sim.S_parameters(
                orders=[0, 0], direction="forward",
                port="reflection", polarization="xx", ref_order=[0, 0],
            )
            results.append(float(abs(R_s) ** 2))
        except Exception:
            # Rayleigh anomaly (Kz=0) at λ ≈ period/m: fill with NaN, interpolate after
            results.append(float("nan"))
        finally:
            try:
                del sim
            except NameError:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Interpolate NaN entries from neighbours
    arr = np.array(results)
    nans = np.isnan(arr)
    if nans.any():
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
        print(f"  (interpolated {nans.sum()} Rayleigh-anomaly wavelengths)")
    return arr


# ── Load measured data ─────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

samples = []

# Sample A: d680_el300 → pillar=680-370.9=309.1nm, h=209.7nm (Fig.6 Sample A)
h5_A = os.path.join(BASE, "v1_A_256.h5")
if os.path.exists(h5_A):
    with h5py.File(h5_A, "r") as f:
        wl_nm  = f["lambda"][:].flatten()
        params = f["params"]["d680_el300"][0]
        eta_r  = f["eta_r"]["d680_el300"][0]
        period, height, groove = params[0], params[1], params[2]
        pillar = period - groove
        samples.append({
            "label": f"Sample A (p={period:.0f}/ridge={pillar:.1f}/h={height:.1f}nm)",
            "period": period, "pillar": pillar, "height": height,
            "wl_nm": wl_nm, "eta_r_meas": eta_r,
        })
    print(f"Loaded Sample A: period={period:.0f}nm  pillar={pillar:.1f}nm  h={height:.1f}nm")
else:
    print("v1_A_256.h5 not found")

# Sample B: from v1_B_256.h5 (different wafer, higher heights ~280nm)
# Fallback: pick a contrasting sample from v1_A_256.h5 if v1_B not available
h5_B = os.path.join(BASE, "v1_B_256.h5")
if os.path.exists(h5_B):
    with h5py.File(h5_B, "r") as f:
        wl_nm_B = f["lambda"][:].flatten()
        print("\nv1_B_256.h5 keys:")
        for key in sorted(f["eta_r"].keys()):
            p = f["params"][key][0]
            pillar_b = p[0] - p[2]
            print(f"  {key}: period={p[0]:.0f}  h={p[1]:.1f}  pillar={pillar_b:.1f}")

        # Find closest to Sample B: p=675, pillar=351, h=283
        best_key = None
        best_err = 1e9
        for key in f["eta_r"].keys():
            p = f["params"][key][0]
            pillar_b = p[0] - p[2]
            err = abs(p[0] - 675) + abs(pillar_b - 351) + abs(p[1] - 283)
            if err < best_err:
                best_err, best_key = err, key

        if best_key:
            params_b = f["params"][best_key][0]
            eta_r_b  = f["eta_r"][best_key][0]
            period_b, height_b, groove_b = params_b[0], params_b[1], params_b[2]
            pillar_b = period_b - groove_b
            samples.append({
                "label": f"Sample B (p={period_b:.0f}/ridge={pillar_b:.1f}/h={height_b:.1f}nm)",
                "period": period_b, "pillar": pillar_b, "height": height_b,
                "wl_nm": wl_nm_B, "eta_r_meas": eta_r_b,
            })
            print(f"\nLoaded Sample B [{best_key}]: period={period_b:.0f}nm  pillar={pillar_b:.1f}nm  h={height_b:.1f}nm")
elif os.path.exists(h5_A):
    # Fallback: use a second contrasting sample from v1_A_256.h5
    FALLBACK_KEY = "d660_el330"  # period=660, pillar=340nm, h=225nm — maximally different from Sample A
    with h5py.File(h5_A, "r") as f:
        wl_nm_B = f["lambda"][:].flatten()
        params_b = f["params"][FALLBACK_KEY][0]
        eta_r_b  = f["eta_r"][FALLBACK_KEY][0]
    period_b, height_b, groove_b = params_b[0], params_b[1], params_b[2]
    pillar_b = period_b - groove_b
    samples.append({
        "label": f"Sample B* (p={period_b:.0f}/ridge={pillar_b:.1f}/h={height_b:.1f}nm) [fallback from A]",
        "period": period_b, "pillar": pillar_b, "height": height_b,
        "wl_nm": wl_nm_B, "eta_r_meas": eta_r_b,
    })
    print(f"v1_B_256.h5 not found — using fallback [{FALLBACK_KEY}]: "
          f"period={period_b:.0f}nm  pillar={pillar_b:.1f}nm  h={height_b:.1f}nm")
else:
    print("v1_B_256.h5 not found — only Sample A will be validated")

# ── Run simulations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
if len(samples) == 1:
    axes = [axes]

maes = []
for i, s in enumerate(samples):
    wl_nm  = s["wl_nm"]
    eps    = load_eps("main/SiO2/nk/Franta.yml", wl_nm)

    print(f"\nSimulating {s['label']} ...")
    t0 = time.perf_counter()
    R_sim = simulate(s["period"], s["pillar"], s["height"], wl_nm, eps)
    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.1f}s")

    eta_meas = s["eta_r_meas"]
    mae = np.abs(R_sim - eta_meas).mean() * 100
    maes.append(mae)
    print(f"  MAE vs measured: {mae:.4f}%")

    ax0, ax1 = axes[i]
    ax0.plot(wl_nm, eta_meas * 100, "C1-",  lw=2, label="Measured (Mattila 2024)")
    ax0.plot(wl_nm, R_sim    * 100, "C0--", lw=2, label=f"torcwa NG={NG}")
    ax0.set_xlabel("Wavelength (nm)")
    ax0.set_ylabel("η_R (%)")
    ax0.set_title(s["label"])
    ax0.legend()
    ax0.grid(True, ls="--", alpha=0.4)

    diff = (R_sim - eta_meas) * 100
    ax1.plot(wl_nm, diff, "C2-", lw=1.5)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.fill_between(wl_nm, diff, alpha=0.3, color="C2")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("torcwa − measured (%)")
    ax1.set_title(f"MAE = {mae:.4f}%")
    ax1.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
out = os.path.join(BASE, "validate_mattila2024.png")
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")
print("\n=== SUMMARY ===")
for s, mae in zip(samples, maes):
    status = "PASS" if mae < 2.0 else "CHECK"
    print(f"  {s['label']}: MAE={mae:.4f}%  [{status}]")
