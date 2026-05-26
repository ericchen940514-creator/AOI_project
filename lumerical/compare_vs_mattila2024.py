"""
Compare Lumerical FDTD against Mattila et al. 2024 measured data AND torcwa.

Paper: Mattila et al., Meas. Sci. Technol. 35, 085025 (2024)
Structure: SiO2 1D grating on SiO2 substrate, TE/s-pol, normal incidence, 400-750nm

Measured data: v1_A_256.h5 / v1_B_256.h5  (in validation/)

Usage:
    python lumerical/compare_vs_mattila2024.py
"""
import sys, os, io, time
import numpy as np
import yaml
import h5py
import torch
import torcwa
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python"
sys.path.append(LUMAPI_PATH)
import lumapi

BASE       = os.path.dirname(os.path.abspath(__file__))
VALID_DIR  = os.path.join(BASE, "..", "validation")
H5_A       = os.path.join(VALID_DIR, "v1_A_256.h5")
H5_B       = os.path.join(VALID_DIR, "v1_B_256.h5")

_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)

# torcwa settings (matches validate_mattila2024.py)
NG_RCWA = 31
NX_RCWA = 65
sim_dtype = torch.complex128
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lumerical constants
Y_SPAN    = 20e-9
Z_BELOW   = 1000e-9
Z_ABOVE   = 1000e-9
SRC_ABOVE = 600e-9
MON_ABOVE = 750e-9
NUM_WL    = 100
MATERIAL_SIO2 = "SiO2 (Glass) - Palik"


# ── Load SiO2 n,k (Franta) ────────────────────────────────────────────────────
def load_nk_sio2(wl_nm_arr):
    with open(os.path.join(DB, "main/SiO2/nk/Franta.yml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        rows = np.array([[float(v) for v in ln.split()]
                         for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
        wl_db, n_db, k_db = rows[:, 0] * 1000, rows[:, 1], rows[:, 2]
    n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
    return n_fn, k_fn


def add_franta_material(fdtd, n_fn, k_fn):
    wl_pts  = np.linspace(300e-9, 800e-9, 200)
    n_pts   = n_fn(wl_pts * 1e9)
    k_pts   = k_fn(wl_pts * 1e9)
    # 2-col [wl_m, n] — SiO2 k≈0 in 400-750nm so lossless approx is fine
    nk_data  = np.column_stack([wl_pts, n_pts])
    mat_name = fdtd.addmaterial("Sampled data")
    fdtd.setmaterial(mat_name, "name", MATERIAL_SIO2)
    fdtd.setmaterial(MATERIAL_SIO2, "sampled data", nk_data)


# ── torcwa single geometry ────────────────────────────────────────────────────
def run_torcwa(period_nm, pillar_nm, height_nm, wl_nm, eps_sio2):
    p, w, h = period_nm/1000, pillar_nm/1000, height_nm/1000
    wl_um   = wl_nm / 1000
    si_cols = int(NX_RCWA * w / p)

    results = []
    for i, wl in enumerate(wl_um):
        ep = complex(eps_sio2[i])
        try:
            sim = torcwa.rcwa(freq=1.0/wl, order=[1, NG_RCWA], L=[p, p],
                              dtype=sim_dtype, device=device)
            sim.add_input_layer(eps=torch.tensor(1.0, dtype=sim_dtype, device=device))
            sim.add_output_layer(eps=torch.tensor(ep, dtype=sim_dtype, device=device))
            sim.set_incident_angle(inc_ang=1e-6, azi_ang=0.0)
            ep_grid = torch.ones(NX_RCWA, NX_RCWA, dtype=sim_dtype, device=device)
            ep_grid[:, :si_cols] = ep
            sim.add_layer(thickness=h, eps=ep_grid)
            sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
            sim.solve_global_smatrix()
            R = sim.S_parameters(orders=[0, 0], direction="forward",
                                 port="reflection", polarization="xx", ref_order=[0, 0])
            results.append(float(abs(R)**2))
        except Exception:
            results.append(float("nan"))
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    arr = np.array(results)
    nans = np.isnan(arr)
    if nans.any():
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


# ── Lumerical single geometry ─────────────────────────────────────────────────
def run_lumerical(fdtd, period_nm, pillar_nm, height_nm, wl_nm):
    period, pillar, height = period_nm*1e-9, pillar_nm*1e-9, height_nm*1e-9

    fdtd.switchtolayout()
    fdtd.eval("selectall; delete;")

    fdtd.addfdtd()
    fdtd.set("x", 0.0);          fdtd.set("x span", period)
    fdtd.set("y", 0.0);          fdtd.set("y span", Y_SPAN)
    fdtd.set("z min", -Z_BELOW); fdtd.set("z max", height + Z_ABOVE)
    fdtd.set("x min bc", "Periodic"); fdtd.set("x max bc", "Periodic")
    fdtd.set("y min bc", "Periodic"); fdtd.set("y max bc", "Periodic")

    fdtd.addrect()
    fdtd.set("name", "substrate")
    fdtd.set("x span", period*2); fdtd.set("y span", Y_SPAN*2)
    fdtd.set("z min", -Z_BELOW*1.5); fdtd.set("z max", 0.0)
    fdtd.set("material", MATERIAL_SIO2)

    fdtd.addrect()
    fdtd.set("name", "pillar")
    fdtd.set("x", 0.0); fdtd.set("x span", pillar)
    fdtd.set("y span", Y_SPAN*2)
    fdtd.set("z min", 0.0); fdtd.set("z max", height)
    fdtd.set("material", MATERIAL_SIO2)

    fdtd.addplane()
    fdtd.set("injection axis", "z-axis"); fdtd.set("direction", "Backward")
    fdtd.set("x span", period*2); fdtd.set("y span", Y_SPAN*2)
    fdtd.set("z", height + SRC_ABOVE)
    fdtd.set("polarization angle", 90.0)   # TE (s-pol), E in y, matches torcwa amplitude=[1,0]
    fdtd.set("wavelength start", wl_nm.min()*1e-9)
    fdtd.set("wavelength stop",  wl_nm.max()*1e-9)

    fdtd.addpower()
    fdtd.set("name", "R_mon")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x span", period*2); fdtd.set("y span", Y_SPAN*2)
    fdtd.set("z", height + MON_ABOVE)
    fdtd.set("override global monitor settings", 1)
    fdtd.set("frequency points", NUM_WL)

    fdtd.run()

    T_R    = fdtd.transmission("R_mon")
    f_data = np.array(fdtd.getdata("R_mon", "f")).flatten()
    wl_raw = (3e8 / f_data) * 1e9
    idx    = np.argsort(wl_raw)
    wl_raw = wl_raw[idx]
    R_raw  = np.array(T_R).flatten()[idx]

    return np.interp(wl_nm, wl_raw, R_raw).astype(np.float32)


# ── Load measured samples from h5 ────────────────────────────────────────────
def load_samples():
    samples = []
    if os.path.exists(H5_A):
        with h5py.File(H5_A, "r") as f:
            wl_nm  = f["lambda"][:].flatten()
            params = f["params"]["d680_el300"][0]
            eta_r  = f["eta_r"]["d680_el300"][0]
            period, height, groove = params[0], params[1], params[2]
            pillar = period - groove
            samples.append({"label": "A", "period": period, "pillar": pillar,
                             "height": height, "wl_nm": wl_nm, "meas": eta_r})
        print(f"Sample A: p={period:.0f}  pillar={pillar:.1f}  h={height:.1f} nm")
    if os.path.exists(H5_B):
        with h5py.File(H5_B, "r") as f:
            wl_nm_B = f["lambda"][:].flatten()
            best_key, best_err = None, 1e9
            for key in f["eta_r"].keys():
                p = f["params"][key][0]
                pb = p[0] - p[2]
                err = abs(p[0]-675) + abs(pb-351) + abs(p[1]-283)
                if err < best_err:
                    best_err, best_key = err, key
            if best_key:
                params = f["params"][best_key][0]
                eta_r  = f["eta_r"][best_key][0]
                period, height, groove = params[0], params[1], params[2]
                pillar = period - groove
                samples.append({"label": "B", "period": period, "pillar": pillar,
                                 "height": height, "wl_nm": wl_nm_B, "meas": eta_r})
                print(f"Sample B: p={period:.0f}  pillar={pillar:.1f}  h={height:.1f} nm  (key={best_key})")
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    if DB is None:
        raise RuntimeError("refractiveindex database not found.")

    samples = load_samples()
    if not samples:
        raise RuntimeError("No h5 data found in validation/")

    n_fn, k_fn = load_nk_sio2(np.linspace(400, 750, 100))

    fdtd = lumapi.FDTD(hide=True)
    # Using Lumerical built-in Palik SiO2; torcwa uses Franta — small n(λ) difference expected

    fig, axes = plt.subplots(1, len(samples), figsize=(6*len(samples), 4.5), squeeze=False)

    for i, s in enumerate(samples):
        wl_nm  = s["wl_nm"]
        period, pillar, height = s["period"], s["pillar"], s["height"]

        # Interpolate eps onto this sample's wavelength grid
        eps = (n_fn(wl_nm) + 1j * k_fn(wl_nm)) ** 2

        print(f"\n[Sample {s['label']}] Running torcwa ...")
        t0 = time.time()
        R_rcwa = run_torcwa(period, pillar, height, wl_nm, eps)
        print(f"  torcwa done in {time.time()-t0:.1f}s")

        print(f"[Sample {s['label']}] Running Lumerical ...")
        t0 = time.time()
        R_fdtd = run_lumerical(fdtd, period, pillar, height, wl_nm)
        print(f"  FDTD done in {time.time()-t0:.1f}s")

        mae_rcwa = np.mean(np.abs(R_rcwa  - s["meas"]))
        mae_fdtd = np.mean(np.abs(R_fdtd  - s["meas"]))
        print(f"  MAE torcwa vs meas : {mae_rcwa:.5f}")
        print(f"  MAE FDTD   vs meas : {mae_fdtd:.5f}")

        ax = axes[0][i]
        ax.plot(wl_nm, s["meas"],  "k-",  lw=2,   label="Measured (Mattila 2024)")
        ax.plot(wl_nm, R_rcwa,     "b-",  lw=1.5, label=f"torcwa NG={NG_RCWA}  MAE={mae_rcwa:.4f}")
        ax.plot(wl_nm, R_fdtd,     "r--", lw=1.5, label=f"Lumerical FDTD  MAE={mae_fdtd:.4f}")
        ax.set_title(f"Sample {s['label']}\n"
                     f"p={period:.0f}  w={pillar:.1f}  h={height:.1f} nm")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.legend(fontsize=8)
        ax.set_xlim(wl_nm.min(), wl_nm.max())

    fdtd.close()
    fig.suptitle("Lumerical FDTD & torcwa vs Mattila 2024 measured data\n"
                 "SiO2 grating, TE/s-pol, normal incidence", fontsize=11)
    fig.tight_layout()
    out = os.path.join(BASE, "compare_vs_mattila2024.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
