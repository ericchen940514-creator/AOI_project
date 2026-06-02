"""
Compare Lumerical RCWA vs Mattila et al. 2024 measured reflectance data.

Paper: Mattila et al., Meas. Sci. Technol. 35, 085025 (2024)
Structure: SiO2 1D grating on SiO2 substrate, TE/s-pol, normal incidence, 400-750nm

Key RCWA settings required:
  - propagation direction = "backward" (-z, light from air going down into SiO2)
  - report grating orders = 1  (enables grating_orders result)
  - Rs_grating(0,0) = 0th-order specular reflectance from air side = eta_r

Usage:
    python lumerical/compare_vs_mattila2024.py
"""
import sys, os, io, time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import yaml
from scipy.interpolate import interp1d

# ── Lumerical ──────────────────────────────────────────────────────────────────
LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python"
sys.path.insert(0, LUMAPI_PATH)
import lumapi

BASE      = os.path.dirname(os.path.abspath(__file__))
VALID_DIR = os.path.join(BASE, "..", "validation")
H5_A      = os.path.join(VALID_DIR, "v1_A_256.h5")
H5_B      = os.path.join(VALID_DIR, "v1_B_256.h5")

_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)

# ── Optional torcwa ────────────────────────────────────────────────────────────
try:
    import torch, torcwa
    HAS_TORCWA = True
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NG_RCWA    = 31
    NX_RCWA    = 65
    sim_dtype  = torch.complex128
except ImportError:
    HAS_TORCWA = False


# ── Load SiO2 n(λ) from refractiveindex database (Franta) ─────────────────────
def load_nk_sio2():
    if DB is None:
        return None, None
    try:
        with open(os.path.join(DB, "main/SiO2/nk/Franta.yml"), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for blk in data["DATA"]:
            rows = np.array([[float(v) for v in ln.split()]
                             for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
        wl_db, n_db, k_db = rows[:, 0]*1000, rows[:, 1], rows[:, 2]
        n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
        k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
        return n_fn, k_fn
    except Exception as e:
        print(f"  [torcwa] SiO2 database load failed: {e}")
        return None, None


# ── Load samples from h5 ───────────────────────────────────────────────────────
def extract_zero_order(result, key):
    if key not in result:
        raise KeyError(f"{key} not found. Available keys: {list(result.keys())}")

    values = np.asarray(result[key])
    n_order = np.asarray(result.get("n", result.get("N", [0]))).astype(int).flatten()
    m_order = np.asarray(result.get("m", result.get("M", np.zeros_like(n_order)))).astype(int).flatten()
    zero_order = np.where((n_order == 0) & (m_order == 0))[0]
    if zero_order.size == 0:
        raise ValueError(f"0th order not found. n={n_order.tolist()} m={m_order.tolist()}")

    order_axis = next(
        (axis for axis, size in enumerate(values.shape) if size == n_order.size),
        0,
    )
    return np.squeeze(np.take(values, int(zero_order[0]), axis=order_axis)).astype(float).flatten()


def load_samples():
    samples = []
    for h5_path, target, label in [
        (H5_A, {"period": 680, "height": 209.7, "pillar": 309.1}, "A"),
        (H5_B, {"period": 675, "height": 282.6, "pillar": 351.1}, "B"),
    ]:
        if not os.path.exists(h5_path):
            print(f"  WARNING: {h5_path} not found, skipping Sample {label}")
            continue
        with h5py.File(h5_path, "r") as f:
            wl_nm = np.array(f["lambda"]).flatten()
            # Find key whose params best match target
            best_key, best_err = None, 1e9
            for key in f["params"].keys():
                p = np.array(f["params"][key])[0]
                period_h5 = p[0]
                height_h5 = p[1]
                groove_h5 = p[2]
                pillar_h5 = period_h5 - groove_h5
                err = (abs(period_h5 - target["period"]) +
                       abs(height_h5 - target["height"]) +
                       abs(pillar_h5  - target["pillar"]))
                if err < best_err:
                    best_err, best_key = err, key
            p      = np.array(f["params"][best_key])[0]
            eta_r  = np.array(f["eta_r"][best_key]).flatten()
            period = p[0]; height = p[1]; groove = p[2]
            pillar = period - groove
        print(f"Sample {label}: key={best_key}  p={period:.0f}nm  "
              f"h={height:.1f}nm  pillar={pillar:.1f}nm  err={best_err:.1f}nm")
        samples.append({"label": label, "period": period, "pillar": pillar,
                        "height": height, "wl_nm": wl_nm, "eta_r": eta_r,
                        "period_m": period*1e-9})
    return samples


# ── Lumerical RCWA (1D grating, 2D Y-normal) ──────────────────────────────────
MATERIAL_SIO2 = "SiO2_Franta"


def add_franta_material(sim, n_fn, k_fn):
    if n_fn is None or k_fn is None:
        return "SiO2 (Glass) - Palik"
    try:
        wl_pts  = np.linspace(300e-9, 800e-9, 200)
        nk_data = np.column_stack([wl_pts, n_fn(wl_pts * 1e9)])
        mat_name = sim.addmaterial("Sampled data")
        sim.setmaterial(mat_name, "name", MATERIAL_SIO2)
        sim.setmaterial(MATERIAL_SIO2, "sampled data", nk_data)
        sim.setmaterial(MATERIAL_SIO2, "color", np.array([0.5, 0.8, 1.0, 1.0]))
        print(f"  Material '{MATERIAL_SIO2}' added (Franta SiO2)")
    except Exception as e:
        print(f"  Material '{MATERIAL_SIO2}' already exists or fallback: {e}")
    return MATERIAL_SIO2


def run_lumerical_rcwa(sim, period_nm, pillar_nm, height_nm, n_wl, material_name):
    period    = period_nm * 1e-9
    pillar    = pillar_nm * 1e-9
    height    = height_nm * 1e-9
    sub_thick = 2e-6
    zmin      = -sub_thick
    zmax      =  height    + 1e-6

    lsf = f"""
switchtolayout; selectall; delete;

addrect; set("name","substrate");
set("x",0); set("x span",{period});
set("y",0); set("y span",{period});
set("z min",{zmin - 1e-6}); set("z max",0);
set("material","{material_name}");

addrect; set("name","ridge");
set("x",0); set("x span",{pillar});
set("y",0); set("y span",{period*4});
set("z min",0); set("z max",{height});
set("material","{material_name}");

addrcwa; set("name","RCWA");
set("simulation region","2D Y-normal");
set("x",0); set("x span",{period});
set("y",0); set("y span",{period});
set("z min",{zmin}); set("z max",{zmax});
set("mesh cells x",100); set("mesh cells z",50);
set("frequency points",{n_wl});
set("angle theta",0); set("angle phi",0);
set("report grating orders",1);
set("propagation direction","backward");

setnamed("RCWA","interface absolute positions",[0,{height}]);

run;
"""
    sim.eval(lsf)

    te  = sim.getresult("RCWA", "total_energy")
    go  = sim.getresult("RCWA", "grating_orders")
    lam = np.array(te["lambda"]).flatten() * 1e9
    Rs  = np.array(te["Rs"]).flatten()
    Ts  = np.array(te["Ts"]).flatten()

    # grating_orders shape: (wl, pol, n_orders, m_orders)
    # axis2 = n (x-direction, 201 values in 2D Y-normal), axis3 = m (y-direction, 1 value)
    Rs_g  = np.array(go["Rs_grating"])
    n_arr = np.array(go["n"]).flatten().astype(int)
    m_arr = np.array(go["m"]).flatten().astype(int)
    n0    = int(np.where(n_arr == 0)[0][0])
    m0    = int(np.where(m_arr == 0)[0][0])
    R0s   = Rs_g[:, 0, n0, m0].flatten()

    idx = np.argsort(lam)
    return lam[idx], Rs[idx], Ts[idx], R0s[idx]


# ── torcwa 0th-order R ────────────────────────────────────────────────────────
def run_torcwa_0th(period_nm, pillar_nm, height_nm, wl_nm, eps_sio2):
    if not HAS_TORCWA:
        return None
    p, w, h = period_nm/1000, pillar_nm/1000, height_nm/1000
    si_cols = int(NX_RCWA * w / p)
    results = []
    for i, wl_um in enumerate(wl_nm / 1000):
        ep = complex(eps_sio2[i])
        try:
            s = torcwa.rcwa(freq=1.0/wl_um, order=[1, NG_RCWA], L=[p, p],
                            dtype=sim_dtype, device=device)
            s.add_input_layer(eps=torch.tensor(1.0, dtype=sim_dtype, device=device))
            s.add_output_layer(eps=torch.tensor(ep,  dtype=sim_dtype, device=device))
            s.set_incident_angle(inc_ang=1e-6, azi_ang=0.0)
            ep_grid = torch.ones(NX_RCWA, NX_RCWA, dtype=sim_dtype, device=device)
            ep_grid[:, :si_cols] = ep
            s.add_layer(thickness=h, eps=ep_grid)
            s.source_planewave(amplitude=[1.0, 0.0], direction="forward")
            s.solve_global_smatrix()
            R = s.S_parameters(orders=[0, 0], direction="forward",
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


CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_cache.npz")


def run_or_load_simulations(samples, n_fn, k_fn, force_rerun=False):
    if not force_rerun and os.path.exists(CACHE_FILE):
        print(f"Loading cached results from {CACHE_FILE}")
        d = np.load(CACHE_FILE, allow_pickle=True)
        cache = []
        for i in range(int(d["n_samples"])):
            cache.append({
                "lam_l": d[f"lam_l_{i}"],
                "R0_l":  d[f"R0_l_{i}"],
                "R_tw":  d[f"R_tw_{i}"] if f"R_tw_{i}" in d else None,
            })
        return cache

    sim = lumapi.FDTD(hide=True)
    material_name = add_franta_material(sim, n_fn, k_fn)
    cache = []
    save_dict = {"n_samples": len(samples)}

    for i, s in enumerate(samples):
        wl_nm  = s["wl_nm"]
        period = s["period"]; pillar = s["pillar"]; height = s["height"]
        label  = s["label"]

        print(f"\nSample {label}: Lumerical RCWA ...")
        t0 = time.time()
        sim.eval("switchtolayout;")
        sim.setglobalsource("wavelength start", wl_nm.min()*1e-9)
        sim.setglobalsource("wavelength stop",  wl_nm.max()*1e-9)
        sim.setglobalmonitor("use wavelength spacing", True)
        sim.setglobalmonitor("use source limits", True)
        sim.setglobalmonitor("frequency points", len(wl_nm))
        lam_l, Rs_l, Ts_l, R0_l = run_lumerical_rcwa(
            sim, period, pillar, height, len(wl_nm), material_name
        )
        print(f"  done in {time.time()-t0:.1f}s")

        R_tw = None
        if HAS_TORCWA and n_fn is not None:
            print(f"Sample {label}: torcwa ...")
            eps = (n_fn(wl_nm) + 1j*k_fn(wl_nm))**2
            t0  = time.time()
            R_tw = run_torcwa_0th(period, pillar, height, wl_nm, eps)
            print(f"  done in {time.time()-t0:.1f}s")

        cache.append({"lam_l": lam_l, "R0_l": R0_l, "R_tw": R_tw})
        save_dict[f"lam_l_{i}"] = lam_l
        save_dict[f"R0_l_{i}"]  = R0_l
        if R_tw is not None:
            save_dict[f"R_tw_{i}"] = R_tw

    sim.close()
    np.savez(CACHE_FILE, **save_dict)
    print(f"\nResults cached to {CACHE_FILE}")
    return cache


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="Force re-run simulations (ignore cache)")
    args = ap.parse_args()

    samples = load_samples()
    if not samples:
        print("No h5 data found.")
        return

    n_fn, k_fn = load_nk_sio2()
    cache = run_or_load_simulations(samples, n_fn, k_fn, force_rerun=args.rerun)

    n_results = len(samples)
    fig, axes = plt.subplots(1, n_results, figsize=(7*n_results, 5), squeeze=False)

    for i, s in enumerate(samples):
        wl_nm  = s["wl_nm"]
        period = s["period"]; pillar = s["pillar"]; height = s["height"]
        label  = s["label"]
        ax     = axes[0][i]
        c      = cache[i]

        lam_l = c["lam_l"]; R0_l = c["R0_l"]; R_tw = c["R_tw"]

        R0_interp = np.interp(wl_nm, lam_l, R0_l)
        mae_0th   = np.mean(np.abs(R0_interp - s["eta_r"]))

        mae_tw = np.mean(np.abs(R_tw - s["eta_r"])) if R_tw is not None else np.nan
        print(f"Sample {label}  Lumerical MAE={mae_0th:.4f}  torcwa MAE={mae_tw:.4f}")

        ax.plot(wl_nm, s["eta_r"], "k-",  lw=2.5, label="Mattila 2024 (measured)")
        ax.plot(lam_l, R0_l, "b-", lw=1.8,
                label=f"Lumerical RCWA 0th-order\nMAE={mae_0th:.4f}")
        if R_tw is not None:
            ax.plot(wl_nm, R_tw, "g--", lw=1.5,
                    label=f"torcwa NG={NG_RCWA}  MAE={mae_tw:.4f}")

        ax.set_title(f"Sample {label}: p={period:.0f}nm  w={pillar:.1f}nm  h={height:.1f}nm")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.legend(fontsize=7.5)
        ax.set_xlim(wl_nm.min(), wl_nm.max())
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Lumerical RCWA vs torcwa vs Mattila 2024\n"
                 "SiO2 1D grating, TE/s-pol, normal incidence", fontsize=11)
    fig.tight_layout()
    out = os.path.join(BASE, "compare_vs_mattila2024.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
