"""
Compare torcwa (RCWA) vs Lumerical FDTD for the same SiO2 grating geometry.

Usage:
    python lumerical/compare_torcwa_vs_fdtd.py
    python lumerical/compare_torcwa_vs_fdtd.py --period 680 --pillar 300 --height 200
    python lumerical/compare_torcwa_vs_fdtd.py --cases 5   # run N random cases
"""
import sys, os, io, argparse
import numpy as np
import yaml
import torch
import torcwa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python"
sys.path.append(LUMAPI_PATH)
import lumapi

BASE = os.path.dirname(os.path.abspath(__file__))

_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)

# ── Shared wavelength grid ────────────────────────────────────────────────────
WL_NM   = np.linspace(400.0, 750.0, 100)
WL_UM   = WL_NM / 1000.0
NG, NX  = 9, 21

# Lumerical geometry constants
Y_SPAN    = 20e-9
Z_BELOW   = 1000e-9
Z_ABOVE   = 1000e-9
SRC_ABOVE = 600e-9
MON_ABOVE = 750e-9
MON_BELOW = -500e-9
NUM_WL    = 100
MATERIAL_SIO2 = "SiO2_Franta"   # custom material added from same database as torcwa


# ── Load Franta SiO2 n,k ─────────────────────────────────────────────────────
def load_nk_sio2():
    with open(os.path.join(DB, "main/SiO2/nk/Franta.yml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        rows = np.array([[float(v) for v in ln.split()]
                         for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
        wl_db, n_db, k_db = rows[:, 0] * 1000, rows[:, 1], rows[:, 2]
    n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
    return n_fn, k_fn


def load_eps_sio2(n_fn, k_fn):
    return (n_fn(WL_NM) + 1j * k_fn(WL_NM)) ** 2


# ── Add Franta SiO2 as a sampled material in the open FDTD session ────────────
def add_franta_material(fdtd, n_fn, k_fn):
    """Adds 'SiO2_Franta' sampled material if it doesn't already exist."""
    # Check if already added this session
    try:
        fdtd.eval(f'getindex("{MATERIAL_SIO2}", 0.5e-6);')
        return   # already exists
    except Exception:
        pass

    wl_pts = np.linspace(300e-9, 800e-9, 200)   # dense grid (m)
    n_pts  = n_fn(wl_pts * 1e9)
    k_pts  = k_fn(wl_pts * 1e9)

    # Build Nx3 matrix: [wavelength(m), n, k]
    nk_data  = np.column_stack([wl_pts, n_pts, k_pts])
    mat_name = fdtd.addmaterial("Sampled data")
    fdtd.setmaterial(mat_name, "name", MATERIAL_SIO2)
    fdtd.setmaterial(MATERIAL_SIO2, "sampled data", nk_data)
    fdtd.setmaterial(MATERIAL_SIO2, "color", np.array([0.5, 0.8, 1.0, 1.0]))


# ── torcwa single simulation ──────────────────────────────────────────────────
def run_torcwa(period_nm, pillar_nm, height_nm, eps_sio2):
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p, w, h = period_nm/1000, pillar_nm/1000, height_nm/1000
    si_cols = max(1, min(int(NX * w / p), NX - 1))
    mask = torch.zeros(NX, NX, dtype=torch.bool, device=dev)
    mask[:, :si_cols] = True

    R_list = []
    for i, wl in enumerate(WL_UM):
        ep   = complex(eps_sio2[i])
        ep_t = torch.tensor(ep, dtype=torch.complex64, device=dev)

        sim = torcwa.rcwa(freq=1.0/wl, order=[1, NG], L=[p, p],
                          dtype=torch.complex64, device=dev)
        sim.add_input_layer(eps=torch.tensor(1.0, dtype=torch.complex64, device=dev))
        sim.add_output_layer(eps=ep_t)
        sim.set_incident_angle(inc_ang=1e-6, azi_ang=0.0)

        ep_grid = torch.ones(NX, NX, dtype=torch.complex64, device=dev)
        ep_grid[mask] = ep_t
        sim.add_layer(thickness=h, eps=ep_grid)

        sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
        sim.solve_global_smatrix()

        R = sim.S_parameters(orders=[0, 0], direction="forward",
                             port="reflection", polarization="xx", ref_order=[0, 0])
        R_list.append(float(abs(R) ** 2))

    return np.array(R_list, dtype=np.float32)


# ── Lumerical single simulation ───────────────────────────────────────────────
def run_lumerical(fdtd, period_nm, pillar_nm, height_nm):
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
    fdtd.set("polarization angle", 90.0)   # TE (s-pol), matches torcwa amplitude=[1,0]
    fdtd.set("wavelength start", 400e-9); fdtd.set("wavelength stop", 750e-9)

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

    return np.interp(WL_NM, wl_raw, R_raw).astype(np.float32)


# ── Compare one case ──────────────────────────────────────────────────────────
def compare_one(fdtd, eps_sio2, period_nm, pillar_nm, height_nm):
    print(f"  torcwa ... p={period_nm:.0f} w={pillar_nm:.0f} h={height_nm:.0f}")
    R_rcwa = run_torcwa(period_nm, pillar_nm, height_nm, eps_sio2)

    print(f"  FDTD   ...")
    R_fdtd = run_lumerical(fdtd, period_nm, pillar_nm, height_nm)

    mae  = np.mean(np.abs(R_rcwa - R_fdtd))
    rmse = np.sqrt(np.mean((R_rcwa - R_fdtd)**2))
    return R_rcwa, R_fdtd, mae, rmse


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=float, default=None)
    parser.add_argument("--pillar", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--cases",  type=int,   default=1,
                        help="Number of random cases to compare")
    parser.add_argument("--seed",   type=int,   default=0)
    args = parser.parse_args()

    if DB is None:
        raise RuntimeError("refractiveindex database not found.")

    n_fn, k_fn = load_nk_sio2()
    eps_sio2   = load_eps_sio2(n_fn, k_fn)
    fdtd       = lumapi.FDTD(hide=True)
    add_franta_material(fdtd, n_fn, k_fn)

    # Build case list
    if args.period and args.pillar and args.height:
        cases = [(args.period, args.pillar, args.height)]
    else:
        rng = np.random.default_rng(args.seed)
        periods = rng.uniform(600, 760, args.cases)
        heights = rng.uniform(100, 400, args.cases)
        pillars = [rng.uniform(150, min(500, p-10)) for p in periods]
        cases   = list(zip(periods, pillars, heights))

    all_mae, all_rmse = [], []

    ncols = min(len(cases), 3)
    nrows = (len(cases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5*ncols, 3.5*nrows), squeeze=False)

    for i, (period, pillar, height) in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}]")
        R_rcwa, R_fdtd, mae, rmse = compare_one(
            fdtd, eps_sio2, period, pillar, height)

        all_mae.append(mae)
        all_rmse.append(rmse)
        print(f"  MAE={mae:.5f}  RMSE={rmse:.5f}")

        ax = axes[i // ncols][i % ncols]
        ax.plot(WL_NM, R_rcwa, label="torcwa (RCWA)")
        ax.plot(WL_NM, R_fdtd, "--", label="Lumerical (FDTD)")
        ax.set_title(f"p={period:.0f} w={pillar:.0f} h={height:.0f}\n"
                     f"MAE={mae:.4f}  RMSE={rmse:.4f}", fontsize=9)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        ax.legend(fontsize=8)
        ax.set_xlim(400, 750)

    # Hide unused subplots
    for j in range(len(cases), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fdtd.close()

    print(f"\n=== Summary ({'%d cases' % len(cases)}) ===")
    print(f"  Mean MAE  : {np.mean(all_mae):.5f}")
    print(f"  Mean RMSE : {np.mean(all_rmse):.5f}")
    print(f"  Max  MAE  : {np.max(all_mae):.5f}")

    fig.suptitle("torcwa RCWA vs Lumerical FDTD  —  SiO2 grating", fontsize=11)
    fig.tight_layout()
    out = os.path.join(BASE, "compare_torcwa_vs_fdtd.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
