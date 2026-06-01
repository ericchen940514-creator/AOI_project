"""
Batch SiO2 grating FDTD dataset generator using Lumerical v241.
Same parameter distribution as sio2/generate_torcwa_sio2.py.
Output CSV has identical columns so both datasets can be compared directly.

Usage:
    python lumerical/generate_lumerical_sio2.py                   # 500 rows
    python lumerical/generate_lumerical_sio2.py --nrows 100 --test
    python lumerical/generate_lumerical_sio2.py --nrows 2000
"""
import sys, os, csv, time, argparse, io
import numpy as np
import yaml
from scipy.interpolate import interp1d

LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python"
sys.path.append(LUMAPI_PATH)
import lumapi

# ── Match torcwa parameter space ──────────────────────────────────────────────
WL_START_NM = 400.0
WL_STOP_NM  = 750.0
NUM_WL      = 100

PERIOD_MIN, PERIOD_MAX = 600, 760
PILLAR_MIN, PILLAR_MAX = 150, 500
HEIGHT_MIN, HEIGHT_MAX = 100, 400

# Geometry constants (metres)
Z_BELOW   = 1000e-9
Z_ABOVE   = 1000e-9
Y_SPAN    = 20e-9
SRC_ABOVE = 600e-9
MON_ABOVE = 750e-9
MON_BELOW = -500e-9

MATERIAL_SIO2 = "SiO2_Franta"

BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "..", "data_generation", "output")

_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)


def load_nk_sio2():
    if DB is None:
        raise RuntimeError("refractiveindex database not found.")
    with open(os.path.join(DB, "main/SiO2/nk/Franta.yml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for block in data["DATA"]:
        rows = np.array([
            [float(v) for v in line.split()]
            for line in io.StringIO(block["data"]).readlines()
            if line.strip()
        ])
        wl_db, n_db, k_db = rows[:, 0] * 1000, rows[:, 1], rows[:, 2]
    n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
    return n_fn, k_fn


def add_franta_material(fdtd, n_fn, k_fn):
    try:
        fdtd.eval(f'getindex("{MATERIAL_SIO2}", 0.5e-6);')
        return
    except Exception:
        pass

    wl_pts = np.linspace(300e-9, 800e-9, 200)
    # k≈0 for SiO2 in 300-800nm; Lumerical v241 requires 2 or 4 cols, not 3
    nk_data = np.column_stack([wl_pts, n_fn(wl_pts * 1e9)])
    mat_name = fdtd.addmaterial("Sampled data")
    fdtd.setmaterial(mat_name, "name", MATERIAL_SIO2)
    fdtd.setmaterial(MATERIAL_SIO2, "sampled data", nk_data)
    fdtd.setmaterial(MATERIAL_SIO2, "color", np.array([0.5, 0.8, 1.0, 1.0]))


# ── Parameter sampling (same seed/distribution as torcwa script) ──────────────
def sample_params(nrows: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    periods = rng.uniform(PERIOD_MIN, PERIOD_MAX, nrows)
    heights = rng.uniform(HEIGHT_MIN, HEIGHT_MAX, nrows)
    pillars = np.array([
        rng.uniform(max(PILLAR_MIN, 10), min(PILLAR_MAX, p - 10))
        for p in periods
    ])
    return periods, pillars, heights


# ── Single FDTD simulation ────────────────────────────────────────────────────
def simulate_one(fdtd, period_nm: float, pillar_nm: float, height_nm: float) -> np.ndarray:
    """
    Reuses an open FDTD session. Clears and rebuilds the project each call.
    Returns reflectance array (NUM_WL,) on a fixed wavelength grid.
    """
    period = period_nm * 1e-9
    pillar = pillar_nm * 1e-9
    height = height_nm * 1e-9

    # Clear previous simulation without triggering a save dialog
    fdtd.switchtolayout()
    fdtd.eval("selectall; delete;")

    # FDTD region
    fdtd.addfdtd()
    fdtd.set("x", 0.0);          fdtd.set("x span", period)
    fdtd.set("y", 0.0);          fdtd.set("y span", Y_SPAN)
    fdtd.set("z min", -Z_BELOW); fdtd.set("z max", height + Z_ABOVE)
    fdtd.set("x min bc", "Periodic"); fdtd.set("x max bc", "Periodic")
    fdtd.set("y min bc", "Periodic"); fdtd.set("y max bc", "Periodic")

    # Substrate
    fdtd.addrect()
    fdtd.set("name", "substrate")
    fdtd.set("x span", period * 2); fdtd.set("y span", Y_SPAN * 2)
    fdtd.set("z min", -Z_BELOW * 1.5); fdtd.set("z max", 0.0)
    fdtd.set("material", MATERIAL_SIO2)

    # Grating pillar
    fdtd.addrect()
    fdtd.set("name", "pillar")
    fdtd.set("x", 0.0); fdtd.set("x span", pillar)
    fdtd.set("y span", Y_SPAN * 2)
    fdtd.set("z min", 0.0); fdtd.set("z max", height)
    fdtd.set("material", MATERIAL_SIO2)

    # Source
    fdtd.addplane()
    fdtd.set("name", "source")
    fdtd.set("injection axis", "z-axis"); fdtd.set("direction", "Backward")
    fdtd.set("x span", period * 2);      fdtd.set("y span", Y_SPAN * 2)
    fdtd.set("z", height + SRC_ABOVE)
    fdtd.set("polarization angle", 90.0)   # TE (s-pol), matches torcwa amplitude=[1,0]
    fdtd.set("wavelength start", WL_START_NM * 1e-9)
    fdtd.set("wavelength stop",  WL_STOP_NM  * 1e-9)

    # R monitor (above source)
    fdtd.addpower()
    fdtd.set("name", "R_mon")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x span", period * 2); fdtd.set("y span", Y_SPAN * 2)
    fdtd.set("z", height + MON_ABOVE)
    fdtd.set("override global monitor settings", 1)
    fdtd.set("frequency points", NUM_WL)

    fdtd.run()

    T_R    = fdtd.transmission("R_mon")
    f_data = np.array(fdtd.getdata("R_mon", "f")).flatten()
    wl     = (3e8 / f_data) * 1e9          # m → nm, may be descending
    sort_i = np.argsort(wl)
    wl     = wl[sort_i]

    R_raw = np.array(T_R).flatten()[sort_i]

    # Interpolate onto fixed wavelength grid (matches torcwa output)
    wl_grid = np.linspace(WL_START_NM, WL_STOP_NM, NUM_WL)
    R_interp = np.interp(wl_grid, wl, R_raw)

    return R_interp


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows",   type=int,   default=500)
    parser.add_argument("--test",    action="store_true", help="Run 10 rows only")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--out",     default=None)
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    nrows   = 10 if args.test else args.nrows
    outfile = args.out or os.path.join(OUTDIR, "lumerical_sio2_train.csv")

    wl_grid = np.linspace(WL_START_NM, WL_STOP_NM, NUM_WL)
    wl_cols = [f"R_{int(wl)}" for wl in wl_grid]
    cols    = ["period_nm", "pillar_nm", "height_nm"] + wl_cols

    # Resume support
    if os.path.exists(outfile):
        done = max(sum(1 for _ in open(outfile, encoding="utf-8")) - 1, 0)
        print(f"Resuming from row {done}/{nrows}")
    else:
        done = 0
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)

    if done >= nrows:
        print("Already complete.")
        return

    periods, pillars, heights = sample_params(nrows, seed=args.seed)

    print(f"Lumerical v241 | rows={nrows} | output={outfile}")

    n_fn, k_fn = load_nk_sio2()
    fdtd = lumapi.FDTD(hide=True)
    add_franta_material(fdtd, n_fn, k_fn)
    t0   = time.time()
    count = 0

    with open(outfile, "a", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        for i in range(done, nrows):
            try:
                R = simulate_one(fdtd, periods[i], pillars[i], heights[i])
                writer.writerow([periods[i], pillars[i], heights[i]] + R.tolist())
                fout.flush()
                count += 1
                elapsed = (time.time() - t0) / 60
                rate    = count / max(elapsed, 1e-6)
                remaining = (nrows - done - count) / max(rate, 1e-6)
                print(f"  [{done+count}/{nrows}]  "
                      f"p={periods[i]:.0f} w={pillars[i]:.0f} h={heights[i]:.0f}  "
                      f"{elapsed:.1f}min elapsed  ~{remaining:.1f}min left")
            except Exception as e:
                print(f"  [ERROR row {i}] {e} — skipping")

    fdtd.close()
    total_min = (time.time() - t0) / 60
    print(f"\nDone: {count} rows in {total_min:.1f}min  ({count/max(total_min,0.01):.1f} rows/min)")
    print(f"Saved → {outfile}")


if __name__ == "__main__":
    main()
