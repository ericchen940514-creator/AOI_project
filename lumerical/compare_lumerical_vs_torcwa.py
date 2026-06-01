"""
Compare Lumerical RCWA vs torcwa on 500 identical SiO2 grating samples.

Both solvers use the same:
  - 500 parameter sets (seed=42, same sampling as generate scripts)
  - Wavelength grid: 400-750 nm, 100 points
  - Material: SiO2 Franta (refractiveindex.info database)
  - Structure: 1D grating on SiO2 substrate, air superstrate, s-pol normal incidence

Lumerical RCWA settings (validated vs Mattila 2024):
  - 2D Y-normal, propagation direction = "backward", mesh_cells_x=100
  - 0th-order specular R from grating_orders.Rs_grating

torcwa settings (matches generate_torcwa_sio2.py):
  - NG=9, NX=21, complex64, multiprocessing (--workers N), forward direction, 0th-order R

Outputs:
  data_generation/output/lumerical_rcwa_500.csv
  data_generation/output/torcwa_rcwa_500.csv
  lumerical/compare_lumerical_vs_torcwa.png

Usage:
    python lumerical/compare_lumerical_vs_torcwa.py           # full 500 rows
    python lumerical/compare_lumerical_vs_torcwa.py --test    # 10 rows
    python lumerical/compare_lumerical_vs_torcwa.py --plot-only  # skip sim, just plot
"""
import sys, os, csv, io, time, argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Lumerical ──────────────────────────────────────────────────────────────────
LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python"
sys.path.insert(0, LUMAPI_PATH)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "..", "data_generation", "output")

_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)

# ── Shared settings ────────────────────────────────────────────────────────────
WL_NM_MIN, WL_NM_MAX, NUM_WL = 400.0, 750.0, 100
PERIOD_MIN, PERIOD_MAX = 600, 760
PILLAR_MIN, PILLAR_MAX = 150, 500
HEIGHT_MIN, HEIGHT_MAX = 100, 400

WL_GRID = np.linspace(WL_NM_MIN, WL_NM_MAX, NUM_WL)
WL_COLS = [f"R_{int(w)}" for w in WL_GRID]
CSV_COLS = ["period_nm", "pillar_nm", "height_nm"] + WL_COLS

# ── torcwa settings (matches generate_torcwa_sio2.py default) ─────────────────
NG_TW = 9
NX_TW = 21

# ── Material ───────────────────────────────────────────────────────────────────
def load_nk_sio2():
    if DB is None:
        raise RuntimeError("refractiveindex database not found")
    with open(os.path.join(DB, "main/SiO2/nk/Franta.yml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        rows = np.array([[float(v) for v in ln.split()]
                         for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
        wl_db, n_db, k_db = rows[:, 0] * 1000, rows[:, 1], rows[:, 2]
    n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
    return n_fn, k_fn


# ── Parameter sampling ─────────────────────────────────────────────────────────
def sample_params(nrows, seed=42):
    rng = np.random.default_rng(seed)
    periods = rng.uniform(PERIOD_MIN, PERIOD_MAX, nrows)
    heights = rng.uniform(HEIGHT_MIN, HEIGHT_MAX, nrows)
    pillars = np.array([
        rng.uniform(max(PILLAR_MIN, 10), min(PILLAR_MAX, p - 10))
        for p in periods
    ])
    return periods, pillars, heights


# ── Lumerical RCWA ─────────────────────────────────────────────────────────────
MATERIAL_SIO2 = "SiO2_Franta"


def add_franta_material(sim, n_fn, k_fn):
    try:
        sim.eval(f'getindex("{MATERIAL_SIO2}", 0.5e-6);')
        return
    except Exception:
        pass
    wl_pts = np.linspace(300e-9, 800e-9, 200)
    nk_data = np.column_stack([wl_pts, n_fn(wl_pts * 1e9)])
    mat_name = sim.addmaterial("Sampled data")
    sim.setmaterial(mat_name, "name", MATERIAL_SIO2)
    sim.setmaterial(MATERIAL_SIO2, "sampled data", nk_data)
    sim.setmaterial(MATERIAL_SIO2, "color", np.array([0.5, 0.8, 1.0, 1.0]))


def lumerical_rcwa_one(sim, period_nm, pillar_nm, height_nm, n_wl):
    period = period_nm * 1e-9
    pillar = pillar_nm * 1e-9
    height = height_nm * 1e-9
    sub_thick = 2e-6
    zmin = -sub_thick
    zmax = height + 1e-6

    lsf = f"""
switchtolayout; selectall; delete;

addrect; set("name","substrate");
set("x",0); set("x span",{period});
set("y",0); set("y span",{period});
set("z min",{zmin - 1e-6}); set("z max",0);
set("material","{MATERIAL_SIO2}");

addrect; set("name","ridge");
set("x",0); set("x span",{pillar});
set("y",0); set("y span",{period * 4});
set("z min",0); set("z max",{height});
set("material","{MATERIAL_SIO2}");

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
    go  = sim.getresult("RCWA", "grating_orders")
    te  = sim.getresult("RCWA", "total_energy")
    lam = np.array(te["lambda"]).flatten() * 1e9

    Rs_g  = np.array(go["Rs_grating"])
    m_arr = np.array(go["m"]).flatten().astype(int)
    m0    = int(np.where(m_arr == 0)[0][0])
    n_ax  = 0 if Rs_g.ndim < 4 or Rs_g.shape[3] == 1 else int(
        np.where(np.array(go["n"]).flatten().astype(int) == 0)[0][0])
    R0 = Rs_g[:, 0, m0, n_ax].flatten()

    idx = np.argsort(lam)
    lam_s, R0_s = lam[idx], R0[idx]
    return np.interp(WL_GRID, lam_s, R0_s)


def run_lumerical(nrows, seed, outfile):
    import lumapi

    os.makedirs(OUTDIR, exist_ok=True)
    done = 0
    if os.path.exists(outfile):
        done = max(sum(1 for _ in open(outfile, encoding="utf-8")) - 1, 0)
        print(f"  [Lumerical] Resuming from row {done}/{nrows}")
    else:
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLS)

    if done >= nrows:
        print("  [Lumerical] Already complete.")
        return

    periods, pillars, heights = sample_params(nrows, seed)
    n_fn, k_fn = load_nk_sio2()

    sim = lumapi.FDTD(hide=True)
    sim.setglobalsource("wavelength start", WL_NM_MIN * 1e-9)
    sim.setglobalsource("wavelength stop",  WL_NM_MAX * 1e-9)
    sim.setglobalmonitor("use wavelength spacing", True)
    sim.setglobalmonitor("use source limits", True)
    sim.setglobalmonitor("frequency points", NUM_WL)
    add_franta_material(sim, n_fn, k_fn)

    t0 = time.time()
    count = 0
    with open(outfile, "a", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        for i in range(done, nrows):
            try:
                R = lumerical_rcwa_one(sim, periods[i], pillars[i], heights[i], NUM_WL)
                writer.writerow([periods[i], pillars[i], heights[i]] + R.tolist())
                fout.flush()
                count += 1
                elapsed = (time.time() - t0) / 60
                rate = count / max(elapsed, 1e-6)
                eta  = (nrows - done - count) / max(rate, 1e-6)
                print(f"  [Lum {done+count:4d}/{nrows}] "
                      f"p={periods[i]:.0f} w={pillars[i]:.0f} h={heights[i]:.0f}  "
                      f"{elapsed:.1f}min  ETA {eta:.1f}min")
            except Exception as e:
                print(f"  [Lum ERROR row {i}] {e} — skipping")

    sim.close()
    print(f"  [Lumerical] Done: {count} rows in {(time.time()-t0)/60:.1f}min")


# ── torcwa multiprocessing workers (mirrors generate_torcwa_sio2.py) ───────────
_tw_eps = None
_tw_dev = None
_tw_dtype = None


def _tw_worker_init():
    global _tw_eps, _tw_dev, _tw_dtype
    import torch
    _tw_dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tw_dtype = torch.complex64
    n_fn, k_fn = load_nk_sio2()
    _tw_eps = (n_fn(WL_GRID) + 1j * k_fn(WL_GRID)).astype(np.complex64) ** 2


def _tw_worker_fn(args):
    import torch
    import torcwa as tw
    idx, period, pillar, height = args
    p = period / 1000.0
    w = pillar  / 1000.0
    h = height  / 1000.0
    si_cols = max(1, min(int(NX_TW * w / p), NX_TW - 1))

    results = []
    for j, wl_um in enumerate(WL_GRID / 1000.0):
        ep   = complex(_tw_eps[j])
        ep_t = torch.tensor(ep, dtype=_tw_dtype, device=_tw_dev)
        s = tw.rcwa(freq=1.0 / wl_um, order=[1, NG_TW],
                    L=[p, p], dtype=_tw_dtype, device=_tw_dev)
        s.add_input_layer(eps=torch.tensor(1.0, dtype=_tw_dtype, device=_tw_dev))
        s.add_output_layer(eps=ep_t)
        s.set_incident_angle(inc_ang=1e-6, azi_ang=0.0)
        ep_grid = torch.ones(NX_TW, NX_TW, dtype=_tw_dtype, device=_tw_dev)
        ep_grid[:, :si_cols] = ep_t
        s.add_layer(thickness=h, eps=ep_grid)
        s.source_planewave(amplitude=[1.0, 0.0], direction="forward")
        s.solve_global_smatrix()
        R = s.S_parameters(orders=[0, 0], direction="forward",
                           port="reflection", polarization="xx",
                           ref_order=[0, 0])
        results.append(float(abs(R) ** 2))
        del s, ep_grid, ep_t

    if _tw_dev.type == "cuda":
        torch.cuda.empty_cache()
    return idx, period, pillar, height, np.array(results, dtype=np.float32)


def run_torcwa(nrows, seed, outfile, workers=3):
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  [torcwa] {gpu_name}  NG={NG_TW}  NX={NX_TW}  dtype=complex64  workers={workers}")

    os.makedirs(OUTDIR, exist_ok=True)
    done = 0
    if os.path.exists(outfile):
        done = max(sum(1 for _ in open(outfile, encoding="utf-8")) - 1, 0)
        print(f"  [torcwa] Resuming from row {done}/{nrows}")
    else:
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLS)

    if done >= nrows:
        print("  [torcwa] Already complete.")
        return

    periods, pillars, heights = sample_params(nrows, seed)
    tasks = [(i, periods[i], pillars[i], heights[i]) for i in range(done, nrows)]

    t0 = time.time()
    count = 0

    # Collect results keyed by idx to write in order
    pending = {}

    with open(outfile, "a", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        next_write = done

        def flush_pending():
            nonlocal next_write
            while next_write in pending:
                _, per, pil, hei, R = pending.pop(next_write)
                writer.writerow([per, pil, hei] + R.tolist())
                next_write += 1
            fout.flush()

        if workers <= 1:
            _tw_worker_init()
            for t in tasks:
                try:
                    idx, per, pil, hei, R = _tw_worker_fn(t)
                    pending[idx] = (idx, per, pil, hei, R)
                    flush_pending()
                    count += 1
                    elapsed = (time.time() - t0) / 60
                    rate = count / max(elapsed, 1e-6)
                    eta  = (len(tasks) - count) / max(rate, 1e-6)
                    print(f"  [tw  {done+count:4d}/{nrows}] "
                          f"p={per:.0f} w={pil:.0f} h={hei:.0f}  "
                          f"{elapsed:.1f}min  ETA {eta:.1f}min")
                except Exception as e:
                    print(f"  [tw  ERROR] {e}")
        else:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_tw_worker_init) as pool:
                futures = {pool.submit(_tw_worker_fn, t): t[0] for t in tasks}
                for future in as_completed(futures):
                    try:
                        idx, per, pil, hei, R = future.result()
                        pending[idx] = (idx, per, pil, hei, R)
                        flush_pending()
                        count += 1
                        elapsed = (time.time() - t0) / 60
                        rate = count / max(elapsed, 1e-6)
                        eta  = (len(tasks) - count) / max(rate, 1e-6)
                        print(f"  [tw  {done+count:4d}/{nrows}] "
                              f"p={per:.0f} w={pil:.0f} h={hei:.0f}  "
                              f"{elapsed:.1f}min  ETA {eta:.1f}min")
                    except Exception as e:
                        print(f"  [tw  ERROR] {e}")

    print(f"  [torcwa] Done: {count} rows in {(time.time()-t0)/60:.1f}min")


# ── Plot ───────────────────────────────────────────────────────────────────────
def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    params = data[:, :3]
    spectra = data[:, 3:]
    return params, spectra


def make_plot(lum_csv, tw_csv, out_png):
    params_l, spec_l = load_csv(lum_csv)
    params_t, spec_t = load_csv(tw_csv)

    # Align on common rows (in case one has more rows)
    n = min(len(spec_l), len(spec_t))
    spec_l, spec_t = spec_l[:n], spec_t[:n]
    params = params_l[:n]

    diff    = spec_l - spec_t                  # (n, wl)
    mae_per = np.mean(np.abs(diff), axis=1)    # per-sample MAE
    mae_per_wl = np.mean(np.abs(diff), axis=0) # per-wavelength MAE

    print(f"\n=== Comparison ({n} samples) ===")
    print(f"  Mean MAE  : {np.mean(mae_per):.5f}  ({np.mean(mae_per)*100:.3f}%)")
    print(f"  Median MAE: {np.median(mae_per):.5f}")
    print(f"  Max MAE   : {np.max(mae_per):.5f}")
    print(f"  p95 MAE   : {np.percentile(mae_per, 95):.5f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 1. MAE histogram
    ax = axes[0, 0]
    ax.hist(mae_per, bins=50, edgecolor="k", alpha=0.7, color="steelblue")
    ax.axvline(np.mean(mae_per), color="r", ls="--", label=f"mean={np.mean(mae_per):.4f}")
    ax.set_xlabel("Per-sample MAE"); ax.set_ylabel("Count")
    ax.set_title("MAE distribution (per sample)")
    ax.legend()

    # 2. MAE vs wavelength
    ax = axes[0, 1]
    ax.plot(WL_GRID, mae_per_wl * 100, color="steelblue")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("MAE (%)")
    ax.set_title("MAE vs wavelength (averaged over all samples)")
    ax.grid(True, alpha=0.3)

    # 3. Scatter: Lumerical vs torcwa at 550nm
    wl_idx = np.argmin(np.abs(WL_GRID - 550))
    ax = axes[0, 2]
    ax.scatter(spec_t[:, wl_idx], spec_l[:, wl_idx], s=4, alpha=0.4, color="steelblue")
    lim = [0, max(spec_t[:, wl_idx].max(), spec_l[:, wl_idx].max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1, label="1:1")
    ax.set_xlabel("torcwa R"); ax.set_ylabel("Lumerical RCWA R")
    ax.set_title(f"Scatter at λ=550nm (n={n})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4-6. Example spectra (worst, median, best MAE)
    order  = np.argsort(mae_per)
    titles = ["Best match", "Median match", "Worst match"]
    idxs   = [order[0], order[n // 2], order[-1]]
    for k, (ax, idx, title) in enumerate(zip(axes[1], idxs, titles)):
        p, w, h = params[idx]
        ax.plot(WL_GRID, spec_l[idx], "b-",  lw=1.8, label="Lumerical RCWA")
        ax.plot(WL_GRID, spec_t[idx], "g--", lw=1.5, label=f"torcwa (NG={NG_TW})")
        ax.set_title(f"{title}\np={p:.0f} w={w:.0f} h={h:.0f}nm  MAE={mae_per[idx]:.4f}")
        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Lumerical RCWA vs torcwa  —  SiO2 grating, {n} samples\n"
        f"Mean MAE = {np.mean(mae_per)*100:.3f}%   "
        f"Lumerical: mesh_cells_x=100   torcwa: NG={NG_TW} NX={NX_TW}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_png}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows",     type=int, default=500)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--workers",   type=int, default=3, help="torcwa parallel workers")
    parser.add_argument("--test",      action="store_true", help="10 rows only")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulations, just plot")
    parser.add_argument("--lum-only",  action="store_true", help="Run Lumerical only")
    parser.add_argument("--tw-only",   action="store_true", help="Run torcwa only")
    args = parser.parse_args()

    nrows   = 10 if args.test else args.nrows
    lum_csv = os.path.join(OUTDIR, "lumerical_rcwa_500.csv")
    tw_csv  = os.path.join(OUTDIR, "torcwa_rcwa_500.csv")
    out_png = os.path.join(BASE, "compare_lumerical_vs_torcwa.png")

    if not args.plot_only:
        if not args.tw_only:
            print(f"\n=== Lumerical RCWA ({nrows} rows) ===")
            run_lumerical(nrows, args.seed, lum_csv)

        if not args.lum_only:
            print(f"\n=== torcwa NG={NG_TW} ({nrows} rows, {args.workers} workers) ===")
            run_torcwa(nrows, args.seed, tw_csv, workers=args.workers)

    if os.path.exists(lum_csv) and os.path.exists(tw_csv):
        print("\n=== Generating comparison plot ===")
        make_plot(lum_csv, tw_csv, out_png)
    else:
        missing = [p for p in [lum_csv, tw_csv] if not os.path.exists(p)]
        print(f"Missing CSVs, cannot plot: {missing}")


if __name__ == "__main__":
    main()
