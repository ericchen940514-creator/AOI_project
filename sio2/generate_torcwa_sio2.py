"""
Generate SiO2 grating RCWA dataset using torcwa (GPU) + multiprocessing.
Optimised: NG=9, complex64, N_WORKERS parallel processes sharing GPU.

Usage:
    python generate_torcwa_sio2.py                      # 5000 rows, 3 workers
    python generate_torcwa_sio2.py --nrows 2000
    python generate_torcwa_sio2.py --nrows 100 --test   # quick check
    python generate_torcwa_sio2.py --workers 4          # more parallelism

Speed estimate on RTX 4070 Super (100 wl, NG=9, 3 workers):
    ~2-3s per sample → 2000 rows ≈ 1-2 hours
"""
import csv, io, os, time, argparse
import numpy as np
import yaml
import torch
import torcwa
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────────────────────
WL_NM_MIN, WL_NM_MAX, NUM_WL = 400.0, 750.0, 256
NG    = 9
NX    = 21
CHUNK = 50

PERIOD_MIN, PERIOD_MAX = 600, 760
PILLAR_MIN, PILLAR_MAX = 150, 500
HEIGHT_MIN, HEIGHT_MAX = 100, 400

wavelengths_nm = np.linspace(WL_NM_MIN, WL_NM_MAX, NUM_WL)
wavelengths_um = wavelengths_nm / 1000.0

sim_dtype = torch.complex64

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "..", "data_generation", "output")

_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)


# ── SiO2 epsilon ──────────────────────────────────────────────────────────────
def load_eps_sio2():
    with open(os.path.join(DB, "main/SiO2/nk/Franta.yml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        rows = np.array([[float(v) for v in ln.split()]
                         for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
        wl_db, n_db, k_db = rows[:, 0] * 1000, rows[:, 1], rows[:, 2]
    n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
    return (n_fn(wavelengths_nm) + 1j * k_fn(wavelengths_nm)) ** 2


# ── Si epsilon (substrate) ─────────────────────────────────────────────────────
def load_eps_si():
    with open(os.path.join(DB, "main/Si/nk/Aspnes.yml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        if blk["type"].strip() == "tabulated nk":
            rows = np.array([[float(v) for v in ln.split()]
                             for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
            wl_db, n_db, k_db = rows[:, 0] * 1000, rows[:, 1], rows[:, 2]
    n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
    return (n_fn(wavelengths_nm) + 1j * k_fn(wavelengths_nm)) ** 2


# ── Simulation (runs inside worker process) ───────────────────────────────────
def simulate_one(period_nm, pillar_nm, height_nm, eps_sio2, dev):
    p = period_nm / 1000.0
    w = pillar_nm / 1000.0
    h = height_nm / 1000.0

    si_cols = max(1, min(int(NX * w / p), NX - 1))
    pillar_mask = torch.zeros(NX, NX, dtype=torch.bool, device=dev)
    pillar_mask[:, :si_cols] = True

    results = []
    for i, wl in enumerate(wavelengths_um):
        try:
            ep   = complex(eps_sio2[i])
            ep_t = torch.tensor(ep, dtype=sim_dtype, device=dev)

            sim = torcwa.rcwa(freq=1.0/wl, order=[1, NG], L=[p, p],
                              dtype=sim_dtype, device=dev)
            sim.add_input_layer(eps=torch.tensor(1.0, dtype=sim_dtype, device=dev))
            sim.add_output_layer(eps=ep_t)
            sim.set_incident_angle(inc_ang=1e-6, azi_ang=0.0)

            ep_grid = torch.ones(NX, NX, dtype=sim_dtype, device=dev)
            ep_grid[pillar_mask] = ep_t
            sim.add_layer(thickness=h, eps=ep_grid)

            sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
            sim.solve_global_smatrix()

            R = sim.S_parameters(orders=[0, 0], direction="forward",
                                 port="reflection", polarization="xx", ref_order=[0, 0])
            results.append(float(abs(R) ** 2))
        except Exception:
            results.append(float("nan"))

    arr = np.array(results, dtype=np.float32)
    nans = np.isnan(arr)
    if nans.any() and (~nans).any():
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


# ── Worker process: initialised once per process ───────────────────────────────
_w_eps = None
_w_dev = None

def _worker_init():
    global _w_eps, _w_dev
    if DB is None:
        raise RuntimeError("refractiveindex database not found.")
    _w_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _w_eps = load_eps_sio2()


def _worker_fn(args):
    idx, period, pillar, height = args
    R = simulate_one(period, pillar, height, _w_eps, _w_dev)
    return idx, period, pillar, height, R


# ── Parameter sampling ────────────────────────────────────────────────────────
def sample_params(nrows, seed=42):
    rng = np.random.default_rng(seed)
    periods = rng.uniform(PERIOD_MIN, PERIOD_MAX, nrows)
    heights = rng.uniform(HEIGHT_MIN, HEIGHT_MAX, nrows)
    pillars = np.array([
        rng.uniform(max(PILLAR_MIN, 10), min(PILLAR_MAX, p - 10))
        for p in periods
    ])
    return periods, pillars, heights


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows",   type=int, default=5000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--test",    action="store_true")
    parser.add_argument("--out",     default=None)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    if DB is None:
        raise RuntimeError("refractiveindex database not found.")
    os.makedirs(OUTDIR, exist_ok=True)

    nrows   = 100 if args.test else args.nrows
    outfile = args.out or os.path.join(OUTDIR, f"rcwa_sio2_train_ng{NG}.csv")

    print(f"Workers   : {args.workers}")
    print(f"NG={NG}  NX={NX}  dtype=complex64  NUM_WL={NUM_WL}")
    print(f"Rows      : {nrows}")
    print(f"Output    : {outfile}\n")

    wl_cols = [f"R_{int(wl)}" for wl in wavelengths_nm]
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
    tasks = [(i, periods[i], pillars[i], heights[i]) for i in range(done, nrows)]

    t0 = time.time()
    count = 0

    with open(outfile, "a", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        if args.workers <= 1:
            _worker_init()
            with tqdm(total=len(tasks), desc="generate") as pbar:
                for t in tasks:
                    _, period, pillar, height, R = _worker_fn(t)
                    writer.writerow([period, pillar, height] + R.tolist())
                    count += 1
                    pbar.update(1)
                    if count % CHUNK == 0:
                        fout.flush()
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     initializer=_worker_init) as pool:
                futures = {pool.submit(_worker_fn, t): t[0] for t in tasks}
                with tqdm(total=len(tasks), desc="generate") as pbar:
                    for future in as_completed(futures):
                        idx, period, pillar, height, R = future.result()
                        writer.writerow([period, pillar, height] + R.tolist())
                        count += 1
                        pbar.update(1)
                        if count % CHUNK == 0:
                            fout.flush()
        fout.flush()

    elapsed = (time.time() - t0) / 60
    rate    = len(tasks) / max(elapsed, 0.01)
    print(f"\nDone: {nrows} rows in {elapsed:.1f}min  ({rate:.1f} rows/min)")
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
