"""
High-precision torcwa simulation of Sample A and Sample B (Mattila 2024).

Sample A : v1_A_256.h5  key d680_el300  → period=680  pillar=309.1  h=209.7 nm
Sample B : v1_B_256.h5  key d675_el330  → period=675  pillar=351.1  h=282.6 nm

Both keys are fixed — no nearest-neighbour search.

GPU parallelism: each sample's 256 wavelengths are split into WL_CHUNKS chunks.
Total tasks = N_SAMPLES × WL_CHUNKS, run by ProcessPoolExecutor(N_WORKERS).
This keeps the GPU fed continuously (same pattern as run_parallel_gen.ps1).

Usage:
    python validation/simulate_samples_AB.py
"""

import io, os, time
import numpy as np
import yaml
import h5py
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────────────────────
NG        = 51       # harmonic orders; NX >= 2*NG+1 = 103
NX        = 105
WL_CHUNKS = 4        # split 256 wavelengths into 4 chunks per sample
N_WORKERS = 4        # parallel processes → 4 GPU work streams

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

_DB_CANDIDATES = [
    "/mnt/c/Users/user/.refractiveindex.info-database/data",
    os.path.join(os.path.expanduser("~"), ".refractiveindex.info-database", "data"),
]
DB = next((p for p in _DB_CANDIDATES if os.path.isdir(p)), None)


# ── Worker: runs inside a subprocess ──────────────────────────────────────────
_w_dev  = None
_w_eps  = None   # (N_wl,) complex128 array loaded once per worker

def _worker_init():
    import torch, torcwa, yaml, io
    import numpy as np
    from scipy.interpolate import interp1d
    global _w_dev, _w_eps
    _w_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _worker_fn(args):
    """Simulate a wavelength chunk for one sample. Returns (task_id, R_chunk)."""
    import torch, torcwa, yaml, io
    import numpy as np
    from scipy.interpolate import interp1d

    task_id, period_nm, pillar_nm, height_nm, wl_chunk_nm = args

    dev       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_dtype = torch.complex64   # fp32 is fine on RTX; fp64 throughput is 32× slower

    # Load SiO2 eps for this chunk's wavelengths
    with open(os.path.join(DB, "main/SiO2/nk/Franta.yml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for blk in data["DATA"]:
        rows = np.array([[float(v) for v in ln.split()]
                         for ln in io.StringIO(blk["data"]).readlines() if ln.strip()])
        wl_db, n_db, k_db = rows[:, 0] * 1000, rows[:, 1], rows[:, 2]
    n_fn = interp1d(wl_db, n_db, kind="cubic", fill_value="extrapolate")
    k_fn = interp1d(wl_db, k_db, kind="cubic", fill_value="extrapolate")
    eps_chunk = (n_fn(wl_chunk_nm) + 1j * k_fn(wl_chunk_nm)) ** 2

    p   = period_nm / 1000.0
    w   = pillar_nm / 1000.0
    h   = height_nm / 1000.0
    wl_um = wl_chunk_nm / 1000.0

    sio2_cols = int(NX * (w / p))

    results = []
    for i, wl in enumerate(wl_um):
        ep = complex(eps_chunk[i])
        try:
            sim = torcwa.rcwa(
                freq=1.0 / wl,
                order=[1, NG],
                L=[p, p],
                dtype=sim_dtype,
                device=dev,
            )
            sim.add_input_layer(eps=torch.tensor(1.0,  dtype=sim_dtype, device=dev))
            sim.add_output_layer(eps=torch.tensor(ep,   dtype=sim_dtype, device=dev))
            sim.set_incident_angle(inc_ang=1e-6, azi_ang=0.0)

            ep_grid = torch.ones(NX, NX, dtype=sim_dtype, device=dev)
            ep_grid[:, :sio2_cols] = ep

            sim.add_layer(thickness=h, eps=ep_grid)
            sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
            sim.solve_global_smatrix()

            R = sim.S_parameters(
                orders=[0, 0], direction="forward",
                port="reflection", polarization="xx", ref_order=[0, 0],
            )
            results.append(float(abs(R) ** 2))
        except Exception:
            results.append(float("nan"))

    arr = np.array(results, dtype=np.float32)
    nans = np.isnan(arr)
    if nans.any():
        x = np.arange(len(arr))
        arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return task_id, arr


# ── Load samples ───────────────────────────────────────────────────────────────
def load_sample(h5_path, key):
    with h5py.File(h5_path, "r") as f:
        wl_nm  = f["lambda"][:].flatten()
        params = f["params"][key][0]
        eta_r  = f["eta_r"][key][0]
    period = params[0]
    height = params[1]
    groove = params[2]
    pillar = period - groove
    return {"period": period, "pillar": pillar, "height": height,
            "wl_nm": wl_nm, "eta_r_meas": eta_r}


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if DB is None:
        raise RuntimeError("refractiveindex database not found.")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"NG={NG}  NX={NX}  WL_CHUNKS={WL_CHUNKS}  N_WORKERS={N_WORKERS}\n")

    sa = load_sample(os.path.join(BASE, "v1_A_256.h5"), "d680_el300")
    sb = load_sample(os.path.join(BASE, "v1_B_256.h5"), "d675_el330")

    samples = [
        {"label": f"Sample A  d680_el300  p={sa['period']:.0f}  ridge={sa['pillar']:.1f}  h={sa['height']:.1f} nm", **sa},
        {"label": f"Sample B  d675_el330  p={sb['period']:.0f}  ridge={sb['pillar']:.1f}  h={sb['height']:.1f} nm", **sb},
    ]

    for s in samples:
        print(f"  {s['label']}")

    # Build tasks: (task_id, period, pillar, height, wl_chunk_nm)
    tasks = {}   # task_id → (sample_idx, wl_indices)
    task_list = []
    task_id = 0
    for si, s in enumerate(samples):
        wl_nm = s["wl_nm"]
        chunk_indices = np.array_split(np.arange(len(wl_nm)), WL_CHUNKS)
        for ci, idx in enumerate(chunk_indices):
            tasks[task_id] = (si, idx)
            task_list.append((task_id, s["period"], s["pillar"], s["height"], wl_nm[idx]))
            task_id += 1

    print(f"\nTotal tasks: {len(task_list)}  ({WL_CHUNKS} chunks × {len(samples)} samples)")
    print(f"Running with {N_WORKERS} parallel workers ...\n")

    # Collect into per-sample buffers
    R_buffers = [np.full(len(s["wl_nm"]), np.nan, dtype=np.float32) for s in samples]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as pool:
        futures = {pool.submit(_worker_fn, t): t[0] for t in task_list}
        done = 0
        for future in as_completed(futures):
            tid, R_chunk = future.result()
            si, idx      = tasks[tid]
            R_buffers[si][idx] = R_chunk
            done += 1
            print(f"  [{done}/{len(task_list)}] sample={si}  chunk wl={samples[si]['wl_nm'][idx[0]]:.0f}-{samples[si]['wl_nm'][idx[-1]]:.0f} nm  done")

    dt = time.perf_counter() - t0
    print(f"\nAll done in {dt:.1f}s\n")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    maes = []
    for i, s in enumerate(samples):
        wl_nm    = s["wl_nm"]
        R_sim    = R_buffers[i]
        eta_meas = s["eta_r_meas"]
        mae      = float(np.abs(R_sim - eta_meas).mean() * 100)
        maes.append(mae)
        print(f"{s['label']}  MAE={mae:.4f}%")

        ax0, ax1 = axes[i]
        ax0.plot(wl_nm, eta_meas * 100, "C1-",  lw=2, label="Measured (Mattila 2024)")
        ax0.plot(wl_nm, R_sim    * 100, "C0--", lw=2, label=f"torcwa NG={NG}")
        ax0.set_xlabel("Wavelength (nm)")
        ax0.set_ylabel("Reflectance (%)")
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
    out = os.path.join(BASE, "simulate_samples_AB.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    print("\n=== SUMMARY ===")
    for s, mae in zip(samples, maes):
        print(f"  {s['label']}  MAE={mae:.4f}%  [{'PASS' if mae < 2.0 else 'CHECK'}]")
