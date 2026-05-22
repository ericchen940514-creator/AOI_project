"""Generate 5000 RCWA samples: grcwa NG=101, NX=160, 8 cores.
Checkpoints every 100 rows — safe to kill and resume anytime.
"""
import sys, subprocess
for pkg in ["threadpoolctl", "tqdm", "grcwa"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

import os, time, multiprocessing as mp
import pandas as pd
from tqdm import tqdm

BASE = "/mnt/c/Users/user/OneDrive/Desktop/AOI_project"
os.chdir(BASE)
sys.path.insert(0, BASE)

from data_generation.generate import (_worker, _init_worker, _precompute_epsilon, NUM_WL)

PARAMS  = f"{BASE}/data_generation/output/rcwa_params_train.csv"
OUTFILE = f"{BASE}/data_generation/output/rcwa_Si_train_ng101_5000.csv"
NROWS   = 5000
CORES   = min(mp.cpu_count() - 1, 8)
CHUNK   = 100

cols = ["w", "h", "p"] + [f"R_wl_{i}" for i in range(NUM_WL)]

# ── Resume: find how many rows already written ─────────────────────────────────
if os.path.exists(OUTFILE):
    done_count = sum(1 for _ in open(OUTFILE)) - 1  # subtract header
    done_count = max(done_count, 0)
    print(f"Resuming from row {done_count}", flush=True)
else:
    done_count = 0
    pd.DataFrame(columns=cols).to_csv(OUTFILE, index=False)  # write header only

df_params = pd.read_csv(PARAMS).head(NROWS)
remaining = df_params.iloc[done_count:]
tasks = [(row.Index, row.w, row.h, row.p) for row in remaining.itertuples()]

if not tasks:
    print("Already complete.", flush=True)
    sys.exit(0)

print(f"NG=101  NX=160  {len(tasks)} remaining / {NROWS} total  {CORES} cores", flush=True)

ep = _precompute_epsilon("Si")
t0 = time.time()
buffer = []

with mp.Pool(processes=CORES, initializer=_init_worker, initargs=(ep,)) as pool:
    with tqdm(total=len(tasks), desc="RCWA") as pbar:
        for r in pool.imap(_worker, tasks):   # imap keeps order → safe resume
            buffer.append(r[1:])              # drop id
            pbar.update(1)
            if len(buffer) >= CHUNK:
                pd.DataFrame(buffer, columns=cols).to_csv(
                    OUTFILE, mode="a", header=False, index=False)
                buffer.clear()
        if buffer:
            pd.DataFrame(buffer, columns=cols).to_csv(
                OUTFILE, mode="a", header=False, index=False)

dt = (time.time() - t0) / 60
total_done = done_count + len(tasks)
print(f"\nDone: {total_done} rows in {dt:.1f} min → {OUTFILE}", flush=True)
