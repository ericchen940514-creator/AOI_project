"""
SiO2 Grating AOI Pipeline
=========================
Step 0: Generate (period, pillar, height) parameters via Latin Hypercube Sampling
Step 1: Run torcwa RCWA simulation on those parameters → spectra CSV
Step 2: Train forward ANN  (period, pillar, height) → reflectance spectrum

Usage:
    python pipeline.py                          # full pipeline, 5000 rows, 300 epochs
    python pipeline.py --nrows 20000            # larger dataset
    python pipeline.py --epochs 500             # more training
    python pipeline.py --workers 4              # more parallel torcwa workers
    python pipeline.py --skip-params            # skip param generation, use existing CSV
    python pipeline.py --skip-sim               # skip simulation, train on existing CSV
    python pipeline.py --skip-train             # only generate params + simulate
    python pipeline.py --fresh                  # fresh ANN training (delete checkpoint)
"""
import argparse, os, subprocess, sys

BASE        = os.path.dirname(os.path.abspath(__file__))
PARAMS_SCRIPT   = os.path.join(BASE, "data_generation", "params.py")
GENERATE_SCRIPT = os.path.join(BASE, "sio2", "generate_torcwa_sio2.py")
TRAIN_SCRIPT    = os.path.join(BASE, "train_ann_sio2.py")

DEFAULT_PARAMS_CSV = os.path.join(BASE, "data_generation", "output", "sio2_params.csv")
DEFAULT_SPECTRA_CSV = os.path.join(BASE, "data_generation", "output", "rcwa_sio2_train_ng9.csv")


def run(cmd):
    print(f"\n{'='*60}")
    print(f"$ {' '.join(cmd)}")
    print('='*60)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="SiO2 grating AOI pipeline")
    parser.add_argument("--nrows",       type=int,   default=5000)
    parser.add_argument("--workers",     type=int,   default=3)
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--batch",       type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--smooth",      type=float, default=0.1)
    parser.add_argument("--params-csv",  default=DEFAULT_PARAMS_CSV)
    parser.add_argument("--spectra-csv", default=DEFAULT_SPECTRA_CSV)
    parser.add_argument("--skip-params", action="store_true", help="Skip LHS param generation")
    parser.add_argument("--skip-sim",    action="store_true", help="Skip torcwa simulation")
    parser.add_argument("--skip-train",  action="store_true", help="Skip ANN training")
    parser.add_argument("--fresh",       action="store_true", help="Delete ANN checkpoint and retrain")
    args = parser.parse_args()

    # ── Step 0: Generate (period, pillar, height) via LHS ─────────────────────
    if not args.skip_params:
        run([sys.executable, PARAMS_SCRIPT,
             "--nrows", str(args.nrows),
             "--out",   args.params_csv])
    else:
        print("\n[Step 0] Skipped — using existing params CSV")

    # ── Step 1: torcwa RCWA simulation ────────────────────────────────────────
    if not args.skip_sim:
        sim_cmd = [sys.executable, GENERATE_SCRIPT,
                   "--workers", str(args.workers),
                   "--out",     args.spectra_csv]
        if os.path.exists(args.params_csv):
            sim_cmd += ["--params", args.params_csv]
        else:
            sim_cmd += ["--nrows", str(args.nrows)]
        run(sim_cmd)
    else:
        print("\n[Step 1] Skipped — using existing spectra CSV")

    # ── Step 2: Train ANN ─────────────────────────────────────────────────────
    if not args.skip_train:
        if not os.path.exists(args.spectra_csv):
            print(f"\nERROR: spectra CSV not found: {args.spectra_csv}")
            sys.exit(1)
        n_rows = sum(1 for _ in open(args.spectra_csv)) - 1
        print(f"\n[Step 2] Training ANN on {n_rows} rows")
        train_cmd = [sys.executable, TRAIN_SCRIPT,
                     "--csv",    args.spectra_csv,
                     "--epochs", str(args.epochs),
                     "--batch",  str(args.batch),
                     "--lr",     str(args.lr),
                     "--smooth", str(args.smooth)]
        if args.fresh:
            train_cmd.append("--fresh")
        run(train_cmd)
    else:
        print("\n[Step 2] Skipped")

    print("\n" + "="*60)
    print("Pipeline complete.")
    print(f"  Params  : {args.params_csv}")
    print(f"  Spectra : {args.spectra_csv}")
    print(f"  Model   : {os.path.join(BASE, 'ann', 'ann_sio2_ng9.pth')}")
    print("="*60)


if __name__ == "__main__":
    main()
