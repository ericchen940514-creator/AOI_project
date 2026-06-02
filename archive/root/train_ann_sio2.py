"""
Train forward ANN: (period_nm, pillar_nm, height_nm) -> reflectance spectrum

Usage:
    python train_ann_sio2.py              # train / continue from checkpoint
    python train_ann_sio2.py --epochs 200
    python train_ann_sio2.py --csv path/to/data.csv
"""
import csv, os, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.signal import savgol_filter

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
OUTDIR    = os.path.join(BASE, "data_generation", "output")
CKPT_PATH = os.path.join(BASE, "ann_sio2.pth")
DEFAULT_CSV = os.path.join(OUTDIR, "rcwa_sio2_train_ng9.csv")

# ── Parameter normalisation bounds (must match generator) ─────────────────────
PERIOD_MIN, PERIOD_MAX = 600.0, 760.0
PILLAR_MIN, PILLAR_MAX = 150.0, 500.0
HEIGHT_MIN, HEIGHT_MAX = 100.0, 400.0


def normalise(period, pillar, height):
    p = (period - PERIOD_MIN) / (PERIOD_MAX - PERIOD_MIN)
    w = (pillar  - PILLAR_MIN) / (PILLAR_MAX - PILLAR_MIN)
    h = (height  - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN)
    return np.stack([p, w, h], axis=-1).astype(np.float32)


# ── Model ─────────────────────────────────────────────────────────────────────
class ForwardANN(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append([float(v) for v in row])
    arr = np.array(rows, dtype=np.float32)
    params = arr[:, :3]   # period, pillar, height
    spectra = arr[:, 3:]  # reflectance values
    n_out = spectra.shape[1]
    X = normalise(params[:, 0], params[:, 1], params[:, 2])
    return X, spectra, n_out


def smooth_labels(spectra, window=11, polyorder=3):
    """Apply Savitzky-Golay smoothing to each spectrum row."""
    return np.array([savgol_filter(row, window, polyorder) for row in spectra], dtype=np.float32)


# ── Train ─────────────────────────────────────────────────────────────────────
def train(csv_path, epochs, batch_size, lr, val_split, smooth_w=0.1, fresh=False, smooth_labels_flag=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, Y, n_out = load_csv(csv_path)
    if smooth_labels_flag:
        Y = smooth_labels(Y)
        print(f"Loaded {len(X)} samples  n_out={n_out}  (labels smoothed with savgol)")
    else:
        print(f"Loaded {len(X)} samples  n_out={n_out}")

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    ds  = TensorDataset(X_t, Y_t)

    n_val  = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)

    # Load checkpoint or init fresh
    model = ForwardANN(n_out).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-6)
    start_epoch = 0

    if fresh and os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
        print("Fresh start: removed old checkpoint")

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        if ckpt["n_out"] == n_out:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}  (prev val_mae={ckpt['val_mae']*100:.3f}%)")
        else:
            print(f"Checkpoint n_out mismatch ({ckpt['n_out']} vs {n_out}) — starting fresh")

    loss_fn = nn.MSELoss()
    best_val_mae = float("inf")
    t0 = time.time()

    for epoch in range(start_epoch, start_epoch + epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            mse  = loss_fn(pred, yb)
            # Smoothness penalty: penalise large jumps between adjacent wavelengths
            smooth = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)
            loss   = mse + smooth_w * smooth
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += mse.item() * len(xb)
        train_loss /= n_train

        # Validate
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_mae += (pred - yb).abs().mean().item() * len(xb)
        val_mae /= n_val
        sched.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "model": model.state_dict(),
                "opt":   opt.state_dict(),
                "epoch": epoch,
                "n_out": n_out,
                "val_mae": val_mae,
                "wl_min": 400.0, "wl_max": 750.0, "n_wl": n_out,
            }, CKPT_PATH)

        if (epoch - start_epoch) % 10 == 0 or epoch == start_epoch + epochs - 1:
            elapsed = (time.time() - t0) / 60
            lr_now  = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}  train_mse={train_loss:.5f}  "
                  f"val_mae={val_mae*100:.3f}%  lr={lr_now:.2e}  {elapsed:.1f}min")

    print(f"\nBest val MAE: {best_val_mae*100:.4f}%")
    print(f"Checkpoint saved: {CKPT_PATH}")
    return best_val_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default=DEFAULT_CSV)
    parser.add_argument("--epochs",  type=int,   default=300)
    parser.add_argument("--batch",   type=int,   default=256)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--val",     type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--smooth",        type=float, default=0.1, help="Smoothness loss weight")
    parser.add_argument("--fresh",         action="store_true", help="Delete checkpoint and start fresh")
    parser.add_argument("--smooth-labels", action="store_true", help="Pre-smooth training labels with Savitzky-Golay")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        print("Run generate_torcwa_sio2.py first.")
        return

    best_mae = train(args.csv, args.epochs, args.batch, args.lr, args.val,
                     smooth_w=args.smooth, fresh=args.fresh,
                     smooth_labels_flag=args.smooth_labels)
    print("\n--- Result ---")
    print(f"Val MAE: {best_mae*100:.4f}%  ({'PASS <1%' if best_mae < 0.01 else 'FAIL - generate more data'})")


if __name__ == "__main__":
    main()
