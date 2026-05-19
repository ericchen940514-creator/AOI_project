"""
ANN Inverse — gradient-based optimisation: spectrum → (w, h, p)
Modes:
  demo      : generate target from DEMO_PARAMS, then invert
  csv       : pick a row from rcwa_spectra_TEST.csv
Run: python ann/inverse.py
"""
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_PATH = "ann/model.pth"
SCALER_PATH = "ann/scaler.pkl"
DATA_PATH = "data/rcwa_spectra_pretrain.csv"
TEST_PATH = "data/rcwa_spectra_TEST.csv"

PARAM_BOUNDS = {"w": (0.05, 0.25), "h": (0.10, 0.50), "p": (0.20, 0.60)}
PARAM_NAMES = ["w", "h", "p"]
WAVELENGTHS = np.linspace(400, 700, 100)

DEMO_PARAMS = {"w": 0.150, "h": 0.250, "p": 0.400}
NUM_RESTARTS = 8
OPTIM_STEPS = 1500
LR = 0.03
FF_RANGE = (0.30, 0.70)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 100), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, device


def _inv_sigmoid(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def _to_logits(params):
    return np.array([
        _inv_sigmoid((params[i] - PARAM_BOUNDS[n][0]) / (PARAM_BOUNDS[n][1] - PARAM_BOUNDS[n][0]))
        for i, n in enumerate(PARAM_NAMES)
    ], dtype=np.float32)


def _from_logits(logits):
    return torch.stack([
        PARAM_BOUNDS[n][0] + (PARAM_BOUNDS[n][1] - PARAM_BOUNDS[n][0]) * torch.sigmoid(logits[i])
        for i, n in enumerate(PARAM_NAMES)
    ])


def _init_candidates(target):
    df = pd.read_csv(DATA_PATH)
    spectra = df.loc[:, "R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)
    params = df[PARAM_NAMES].to_numpy(dtype=np.float32)
    dists = np.mean((spectra - target[None, :]) ** 2, axis=1)
    best = params[np.argsort(dists)[:NUM_RESTARTS]].copy()
    rng = np.random.default_rng(42)
    for i in range(1, len(best)):
        best[i] += rng.normal(0, [0.01, 0.02, 0.02])
        for j, n in enumerate(PARAM_NAMES):
            best[i, j] = np.clip(best[i, j], *PARAM_BOUNDS[n])
    return best


def inverse(model, scaler, device, target_spectrum):
    target_t = torch.tensor(target_spectrum, dtype=torch.float32, device=device).unsqueeze(0)
    s_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    s_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    candidates = _init_candidates(target_spectrum)
    best = None

    for init_p in candidates:
        logits = torch.tensor(_to_logits(init_p), device=device, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=LR)
        best_loss, best_p, best_s = float("inf"), None, None

        for _ in range(OPTIM_STEPS):
            opt.zero_grad()
            p = _from_logits(logits)
            ff = p[0] / p[2]
            ff_pen = torch.relu(FF_RANGE[0] - ff) ** 2 + torch.relu(ff - FF_RANGE[1]) ** 2
            x = ((p - s_mean) / s_scale).unsqueeze(0)
            pred = model(x)
            loss = torch.mean((pred - target_t) ** 2) + 0.1 * ff_pen
            loss.backward()
            opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_p = p.detach().cpu().numpy().copy()
                best_s = pred.detach().cpu().numpy()[0].copy()

        if best is None or best_loss < best["loss"]:
            best = {"loss": best_loss, "params": best_p, "spectrum": best_s}

    return best


def plot(target, recovered, true_p, est_p):
    plt.figure(figsize=(9, 5))
    plt.plot(WAVELENGTHS, target, color="black", linewidth=2, label="Target")
    plt.plot(WAVELENGTHS, recovered, color="tab:red", linewidth=2, linestyle="--", label="Recovered")
    err = np.abs(est_p - true_p) * 1000 if true_p is not None else None
    title = "ANN Inverse Prediction\n"
    if true_p is not None:
        title += f"True:  w={true_p[0]*1000:.1f}  h={true_p[1]*1000:.1f}  p={true_p[2]*1000:.1f} nm\n"
    title += f"Est:   w={est_p[0]*1000:.1f}  h={est_p[1]*1000:.1f}  p={est_p[2]*1000:.1f} nm"
    if err is not None:
        title += f"\nErr:   dw={err[0]:.1f}  dh={err[1]:.1f}  dp={err[2]:.1f} nm"
    plt.title(title, fontsize=11)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ann/inverse_result.png", dpi=200)
    print("Saved: ann/inverse_result.png")
    plt.show()


def main():
    print("Loading ANN model...")
    model, scaler, device = load()
    print(f"Device: {device}")

    mode = input("\nMode — [d]emo / [c]sv row (default: d): ").strip().lower() or "d"

    if mode == "c":
        df = pd.read_csv(TEST_PATH)
        print(f"TEST CSV has {len(df)} rows.")
        idx = int(input("Row index: ").strip())
        row = df.iloc[idx]
        true_p = row[PARAM_NAMES].to_numpy(dtype=np.float32)
        target = row.loc["R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)
        print(f"True params: w={true_p[0]:.4f}  h={true_p[1]:.4f}  p={true_p[2]:.4f} um")
    else:
        true_p = np.array([DEMO_PARAMS[n] for n in PARAM_NAMES], dtype=np.float32)
        x = torch.tensor(scaler.transform([true_p]), dtype=torch.float32).to(device)
        with torch.no_grad():
            target = model(x).cpu().numpy()[0]
        print(f"Demo params: w={true_p[0]:.4f}  h={true_p[1]:.4f}  p={true_p[2]:.4f} um")

    print("Running inverse optimisation...")
    t0 = time.perf_counter()
    result = inverse(model, scaler, device, target)
    ms = (time.perf_counter() - t0) * 1000

    est = result["params"]
    print(f"Done in {ms:.0f} ms  |  loss={result['loss']:.8f}")
    print(f"Estimated: w={est[0]:.4f}  h={est[1]:.4f}  p={est[2]:.4f} um")
    if true_p is not None:
        err = np.abs(est - true_p) * 1000
        print(f"Error:     dw={err[0]:.2f}  dh={err[1]:.2f}  dp={err[2]:.2f} nm")

    plot(target, result["spectrum"], true_p, est)


if __name__ == "__main__":
    main()
