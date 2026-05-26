"""
Evaluate trained ANN against Mattila 2024 measured spectra.
Reports MAE for Sample A and B — use this to decide whether to continue training.

Usage:
    python eval_ann_sio2.py
"""
import os
import numpy as np
import torch
import h5py
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE      = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE, "ann_sio2.pth")

PERIOD_MIN, PERIOD_MAX = 600.0, 760.0
PILLAR_MIN, PILLAR_MAX = 150.0, 500.0
HEIGHT_MIN, HEIGHT_MAX = 100.0, 400.0


def normalise(period, pillar, height):
    return torch.tensor([
        (period - PERIOD_MIN) / (PERIOD_MAX - PERIOD_MIN),
        (pillar  - PILLAR_MIN) / (PILLAR_MAX - PILLAR_MIN),
        (height  - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN),
    ], dtype=torch.float32)


class ForwardANN(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU(),
            torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU(),
            torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU(),
            torch.nn.Linear(512, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU(),
            torch.nn.Linear(512, n_out),
        )
    def forward(self, x):
        return self.net(x)


def predict(model, period_nm, pillar_nm, height_nm, wl_ann, wl_target):
    """Run ANN, smooth output, interpolate to target wavelength grid."""
    model.eval()
    x = normalise(period_nm, pillar_nm, height_nm).unsqueeze(0)
    with torch.no_grad():
        R_ann = model(x).squeeze(0).cpu().numpy()
    R_ann = np.clip(R_ann, 0, 1)
    R_ann = savgol_filter(R_ann, window_length=15, polyorder=3)
    R_ann = np.clip(R_ann, 0, 1)
    interp = interp1d(wl_ann, R_ann, kind="cubic", fill_value="extrapolate")
    return interp(wl_target)


def main():
    if not os.path.exists(CKPT_PATH):
        print(f"No checkpoint found at {CKPT_PATH}")
        print("Run train_ann_sio2.py first.")
        return

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    n_wl    = ckpt["n_wl"]
    wl_min  = ckpt.get("wl_min", 400.0)
    wl_max  = ckpt.get("wl_max", 750.0)
    wl_ann  = np.linspace(wl_min, wl_max, n_wl)

    model = ForwardANN(n_wl)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Model loaded  (epoch={ckpt['epoch']}  val_mae={ckpt['val_mae']*100:.4f}%)")
    print(f"ANN wavelengths: {n_wl} pts  {wl_min:.0f}-{wl_max:.0f}nm\n")

    samples = []

    h5_A = next((p for p in [
        os.path.join(BASE, "v1_A_256.h5"),
        os.path.join(BASE, "validation", "v1_A_256.h5"),
    ] if os.path.exists(p)), None)
    if h5_A:
        with h5py.File(h5_A, "r") as f:
            wl_meas = f["lambda"][:].flatten()
            params  = f["params"]["d680_el300"][0]
            eta_r   = f["eta_r"]["d680_el300"][0]
        period, height, groove = params[0], params[1], params[2]
        pillar = period - groove
        samples.append({"label": f"Sample A  p={period:.0f} ridge={pillar:.1f} h={height:.1f}nm",
                        "period": period, "pillar": pillar, "height": height,
                        "wl_meas": wl_meas, "eta_r": eta_r})

    h5_B = next((p for p in [
        os.path.join(BASE, "v1_B_256.h5"),
        os.path.join(BASE, "validation", "v1_B_256.h5"),
    ] if os.path.exists(p)), None)
    if h5_B:
        with h5py.File(h5_B, "r") as f:
            wl_meas_B = f["lambda"][:].flatten()
            # Same best-match key as validate_mattila2024.py
            best_key, best_err = None, 1e9
            for key in f["eta_r"].keys():
                p = f["params"][key][0]
                err = abs(p[0]-675) + abs((p[0]-p[2])-351) + abs(p[1]-283)
                if err < best_err:
                    best_err, best_key = err, key
            params_b = f["params"][best_key][0]
            eta_r_b  = f["eta_r"][best_key][0]
        period_b = params_b[0]; pillar_b = period_b - params_b[2]; height_b = params_b[1]
        samples.append({"label": f"Sample B  p={period_b:.0f} ridge={pillar_b:.1f} h={height_b:.1f}nm",
                        "period": period_b, "pillar": pillar_b, "height": height_b,
                        "wl_meas": wl_meas_B, "eta_r": eta_r_b})

    if not samples:
        print("No h5 files found. Place v1_A_256.h5 / v1_B_256.h5 in project root.")
        return

    fig, axes = plt.subplots(len(samples), 2, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    all_pass = True
    print(f"{'Sample':<55}  {'MAE':>7}  Status")
    print("-" * 72)
    for i, s in enumerate(samples):
        R_pred = predict(model, s["period"], s["pillar"], s["height"], wl_ann, s["wl_meas"])
        mae    = np.abs(R_pred - s["eta_r"]).mean() * 100
        status = "PASS" if mae < 1.0 else "FAIL"
        if mae >= 1.0:
            all_pass = False
        print(f"{s['label']:<55}  {mae:>6.4f}%  {status}")

        ax0, ax1 = axes[i]
        ax0.plot(s["wl_meas"], s["eta_r"] * 100, "C1-",  lw=2, label="Measured")
        ax0.plot(s["wl_meas"], R_pred     * 100, "C0--", lw=2, label="ANN")
        ax0.set_xlabel("Wavelength (nm)"); ax0.set_ylabel("Reflectance (%)")
        ax0.set_title(s["label"]); ax0.legend(); ax0.grid(True, ls="--", alpha=0.4)

        diff = (R_pred - s["eta_r"]) * 100
        ax1.plot(s["wl_meas"], diff, "C2-", lw=1.5)
        ax1.axhline(0, color="k", lw=0.8)
        ax1.fill_between(s["wl_meas"], diff, alpha=0.3, color="C2")
        ax1.set_xlabel("Wavelength (nm)"); ax1.set_ylabel("ANN − Measured (%)")
        ax1.set_title(f"MAE = {mae:.4f}%"); ax1.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(BASE, "eval_ann_sio2.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved: {out}")
    print("\n" + ("ALL PASS — training complete!" if all_pass else
                  "FAIL — generate more data and run train_ann_sio2.py again"))


if __name__ == "__main__":
    main()
