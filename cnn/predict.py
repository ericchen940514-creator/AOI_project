"""
CNN Inverse Prediction — spectrum → (w, h, p), instant inference
Modes:
  demo  : generate spectrum with ANN surrogate from demo params, then invert with CNN
  csv   : pick a row from rcwa_spectra_TEST.csv
Run: python cnn/predict.py
"""
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CNN_MODEL_PATH  = "cnn/model.pth"
PARAM_SCALER    = "cnn/param_scaler.pkl"
ANN_MODEL_PATH  = "ann/model.pth"
ANN_SCALER_PATH = "ann/scaler.pkl"
TEST_PATH       = "data/rcwa_spectra_TEST.csv"

PARAM_NAMES  = ["w", "h", "p"]
PARAM_BOUNDS = {"w": (0.05, 0.25), "h": (0.10, 0.50), "p": (0.20, 0.60)}
WAVELENGTHS  = np.linspace(400, 700, 100)
DEMO_PARAMS  = {"w": 0.150, "h": 0.250, "p": 0.400}


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=0), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=0), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=0), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 3),    nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 100), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_cnn(device):
    model = CNN1D().to(device)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    scaler = joblib.load(PARAM_SCALER)
    return model, scaler


def load_ann(device):
    model = ANN().to(device)
    model.load_state_dict(torch.load(ANN_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    scaler = joblib.load(ANN_SCALER_PATH)
    return model, scaler


def cnn_predict(cnn, param_scaler, device, spectrum):
    x = torch.tensor(spectrum[None, None, :], dtype=torch.float32).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = cnn(x).cpu().numpy()
    ms = (time.perf_counter() - t0) * 1000
    params = param_scaler.inverse_transform(out)[0]
    return params, ms


def plot(spectrum, true_p, est_p):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: spectrum
    axes[0].plot(WAVELENGTHS, spectrum, color="black", linewidth=2)
    axes[0].set_title("Input Spectrum")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Reflectance")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # right: param bar comparison
    labels = ["w (nm)", "h (nm)", "p (nm)"]
    est_nm = est_p * 1000
    x = np.arange(3)
    axes[1].bar(x - 0.2, est_nm, width=0.35, label="CNN Estimated", color="tab:blue")
    if true_p is not None:
        true_nm = true_p * 1000
        axes[1].bar(x + 0.2, true_nm, width=0.35, label="True", color="tab:orange")
        err = np.abs(est_p - true_p) * 1000
        axes[1].set_title(
            f"CNN Inverse Result\n"
            f"Est:  w={est_nm[0]:.1f}  h={est_nm[1]:.1f}  p={est_nm[2]:.1f} nm\n"
            f"Err:  dw={err[0]:.1f}  dh={err[1]:.1f}  dp={err[2]:.1f} nm",
            fontsize=10,
        )
    else:
        axes[1].set_title(
            f"CNN Inverse Result\n"
            f"Est:  w={est_nm[0]:.1f}  h={est_nm[1]:.1f}  p={est_nm[2]:.1f} nm",
            fontsize=11,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Value (nm)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cnn/inverse_result.png", dpi=200)
    print("Saved: cnn/inverse_result.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading CNN model...")
    cnn, param_scaler = load_cnn(device)

    mode = input("\nMode — [d]emo / [c]sv row (default: d): ").strip().lower() or "d"

    if mode == "c":
        df = pd.read_csv(TEST_PATH)
        print(f"TEST CSV has {len(df)} rows.")
        idx = int(input("Row index: ").strip())
        row = df.iloc[idx]
        true_p = row[PARAM_NAMES].to_numpy(dtype=np.float32)
        spectrum = row.loc["R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)
        print(f"True params: w={true_p[0]:.4f}  h={true_p[1]:.4f}  p={true_p[2]:.4f} um")
    else:
        print("Loading ANN surrogate to generate demo spectrum...")
        ann, ann_scaler = load_ann(device)
        true_p = np.array([DEMO_PARAMS[n] for n in PARAM_NAMES], dtype=np.float32)
        x = torch.tensor(ann_scaler.transform([true_p]), dtype=torch.float32).to(device)
        with torch.no_grad():
            spectrum = ann(x).cpu().numpy()[0]
        print(f"Demo params: w={true_p[0]:.4f}  h={true_p[1]:.4f}  p={true_p[2]:.4f} um")

    est_p, ms = cnn_predict(cnn, param_scaler, device, spectrum)
    print(f"\nCNN inference done in {ms:.3f} ms")
    print(f"Estimated: w={est_p[0]:.4f}  h={est_p[1]:.4f}  p={est_p[2]:.4f} um")
    if true_p is not None:
        err = np.abs(est_p - true_p) * 1000
        print(f"Error:     dw={err[0]:.2f}  dh={err[1]:.2f}  dp={err[2]:.2f} nm")

    plot(spectrum, true_p, est_p)


if __name__ == "__main__":
    main()
