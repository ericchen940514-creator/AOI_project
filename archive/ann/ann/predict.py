"""
ANN Forward Prediction — interactive: input (w, h, p) → plot spectrum
Run: python ann/predict.py
"""
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "ann/model.pth"
SCALER_PATH = "ann/scaler.pkl"
PARAM_BOUNDS = {"w": (0.05, 0.25), "h": (0.10, 0.50), "p": (0.20, 0.60)}
WAVELENGTHS = np.linspace(400, 700, 100)


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


def predict(model, scaler, device, w, h, p):
    x = torch.tensor(scaler.transform([[w, h, p]]), dtype=torch.float32).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        spectrum = model(x).cpu().numpy()[0]
    return spectrum, (time.perf_counter() - t0) * 1000


def read_float(prompt, lo, hi):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "q":
            return None
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
            print(f"    out of range ({lo:.3f}~{hi:.3f} um)")
        except ValueError:
            print("    please enter a number")


def main():
    print("Loading ANN model...")
    model, scaler, device = load()
    print(f"Device: {device}\n")

    while True:
        print("=== ANN Forward Prediction ===")
        w = read_float(f"  w (50~250 nm) [um]: ", *PARAM_BOUNDS["w"])
        if w is None: break
        h = read_float(f"  h (100~500 nm) [um]: ", *PARAM_BOUNDS["h"])
        if h is None: break
        p = read_float(f"  p (200~600 nm) [um]: ", *PARAM_BOUNDS["p"])
        if p is None: break

        spectrum, ms = predict(model, scaler, device, w, h, p)
        print(f"  Done in {ms:.2f} ms\n")

        plt.figure(figsize=(8, 5))
        plt.plot(WAVELENGTHS, spectrum, color="b", linewidth=2, label="ANN Predicted R(λ)")
        plt.title(f"ANN Forward\n(w={w*1000:.0f}nm, h={h*1000:.0f}nm, p={p*1000:.0f}nm)", fontsize=14)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        save = f"ann/forward_w{w*1000:.0f}_h{h*1000:.0f}_p{p*1000:.0f}.png"
        plt.savefig(save, dpi=200)
        print(f"  Saved: {save}\n")
        plt.show()


if __name__ == "__main__":
    main()
