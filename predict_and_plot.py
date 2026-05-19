import torch
import torch.nn as nn
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time


class OCD_Surrogate_Model(nn.Module):
    def __init__(self):
        super(OCD_Surrogate_Model, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


PARAM_BOUNDS = {"w": (0.05, 0.25), "h": (0.10, 0.50), "p": (0.20, 0.60)}
WAVELENGTHS = np.linspace(400, 700, 100)


def load_model_and_scaler():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCD_Surrogate_Model().to(device)
    model.load_state_dict(torch.load("ocd_surrogate_pretrained.pth", map_location=device, weights_only=True))
    model.eval()
    scaler = joblib.load("scaler.pkl")
    return model, scaler, device


def predict_spectrum(model, scaler, device, w, h, p):
    input_data = np.array([[w, h, p]], dtype=np.float32)
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        spectrum = model(input_tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return spectrum.cpu().numpy()[0], elapsed_ms


def prompt_param(name, low, high):
    unit = "um"
    while True:
        raw = input(f"  請輸入 {name} ({low*1000:.0f}~{high*1000:.0f} nm) [{unit}]: ").strip()
        try:
            val = float(raw)
            if low <= val <= high:
                return val
            print(f"    超出範圍，請輸入 {low:.3f} ~ {high:.3f} {unit}")
        except ValueError:
            print("    請輸入數字")


def plot_spectrum(spectrum, w, h, p, save_name="predicted_spectrum.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(WAVELENGTHS, spectrum, color="b", linewidth=2, label="ANN Predicted R(λ)")
    plt.title(
        f"OCD Surrogate Model Prediction\n"
        f"(w={w*1000:.0f}nm, h={h*1000:.0f}nm, p={p*1000:.0f}nm)",
        fontsize=14,
    )
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Reflectance", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    print(f"光譜圖已儲存為: {save_name}")
    plt.show()


def main():
    print("正在載入模型與正規化器...")
    model, scaler, device = load_model_and_scaler()
    print(f"使用裝置: {device}\n")

    while True:
        print("=== OCD 光譜預測 ===")
        print("輸入結構參數 (單位 um，直接 Enter 或輸入 q 離開):")

        raw_w = input(f"  請輸入 w (50~250 nm) [um]: ").strip()
        if raw_w.lower() in ("q", ""):
            break
        try:
            w = float(raw_w)
        except ValueError:
            print("請輸入數字\n")
            continue
        if not (PARAM_BOUNDS["w"][0] <= w <= PARAM_BOUNDS["w"][1]):
            print(f"w 超出範圍 ({PARAM_BOUNDS['w'][0]}~{PARAM_BOUNDS['w'][1]} um)\n")
            continue

        raw_h = input(f"  請輸入 h (100~500 nm) [um]: ").strip()
        if raw_h.lower() in ("q", ""):
            break
        try:
            h = float(raw_h)
        except ValueError:
            print("請輸入數字\n")
            continue
        if not (PARAM_BOUNDS["h"][0] <= h <= PARAM_BOUNDS["h"][1]):
            print(f"h 超出範圍 ({PARAM_BOUNDS['h'][0]}~{PARAM_BOUNDS['h'][1]} um)\n")
            continue

        raw_p = input(f"  請輸入 p (200~600 nm) [um]: ").strip()
        if raw_p.lower() in ("q", ""):
            break
        try:
            p = float(raw_p)
        except ValueError:
            print("請輸入數字\n")
            continue
        if not (PARAM_BOUNDS["p"][0] <= p <= PARAM_BOUNDS["p"][1]):
            print(f"p 超出範圍 ({PARAM_BOUNDS['p'][0]}~{PARAM_BOUNDS['p'][1]} um)\n")
            continue

        spectrum, elapsed_ms = predict_spectrum(model, scaler, device, w, h, p)
        print(f"\n預測完成！花費時間: {elapsed_ms:.3f} ms")
        print(f"w={w*1000:.1f}nm, h={h*1000:.1f}nm, p={p*1000:.1f}nm\n")

        plot_spectrum(spectrum, w, h, p)
        print()


if __name__ == "__main__":
    main()