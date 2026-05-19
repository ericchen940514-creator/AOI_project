import time
import joblib
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd


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
PARAM_NAMES = ["w", "h", "p"]
WAVELENGTHS = np.linspace(400, 700, 100)

NUM_RESTARTS = 8
OPTIM_STEPS = 1500
LEARNING_RATE = 0.03
FILL_FACTOR_RANGE = (0.30, 0.70)


def load_model_and_scaler():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCD_Surrogate_Model().to(device)
    model.load_state_dict(
        torch.load("ocd_surrogate_pretrained.pth", map_location=device, weights_only=True)
    )
    model.eval()
    scaler = joblib.load("scaler.pkl")
    return model, scaler, device


# ── Forward ──────────────────────────────────────────────────────────────────

def forward_predict(model, scaler, device, w, h, p):
    input_data = np.array([[w, h, p]], dtype=np.float32)
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        spectrum = model(input_tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return spectrum.cpu().numpy()[0], elapsed_ms


# ── Inverse helpers ───────────────────────────────────────────────────────────

def _inverse_sigmoid(x):
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _params_to_logits(params_array):
    logits = []
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        normalized = (params_array[i] - low) / (high - low)
        logits.append(_inverse_sigmoid(normalized))
    return np.array(logits, dtype=np.float32)


def _logits_to_params(logits_tensor):
    params = []
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        normalized = torch.sigmoid(logits_tensor[i])
        params.append(low + (high - low) * normalized)
    return torch.stack(params)


def _make_initial_candidates(target_spectrum):
    df = pd.read_csv("rcwa_spectra_pretrain.csv")
    spectra = df.loc[:, "R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)
    params = df[PARAM_NAMES].to_numpy(dtype=np.float32)
    distances = np.mean((spectra - target_spectrum[None, :]) ** 2, axis=1)
    best_indices = np.argsort(distances)[:NUM_RESTARTS]
    candidates = params[best_indices].copy()
    rng = np.random.default_rng(42)
    jitter_scales = np.array([0.01, 0.02, 0.02], dtype=np.float32)
    for i in range(1, len(candidates)):
        candidates[i] += rng.normal(0.0, jitter_scales, size=3)
        for j, name in enumerate(PARAM_NAMES):
            low, high = PARAM_BOUNDS[name]
            candidates[i, j] = np.clip(candidates[i, j], low, high)
    return candidates


def inverse_predict(model, scaler, device, target_spectrum):
    target_tensor = torch.tensor(target_spectrum, dtype=torch.float32, device=device).unsqueeze(0)
    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    initial_candidates = _make_initial_candidates(target_spectrum)
    best = None

    for init_params in initial_candidates:
        init_logits = _params_to_logits(init_params)
        logits = torch.tensor(init_logits, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=LEARNING_RATE)

        restart_best_loss = float("inf")
        restart_best_params = None
        restart_best_spectrum = None

        for _ in range(OPTIM_STEPS):
            optimizer.zero_grad()
            params_tensor = _logits_to_params(logits)
            fill_factor = params_tensor[0] / params_tensor[2]
            ff_low, ff_high = FILL_FACTOR_RANGE
            ff_penalty = torch.relu(ff_low - fill_factor) ** 2 + torch.relu(fill_factor - ff_high) ** 2
            scaled_params = ((params_tensor - scaler_mean) / scaler_scale).unsqueeze(0)
            pred = model(scaled_params)
            loss = torch.mean((pred - target_tensor) ** 2) + 0.1 * ff_penalty
            loss.backward()
            optimizer.step()
            if float(loss.item()) < restart_best_loss:
                restart_best_loss = float(loss.item())
                restart_best_params = params_tensor.detach().cpu().numpy().copy()
                restart_best_spectrum = pred.detach().cpu().numpy()[0].copy()

        if best is None or restart_best_loss < best["loss"]:
            best = {"loss": restart_best_loss, "params": restart_best_params, "spectrum": restart_best_spectrum}

    return best


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_forward(spectrum, w, h, p, save_name="predicted_spectrum.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(WAVELENGTHS, spectrum, color="b", linewidth=2, label="ANN Predicted R(λ)")
    plt.title(
        f"OCD Surrogate Model — Forward Prediction\n"
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


def plot_inverse_result(target_spectrum, recovered_spectrum, true_params, estimated_params):
    plt.figure(figsize=(9, 5))
    plt.plot(WAVELENGTHS, target_spectrum, linewidth=2, color="black", label="Target Spectrum (input)")
    plt.plot(WAVELENGTHS, recovered_spectrum, linewidth=2, linestyle="--", color="tab:red", label="Recovered Spectrum")
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Reflectance", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)

    true_str = f"True:      w={true_params[0]*1000:.1f}nm  h={true_params[1]*1000:.1f}nm  p={true_params[2]*1000:.1f}nm"
    est_str  = f"Estimated: w={estimated_params[0]*1000:.1f}nm  h={estimated_params[1]*1000:.1f}nm  p={estimated_params[2]*1000:.1f}nm"
    err = np.abs(estimated_params - true_params) * 1000
    err_str  = f"Error:     dw={err[0]:.1f}nm  dh={err[1]:.1f}nm  dp={err[2]:.1f}nm"
    plt.title(f"Inverse OCD Prediction\n{true_str}\n{est_str}\n{err_str}", fontsize=11)

    plt.tight_layout()
    plt.savefig("inverse_predicted_result.png", dpi=300, bbox_inches="tight")
    print("反推結果圖已儲存為: inverse_predicted_result.png")
    plt.show()


# ── Input helpers ─────────────────────────────────────────────────────────────

def read_float(prompt, low, high):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "q":
            return None
        try:
            val = float(raw)
            if low <= val <= high:
                return val
            print(f"    超出範圍 ({low:.3f} ~ {high:.3f} um)")
        except ValueError:
            print("    請輸入數字")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("正在載入模型與正規化器...")
    model, scaler, device = load_model_and_scaler()
    print(f"使用裝置: {device}\n")

    while True:
        print("=== OCD 光譜預測 + 反向推參數 ===")
        print("請輸入結構參數 (單位 um，輸入 q 離開):")

        w = read_float("  w (50~250 nm) [um]: ", *PARAM_BOUNDS["w"])
        if w is None:
            break
        h = read_float("  h (100~500 nm) [um]: ", *PARAM_BOUNDS["h"])
        if h is None:
            break
        p = read_float("  p (200~600 nm) [um]: ", *PARAM_BOUNDS["p"])
        if p is None:
            break

        # Forward: 生成光譜
        spectrum, fwd_ms = forward_predict(model, scaler, device, w, h, p)
        print(f"\nForward 預測完成 ({fwd_ms:.2f} ms)")
        print(f"  w={w*1000:.1f}nm, h={h*1000:.1f}nm, p={p*1000:.1f}nm")
        plot_forward(spectrum, w, h, p)

        # Inverse: 用剛才那條光譜反推參數
        print("\n開始反向推算參數（約需數秒）...")
        start = time.perf_counter()
        result = inverse_predict(model, scaler, device, spectrum)
        inv_ms = (time.perf_counter() - start) * 1000

        est = result["params"]
        true_params = np.array([w, h, p], dtype=np.float32)
        err = np.abs(est - true_params) * 1000

        print(f"反推完成 ({inv_ms:.0f} ms)，loss={result['loss']:.6f}")
        print(f"  真實值:   w={w*1000:.1f}nm  h={h*1000:.1f}nm  p={p*1000:.1f}nm")
        print(f"  反推結果: w={est[0]*1000:.1f}nm  h={est[1]*1000:.1f}nm  p={est[2]*1000:.1f}nm")
        print(f"  誤差:     dw={err[0]:.1f}nm  dh={err[1]:.1f}nm  dp={err[2]:.1f}nm")

        plot_inverse_result(spectrum, result["spectrum"], true_params, est)
        print()


if __name__ == "__main__":
    main()
