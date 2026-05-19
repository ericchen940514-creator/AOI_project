import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARAM_BOUNDS = {
    "w": (0.05, 0.25),
    "h": (0.10, 0.50),
    "p": (0.20, 0.60),
}
PARAM_NAMES = ["w", "h", "p"]
WAVELENGTHS = np.linspace(400, 700, 100)

# 可選模式:
# 1. "demo_forward": 先用指定參數生成 target spectrum，再反推回 w/h/p
# 2. "csv_row": 從既有 RCWA/資料集的一列 spectrum 當作目標
# 3. "manual_array": 手動提供 100 個光譜值
TARGET_MODE = "demo_forward"

# demo_forward 模式設定
DEMO_TARGET_PARAMS = {"w": 0.150, "h": 0.250, "p": 0.400}

# csv_row 模式設定
CSV_PATH = "rcwa_spectra_TEST.csv"
CSV_ROW_INDEX = 0

# manual_array 模式設定: 長度需為 100
MANUAL_SPECTRUM = None

# 反推參數
NUM_RESTARTS = 8
OPTIM_STEPS = 1500
LEARNING_RATE = 0.03
FILL_FACTOR_RANGE = (0.30, 0.70)


def load_model_and_scaler():
    model = OCD_Surrogate_Model().to(DEVICE)
    model.load_state_dict(
        torch.load(
            "ocd_surrogate_pretrained.pth",
            map_location=DEVICE,
            weights_only=True,
        )
    )
    model.eval()
    scaler = joblib.load("scaler.pkl")
    return model, scaler


def params_dict_to_array(params_dict):
    return np.array([[params_dict["w"], params_dict["h"], params_dict["p"]]], dtype=np.float32)


def forward_predict_spectrum(model, scaler, params_array):
    scaled = scaler.transform(params_array)
    input_tensor = torch.tensor(scaled, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        spectrum = model(input_tensor).cpu().numpy()[0]
    return spectrum


def build_target_spectrum(model, scaler):
    if TARGET_MODE == "demo_forward":
        target_params = params_dict_to_array(DEMO_TARGET_PARAMS)
        target_spectrum = forward_predict_spectrum(model, scaler, target_params)
        return target_spectrum, target_params[0], "demo_forward"

    if TARGET_MODE == "csv_row":
        df = pd.read_csv(CSV_PATH)
        row = df.iloc[CSV_ROW_INDEX]
        target_params = row[PARAM_NAMES].to_numpy(dtype=np.float32)
        target_spectrum = row.loc["R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)
        return target_spectrum, target_params, f"csv_row:{CSV_PATH}[{CSV_ROW_INDEX}]"

    if TARGET_MODE == "manual_array":
        if MANUAL_SPECTRUM is None or len(MANUAL_SPECTRUM) != 100:
            raise ValueError("MANUAL_SPECTRUM 必須是長度 100 的 list 或 numpy array。")
        return np.array(MANUAL_SPECTRUM, dtype=np.float32), None, "manual_array"

    raise ValueError(f"未知 TARGET_MODE: {TARGET_MODE}")


def make_initial_candidates(target_spectrum):
    df = pd.read_csv("rcwa_spectra_pretrain.csv")
    spectra = df.loc[:, "R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)
    params = df[PARAM_NAMES].to_numpy(dtype=np.float32)

    distances = np.mean((spectra - target_spectrum[None, :]) ** 2, axis=1)
    best_indices = np.argsort(distances)[:NUM_RESTARTS]
    candidates = params[best_indices].copy()

    # 加一些擾動，避免全部都卡在同一個局部最佳點
    rng = np.random.default_rng(42)
    jitter_scales = np.array([0.01, 0.02, 0.02], dtype=np.float32)
    for i in range(1, len(candidates)):
        candidates[i] += rng.normal(0.0, jitter_scales, size=3)
        for j, name in enumerate(PARAM_NAMES):
            low, high = PARAM_BOUNDS[name]
            candidates[i, j] = np.clip(candidates[i, j], low, high)
    return candidates


def inverse_sigmoid(x):
    eps = 1e-6
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def params_to_logits(params_array):
    logits = []
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        normalized = (params_array[i] - low) / (high - low)
        logits.append(inverse_sigmoid(normalized))
    return np.array(logits, dtype=np.float32)


def logits_to_params(logits_tensor):
    params = []
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_BOUNDS[name]
        normalized = torch.sigmoid(logits_tensor[i])
        params.append(low + (high - low) * normalized)
    return torch.stack(params)


def optimize_inverse(model, scaler, target_spectrum, initial_candidates):
    target_tensor = torch.tensor(target_spectrum, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=DEVICE)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=DEVICE)

    best = None

    for restart_idx, init_params in enumerate(initial_candidates):
        init_logits = params_to_logits(init_params)
        logits = torch.tensor(init_logits, dtype=torch.float32, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=LEARNING_RATE)

        restart_best_loss = float("inf")
        restart_best_params = None
        restart_best_spectrum = None

        for _ in range(OPTIM_STEPS):
            optimizer.zero_grad()

            params_tensor = logits_to_params(logits)
            fill_factor = params_tensor[0] / params_tensor[2]
            ff_low, ff_high = FILL_FACTOR_RANGE
            ff_penalty = torch.relu(ff_low - fill_factor) ** 2 + torch.relu(fill_factor - ff_high) ** 2

            scaled_params = ((params_tensor - scaler_mean) / scaler_scale).unsqueeze(0)
            predicted_spectrum = model(scaled_params)

            spectrum_loss = torch.mean((predicted_spectrum - target_tensor) ** 2)
            loss = spectrum_loss + 0.1 * ff_penalty
            loss.backward()
            optimizer.step()

            current_loss = float(loss.item())
            if current_loss < restart_best_loss:
                restart_best_loss = current_loss
                restart_best_params = params_tensor.detach().cpu().numpy().copy()
                restart_best_spectrum = predicted_spectrum.detach().cpu().numpy()[0].copy()

        if best is None or restart_best_loss < best["loss"]:
            best = {
                "loss": restart_best_loss,
                "params": restart_best_params,
                "spectrum": restart_best_spectrum,
                "restart_idx": restart_idx,
            }

    return best


def plot_result(target_spectrum, predicted_spectrum, estimated_params, source_label):
    plt.figure(figsize=(9, 5))
    plt.plot(WAVELENGTHS, target_spectrum, linewidth=2, label="Target Spectrum", color="black")
    plt.plot(WAVELENGTHS, predicted_spectrum, linewidth=2, linestyle="--", label="Recovered Spectrum", color="tab:red")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.title(
        "Inverse OCD Prediction\n"
        f"source={source_label}, "
        f"w={estimated_params[0] * 1000:.1f}nm, "
        f"h={estimated_params[1] * 1000:.1f}nm, "
        f"p={estimated_params[2] * 1000:.1f}nm"
    )
    plt.tight_layout()
    plt.savefig("inverse_predicted_result.png", dpi=300, bbox_inches="tight")
    print("結果圖已儲存為 inverse_predicted_result.png")
    plt.show()


def main():
    print(f"使用裝置: {DEVICE}")
    model, scaler = load_model_and_scaler()

    target_spectrum, ground_truth_params, source_label = build_target_spectrum(model, scaler)
    initial_candidates = make_initial_candidates(target_spectrum)

    print(f"目標光譜來源: {source_label}")
    if ground_truth_params is not None:
        print(
            "已知參數: "
            f"w={ground_truth_params[0]:.4f} um, "
            f"h={ground_truth_params[1]:.4f} um, "
            f"p={ground_truth_params[2]:.4f} um"
        )

    start_time = time.perf_counter()
    result = optimize_inverse(model, scaler, target_spectrum, initial_candidates)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    estimated = result["params"]
    print("\n反推完成")
    print(f"最佳 restart: {result['restart_idx']}")
    print(f"最終 loss: {result['loss']:.8f}")
    print(f"估計 w = {estimated[0]:.6f} um ({estimated[0] * 1000:.2f} nm)")
    print(f"估計 h = {estimated[1]:.6f} um ({estimated[1] * 1000:.2f} nm)")
    print(f"估計 p = {estimated[2]:.6f} um ({estimated[2] * 1000:.2f} nm)")
    print(f"估計 fill factor = {estimated[0] / estimated[2]:.4f}")
    print(f"總耗時: {elapsed_ms:.2f} ms")

    if ground_truth_params is not None:
        abs_error = np.abs(estimated - ground_truth_params)
        print(
            "絕對誤差: "
            f"dw={abs_error[0] * 1000:.2f} nm, "
            f"dh={abs_error[1] * 1000:.2f} nm, "
            f"dp={abs_error[2] * 1000:.2f} nm"
        )

    plot_result(target_spectrum, result["spectrum"], estimated, source_label)


if __name__ == "__main__":
    main()
