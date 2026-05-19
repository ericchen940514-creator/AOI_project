"""
OCD Pipeline — Streamlit UI
Run: streamlit run app.py  (or just press the VS Code run button)
"""
import sys, subprocess, os
if "streamlit" not in sys.argv[0]:
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
    sys.exit()

import io
import os
import subprocess
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="OCD Pipeline", layout="wide")
st.title("OCD 光柵結構分析平台")

PARAM_BOUNDS = {"w": (0.05, 0.25), "h": (0.10, 0.50), "p": (0.20, 0.60)}
PARAM_NAMES  = ["w", "h", "p"]
WAVELENGTHS  = np.linspace(400, 700, 100)

# ── Shared model definitions ──────────────────────────────────────────────────

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
    def forward(self, x): return self.net(x)


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Sigmoid(),
        )
    def forward(self, x): return self.fc(self.conv(x))


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource
def load_ann():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANN().to(device)
    model.load_state_dict(torch.load("ann/model.pth", map_location=device, weights_only=True))
    model.eval()
    scaler = joblib.load("ann/scaler.pkl")
    return model, scaler, device


@st.cache_resource
def load_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D().to(device)
    model.load_state_dict(torch.load("cnn/model.pth", map_location=device, weights_only=True))
    model.eval()
    scaler = joblib.load("cnn/param_scaler.pkl")
    return model, scaler, device


# ── Helper: spectrum plot ─────────────────────────────────────────────────────

def spectrum_fig(spectra: dict, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    for (label, data), color in zip(spectra.items(), colors):
        ax.plot(WAVELENGTHS, data, linewidth=2, label=label, color=color)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    return fig


# ── ANN inverse helpers ───────────────────────────────────────────────────────

def _inv_sig(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))

def _to_logits(params):
    return np.array([
        _inv_sig((params[i] - PARAM_BOUNDS[n][0]) / (PARAM_BOUNDS[n][1] - PARAM_BOUNDS[n][0]))
        for i, n in enumerate(PARAM_NAMES)
    ], dtype=np.float32)

def _from_logits(logits):
    return torch.stack([
        PARAM_BOUNDS[n][0] + (PARAM_BOUNDS[n][1] - PARAM_BOUNDS[n][0]) * torch.sigmoid(logits[i])
        for i, n in enumerate(PARAM_NAMES)
    ])

def ann_inverse(model, scaler, device, target_spectrum,
                num_restarts=8, steps=1500, lr=0.03):
    import pandas as pd
    target_t = torch.tensor(target_spectrum, dtype=torch.float32, device=device).unsqueeze(0)
    s_mean  = torch.tensor(scaler.mean_,  dtype=torch.float32, device=device)
    s_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    # warm-start from nearest neighbours in training set
    df = pd.read_csv("data/rcwa_spectra_pretrain.csv")
    spectra = df.loc[:, "R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)
    params  = df[PARAM_NAMES].to_numpy(dtype=np.float32)
    dists   = np.mean((spectra - target_spectrum[None, :]) ** 2, axis=1)
    inits   = params[np.argsort(dists)[:num_restarts]].copy()
    rng = np.random.default_rng(42)
    for i in range(1, len(inits)):
        inits[i] += rng.normal(0, [0.01, 0.02, 0.02])
        for j, n in enumerate(PARAM_NAMES):
            inits[i, j] = np.clip(inits[i, j], *PARAM_BOUNDS[n])

    best = None
    bar = st.progress(0, text="最佳化中...")
    for ri, init_p in enumerate(inits):
        logits = torch.tensor(_to_logits(init_p), device=device, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=lr)
        best_loss, best_p, best_s = float("inf"), None, None
        for _ in range(steps):
            opt.zero_grad()
            p = _from_logits(logits)
            ff = p[0] / p[2]
            ff_pen = torch.relu(0.30 - ff) ** 2 + torch.relu(ff - 0.70) ** 2
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
        bar.progress((ri + 1) / num_restarts, text=f"Restart {ri+1}/{num_restarts}  loss={best['loss']:.6f}")
    bar.empty()
    return best


# ── Sidebar navigation ────────────────────────────────────────────────────────

page = st.sidebar.radio(
    "功能選單",
    [
        "ANN Forward — 輸入參數預測光譜",
        "ANN Inverse — 梯度反推參數",
        "CNN Inverse — 直接反推參數",
        "生成資料集",
        "訓練模型",
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ANN Forward
# ═══════════════════════════════════════════════════════════════════════════════
if page == "ANN Forward — 輸入參數預測光譜":
    st.header("ANN Forward Prediction")
    st.caption("輸入光柵幾何參數，預測反射光譜 R(λ)")

    col1, col2, col3 = st.columns(3)
    w = col1.slider("w — 線寬 (nm)", 50, 250, 150) / 1000
    h = col2.slider("h — 高度 (nm)", 100, 500, 300) / 1000
    p = col3.slider("p — 週期 (nm)", 200, 600, 400) / 1000

    st.write(f"Fill factor = {w/p:.3f}", "（限制：0.30 ~ 0.70）")
    if not (0.30 <= w / p <= 0.70):
        st.warning("Fill factor 超出物理範圍（0.30 ~ 0.70），預測結果可能不可靠。")

    if st.button("預測光譜", type="primary"):
        if not os.path.exists("ann/model.pth"):
            st.error("找不到 ann/model.pth，請先執行 `python ann/train.py` 訓練模型。")
        else:
            model, scaler, device = load_ann()
            x = torch.tensor(scaler.transform([[w, h, p]]), dtype=torch.float32).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                spectrum = model(x).cpu().numpy()[0]
            ms = (time.perf_counter() - t0) * 1000

            st.success(f"預測完成（{ms:.2f} ms）")
            st.pyplot(spectrum_fig(
                {"ANN Predicted R(λ)": spectrum},
                f"w={w*1000:.0f}nm  h={h*1000:.0f}nm  p={p*1000:.0f}nm",
            ))
            st.session_state["last_spectrum"] = spectrum
            st.session_state["last_params"]   = (w, h, p)

    if "last_spectrum" in st.session_state:
        st.info("此光譜已存入 session，可直接切換到 Inverse 頁面使用。")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANN Inverse
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ANN Inverse — 梯度反推參數":
    st.header("ANN Inverse — 梯度優化反推")
    st.caption("輸入光譜 → 反推 w, h, p（約需 80 秒）")

    mode = st.radio("目標光譜來源", ["使用上一頁生成的光譜", "手動上傳 CSV"])

    target = None
    true_p = None

    if mode == "使用上一頁生成的光譜":
        if "last_spectrum" in st.session_state:
            target = st.session_state["last_spectrum"]
            true_p = st.session_state.get("last_params")
            st.pyplot(spectrum_fig({"Target": target}, "目標光譜"))
        else:
            st.warning("請先到「ANN Forward」頁面生成一條光譜。")

    else:
        uploaded = st.file_uploader("上傳 CSV（需有 R_wl_0 ~ R_wl_99 欄位，或單欄 100 個值）", type="csv")
        if uploaded:
            import pandas as pd
            df = pd.read_csv(uploaded)
            if "R_wl_0" in df.columns:
                target = df.loc[0, "R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)
                if all(c in df.columns for c in PARAM_NAMES):
                    true_p = tuple(df.iloc[0][PARAM_NAMES].to_numpy(dtype=np.float32))
            else:
                target = df.iloc[:, 0].to_numpy(dtype=np.float32)[:100]
            st.pyplot(spectrum_fig({"Target": target}, "目標光譜"))

    if target is not None and st.button("開始反推", type="primary"):
        if not os.path.exists("ann/model.pth"):
            st.error("找不到 ann/model.pth，請先訓練模型。")
        else:
            model, scaler, device = load_ann()
            t0 = time.perf_counter()
            result = ann_inverse(model, scaler, device, target)
            ms = (time.perf_counter() - t0) * 1000

            est = result["params"]
            st.success(f"反推完成（{ms/1000:.1f} 秒）  loss = {result['loss']:.8f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("w", f"{est[0]*1000:.1f} nm",
                      f"誤差 {abs(est[0]-true_p[0])*1000:.1f} nm" if true_p else "")
            c2.metric("h", f"{est[1]*1000:.1f} nm",
                      f"誤差 {abs(est[1]-true_p[1])*1000:.1f} nm" if true_p else "")
            c3.metric("p", f"{est[2]*1000:.1f} nm",
                      f"誤差 {abs(est[2]-true_p[2])*1000:.1f} nm" if true_p else "")

            spectra = {"Target": target, "Recovered": result["spectrum"]}
            if true_p:
                spectra["True params label"] = target  # already shown as target
            st.pyplot(spectrum_fig(spectra, "Target vs Recovered"))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CNN Inverse
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "CNN Inverse — 直接反推參數":
    st.header("CNN Inverse — 直接反推（< 1 ms）")
    st.caption("輸入光譜 → 直接輸出 w, h, p")

    mode = st.radio("目標光譜來源", ["使用上一頁生成的光譜", "手動上傳 CSV"])

    target = None
    true_p = None

    if mode == "使用上一頁生成的光譜":
        if "last_spectrum" in st.session_state:
            target = st.session_state["last_spectrum"]
            true_p = st.session_state.get("last_params")
            st.pyplot(spectrum_fig({"Target": target}, "目標光譜"))
        else:
            st.warning("請先到「ANN Forward」頁面生成一條光譜。")
    else:
        uploaded = st.file_uploader("上傳 CSV", type="csv")
        if uploaded:
            import pandas as pd
            df = pd.read_csv(uploaded)
            if "R_wl_0" in df.columns:
                target = df.loc[0, "R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)
                if all(c in df.columns for c in PARAM_NAMES):
                    true_p = tuple(df.iloc[0][PARAM_NAMES].to_numpy(dtype=np.float32))
            else:
                target = df.iloc[:, 0].to_numpy(dtype=np.float32)[:100]
            st.pyplot(spectrum_fig({"Target": target}, "目標光譜"))

    if target is not None and st.button("CNN 反推", type="primary"):
        if not os.path.exists("cnn/model.pth"):
            st.error("找不到 cnn/model.pth，請先執行 `python cnn/train.py` 訓練模型。")
        else:
            model, param_scaler, device = load_cnn()
            x = torch.tensor(target[None, None, :], dtype=torch.float32).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(x).cpu().numpy()
            ms = (time.perf_counter() - t0) * 1000
            est = param_scaler.inverse_transform(out)[0]

            st.success(f"反推完成（{ms:.3f} ms）")
            c1, c2, c3 = st.columns(3)
            c1.metric("w", f"{est[0]*1000:.1f} nm",
                      f"誤差 {abs(est[0]-true_p[0])*1000:.1f} nm" if true_p else "")
            c2.metric("h", f"{est[1]*1000:.1f} nm",
                      f"誤差 {abs(est[1]-true_p[1])*1000:.1f} nm" if true_p else "")
            c3.metric("p", f"{est[2]*1000:.1f} nm",
                      f"誤差 {abs(est[2]-true_p[2])*1000:.1f} nm" if true_p else "")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — 生成資料集
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "生成資料集":
    st.header("生成 RCWA 資料集")

    st.subheader("Step 1 — 生成參數組合")
    if st.button("執行 params.py"):
        with st.spinner("執行中..."):
            r = subprocess.run(
                [sys.executable, "data_generation/params.py"],
                capture_output=True, text=True
            )
        if r.returncode == 0:
            st.success(r.stdout)
        else:
            st.error(r.stderr)

    st.divider()
    st.subheader("Step 2 — RCWA 模擬")

    from data_generation.materials import MATERIALS
    material = st.selectbox("選擇材質", list(MATERIALS.keys()))
    test_mode = st.checkbox("測試模式（只跑前 100 筆）", value=True)

    if st.button("執行 RCWA 模擬", type="primary"):
        cmd = [sys.executable, "data_generation/generate.py", "--material", material]
        if test_mode:
            cmd.append("--test")
        st.info(f"執行指令：{' '.join(cmd)}")
        st.warning("RCWA 模擬會需要一段時間，請勿關閉視窗。進度請看 terminal。")
        with st.spinner("模擬進行中（請看 terminal 的進度條）..."):
            r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            st.success(r.stdout or "完成！")
        else:
            st.error(r.stderr)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — 訓練模型
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "訓練模型":
    st.header("訓練模型")
    st.warning("訓練需要數分鐘，請勿關閉視窗。訓練進度請看 terminal。")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ANN Surrogate")
        st.caption("訓練方向：(w, h, p) → 光譜")
        if st.button("訓練 ANN", type="primary"):
            with st.spinner("訓練中..."):
                r = subprocess.run(
                    [sys.executable, "ann/train.py"],
                    capture_output=True, text=True
                )
            if r.returncode == 0:
                st.success("ANN 訓練完成！")
                st.text(r.stdout[-1000:])
                if os.path.exists("ann/loss_curve.png"):
                    st.image("ann/loss_curve.png")
            else:
                st.error(r.stderr[-2000:])

    with col2:
        st.subheader("CNN Inverse")
        st.caption("訓練方向：光譜 → (w, h, p)")
        if st.button("訓練 CNN", type="primary"):
            with st.spinner("訓練中..."):
                r = subprocess.run(
                    [sys.executable, "cnn/train.py"],
                    capture_output=True, text=True
                )
            if r.returncode == 0:
                st.success("CNN 訓練完成！")
                st.text(r.stdout[-1000:])
                if os.path.exists("cnn/loss_curve.png"):
                    st.image("cnn/loss_curve.png")
            else:
                st.error(r.stderr[-2000:])
