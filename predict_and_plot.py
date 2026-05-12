import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 重新定義相同的神經網路架構
# ==========================================
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

# ==========================================
# 2. 載入訓練好的大腦 (Weights) 與翻譯機 (Scaler)
# ==========================================
print("正在載入模型與正規化器...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 實體化模型並載入權重
model = OCD_Surrogate_Model().to(device)
model.load_state_dict(torch.load("ocd_surrogate_pretrained.pth", map_location=device))
model.eval() # 設定為評估(預測)模式，這很重要！

# 載入 scaler
scaler = joblib.load("scaler.pkl")

# ==========================================
# 3. 設定要預測的新參數 (你可以隨便改這三個數字測試！)
# ==========================================
# 確保數值在我們訓練的範圍內 (w: 0.05~0.25, h: 0.1~0.5, p: 0.2~0.6)
test_w = 0.150  # 寬度 150 nm
test_h = 0.300  # 高度 300 nm
test_p = 0.400  # 週期 400 nm

print(f"\n目標參數: Width={test_w} um, Height={test_h} um, Pitch={test_p} um")

# ==========================================
# 4. 極速預測 (Inference)
# ==========================================
# 步驟 A：將輸入轉為 NumPy 陣列並做正規化 (Scaler)
input_data = np.array([[test_w, test_h, test_p]])
input_scaled = scaler.transform(input_data)

# 步驟 B：轉成 PyTorch Tensor 並丟進 GPU/CPU
input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

# 步驟 C：計算預測時間並執行預測
start_time = time.perf_counter()
with torch.no_grad(): # 預測時不需要計算梯度，可以省記憶體並加速
    predicted_spectrum = model(input_tensor)
end_time = time.perf_counter()

inference_time = (end_time - start_time) * 1000 # 轉換為毫秒 (ms)

# 將輸出的 Tensor 轉回一般的 NumPy 陣列
spectrum_array = predicted_spectrum.cpu().numpy()[0]

print(f"✅ 預測完成！花費時間: {inference_time:.3f} 毫秒 (ms)")

# ==========================================
# 5. 繪製預測的反射光譜 R(lambda)
# ==========================================
# 建立 X 軸的波長序列 (400nm ~ 700nm，切 100 個點)
wavelengths = np.linspace(400, 700, 100)

plt.figure(figsize=(8, 5))
plt.plot(wavelengths, spectrum_array, color='b', linewidth=2, label='ANN Predicted R(λ)')

# 美化圖表
plt.title(f'OCD Surrogate Model Prediction\n(w={test_w*1000:.0f}nm, h={test_h*1000:.0f}nm, p={test_p*1000:.0f}nm)', fontsize=14)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectance', fontsize=12)
plt.ylim(0, 1) # 反射率範圍固定 0 到 1
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 儲存與顯示
save_name = "predicted_spectrum.png"
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"光譜圖已儲存為: {save_name}")
plt.show()