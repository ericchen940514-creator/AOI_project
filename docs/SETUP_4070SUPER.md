# 4070 Super 機器設置清單

## 前提
- Windows 11
- RTX 4070 Super (12 GB VRAM)
- Git 已安裝

---

## Step 1 — Clone / Pull 專案

```powershell
git clone https://github.com/ericchen940514-creator/<repo-name>.git
cd <repo-name>
# 或者已經有了：
git pull
```

---

## Step 2 — 安裝 Python 環境

```powershell
# 安裝 PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安裝其他套件
pip install torcwa h5py scipy matplotlib pyyaml pandas tqdm scikit-learn
```

驗證 CUDA：
```python
import torch
print(torch.cuda.is_available())       # 必須 True
print(torch.cuda.get_device_name(0))   # RTX 4070 SUPER
```

---

## Step 3 — 安裝 refractiveindex 光學資料庫

```powershell
git clone https://github.com/polyanskiy/refractiveindex.info-database.git "$env:USERPROFILE\.refractiveindex.info-database"
```

驗證路徑存在：`C:\Users\<你的username>\.refractiveindex.info-database\data\main\Si\nk\Aspnes.yml`

---

## Step 4 — 確認資料集在專案根目錄

```
v1_A_256.h5     ← Mattila 2024 Wafer A (已 commit，git pull 會下載)
v1_B_256.h5     ← Mattila 2024 Wafer B (已 commit，git pull 會下載)
```

---

## Step 5 — 依序執行驗證

### 5a. 驗證 torcwa 基本運作（Si 1D grating vs 已知結果）
```powershell
python validate_torcwa.py
```
預期輸出：`MAE < 2%`，存檔 `validate_torcwa.png`

### 5b. 驗證 vs Mattila 2024 實測資料（SiO2 grating）
```powershell
python validate_mattila2024.py
```
預期輸出：`MAE < 5%`（有 sim-to-real gap），存檔 `validate_mattila2024.png`

---

## Step 6 — 生成 Si 訓練資料

```powershell
# 先跑 100 筆測試
python generate_torcwa.py --nrows 100 --test

# 沒問題後跑完整 5000 筆（支援中斷續跑）
python generate_torcwa.py --nrows 5000
```

輸出：`data_generation/output/rcwa_Si_train_torcwa.csv`

---

## 已修正的重要 torcwa 注意事項

1. **呼叫順序**：`set_incident_angle()` 必須在 `add_layer()` 之前呼叫
2. **NX 大小**：必須滿足 `NX ≥ 2 × NG + 1`（NG=101 → NX=210）
3. **1D grating 記憶體優化**：用 `order=[1, NG]` 取代 `[NG, NG]`，結果完全一樣但省 67x 記憶體
4. **Rayleigh anomaly**：波長 ≈ period 時會有奇異矩陣，程式已自動跳過並插值

---

## 預期執行時間（4070 Super）

| 任務 | 估計時間 |
|------|---------|
| validate_torcwa.py (100 wl) | ~30 秒 |
| validate_mattila2024.py (256 wl × 2) | ~2 分鐘 |
| generate_torcwa.py 5000 筆 | ~2-4 小時 |
