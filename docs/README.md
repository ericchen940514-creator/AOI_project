# OCD 矽光柵代理模型開發專案

本專案結合 **RCWA** 物理模擬與**深度學習**，實現奈米級半導體結構的即時光學關鍵尺寸 (OCD) 檢測。輸入光柵幾何參數 (w, h, p) 可在 **1 ms** 內預測反射光譜；輸入量測光譜可反推幾何參數。

---

## 專案結構

```
AOI_project/
├── data_generation/        # 資料生成流程
│   ├── materials.py        # 材質 n/k 資料庫（從 refractiveindex.info 讀取）
│   ├── params.py           # LHS 參數取樣（生成 w/h/p 組合）
│   ├── generate.py         # RCWA 寬頻模擬主程式
│   └── output/             # 生成的 CSV 檔案輸出位置
│
├── ann/                    # ANN 流程（w/h/p ↔ 光譜，梯度反推）
│   ├── train.py            # 訓練 ANN surrogate（w,h,p → 光譜）
│   ├── predict.py          # Forward 互動預測（輸入參數 → 畫出光譜）
│   ├── inverse.py          # Inverse 梯度優化反推（輸入光譜 → 輸出參數）
│   ├── model.pth           # 訓練好的 ANN 權重
│   └── scaler.pkl          # 輸入正規化器
│
├── cnn/                    # CNN 流程（光譜 → 參數，直接 inference）
│   ├── train.py            # 訓練 1D CNN（光譜 → w,h,p）
│   ├── predict.py          # Inverse 預測（輸入光譜 → 輸出參數，< 1 ms）
│   └── model.pth           # 訓練好的 CNN 權重（訓練後產生）
│
├── tidy3d/                 # Tidy3D FDTD 高精度模擬
│   ├── setup_api.py        # 設定並驗證 Tidy3D API Key
│   ├── simulate.py         # 單筆 FDTD 模擬（--w/--h/--p 參數）
│   └── output/             # HDF5、CSV、PNG 輸出位置
│
└── data/                   # 共用資料集
    ├── rcwa_spectra_pretrain.csv   # RCWA 訓練集（20,000 筆）
    └── rcwa_spectra_TEST.csv       # 測試集（500 筆）
```

**參數範圍（單位：μm）**

| 參數 | 說明 | 範圍 |
|------|------|------|
| w | 光柵線寬 | 0.05 ~ 0.25 |
| h | 光柵高度 | 0.10 ~ 0.50 |
| p | 週期 (Pitch) | 0.20 ~ 0.60 |
| w/p | 佔空比 (Fill Factor) | 0.30 ~ 0.70（物理限制） |

---

## 環境安裝

```bash
# 基礎套件
pip install numpy pandas scikit-learn matplotlib joblib scipy tqdm pyyaml

# PyTorch（GPU 版，CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# RCWA 模擬引擎
pip install grcwa

# 材質資料庫（首次執行時自動下載 refractiveindex.info 資料庫）
pip install refractiveindex

# Tidy3D（選用，需要帳號 API Key）
pip install tidy3d
```

---

## 完整流程 SOP

### Step 1 — 生成參數組合

用 Latin Hypercube Sampling 取樣 (w, h, p)，並過濾不合理的佔空比：

```bash
python data_generation/params.py
```

**輸出：**
- `data_generation/output/rcwa_params_train.csv`（20,000 筆）
- `data_generation/output/rcwa_params_test.csv`（500 筆）

---

### Step 2 — 跑 RCWA 模擬，生成光譜資料集

```bash
# 預設材質 Si，全量跑（約需數小時，使用多核心平行）
python data_generation/generate.py

# 換材質
python data_generation/generate.py --material TiO2
python data_generation/generate.py --material GaN

# 先跑前 100 筆確認環境沒問題
python data_generation/generate.py --test
```

**支援材質：**

| 名稱 | 材質 | 資料來源 |
|------|------|----------|
| `Si` | 矽 | Aspnes |
| `SiO2` | 二氧化矽 | Franta |
| `Si3N4` | 氮化矽 | Beliaev |
| `TiO2` | 二氧化鈦 | Franta |
| `GaN` | 氮化鎵 | Kawashima |
| `Ge` | 鍺 | Aspnes |

**新增材質：** 去 [refractiveindex.info](https://refractiveindex.info) 找到目標材質的 URL，在 `data_generation/materials.py` 的 `MATERIALS` 字典加一行：

```python
"Al": ("main", "Al", "nk/Rakic"),
```

**輸出：**
- `data_generation/output/rcwa_Si_train.csv`

---

### Step 3 — 訓練 ANN（Forward Surrogate）

ANN 學習 (w, h, p) → 光譜的對應關係，可在 1 ms 內做 forward 預測：

```bash
python ann/train.py
```

**輸出：**
- `ann/model.pth`（模型權重）
- `ann/scaler.pkl`（正規化器）
- `ann/loss_curve.png`

---

### Step 4 — 訓練 CNN（Inverse 直接推參數）

CNN 學習光譜 → (w, h, p) 的對應關係，inference 瞬間完成：

```bash
python cnn/train.py
```

**輸出：**
- `cnn/model.pth`
- `cnn/loss_curve.png`
- Terminal 會印出 validation MAE（各參數誤差 nm）

---

### Step 5 — 預測與反推

#### ANN Forward：輸入參數 → 畫出光譜

```bash
python ann/predict.py
```
互動式輸入 w、h、p，按 q 離開。

#### ANN Inverse：輸入光譜 → 反推參數（梯度優化，約 80 秒）

```bash
python ann/inverse.py
```
選擇模式：
- `d`（demo）：用已知參數生成光譜再反推，驗證流程
- `c`（csv）：從 `data/rcwa_spectra_TEST.csv` 選一列當輸入

#### CNN Inverse：輸入光譜 → 反推參數（< 1 ms）

```bash
python cnn/predict.py
```
同樣支援 demo / csv 兩種模式。

---

### Step 6 — Tidy3D 高精度 FDTD 模擬（選用）

FDTD 模擬精度高於 RCWA，可作為模型校準的黃金標準。需要 Tidy3D 帳號。

```bash
# 設定 API Key（PowerShell）
$env:TIDY3D_API_KEY="your_api_key"

# 驗證 API 連線
python tidy3d/setup_api.py

# 執行單筆模擬
python tidy3d/simulate.py --w 0.15 --h 0.30 --p 0.40

# 只跑 reference simulation（不含元件）
python tidy3d/simulate.py --w 0.15 --h 0.30 --p 0.40 --reference-only
```

**輸出：**（存放於 `tidy3d/output/`）
- `dev_w150_h300_p400.hdf5`
- `ref_w150_h300_p400.hdf5`
- `tidy3d_w150_h300_p400.csv`
- `tidy3d_w150_h300_p400.png`

---

## ANN vs CNN 比較

| | ANN Inverse（梯度優化） | CNN Inverse（直接推） |
|---|---|---|
| 方法 | 梯度下降迭代 1500 步 × 8 restarts | 單次 forward pass |
| 速度 | ~80 秒 | < 1 ms |
| 需要訓練 | 不需要（用 ANN surrogate） | 需要先跑 `cnn/train.py` |
| 精度 | 通常較高（可收斂到 0 誤差） | 依訓練資料量與 MAE |

---

## 開發者

- **Chen Guan-ting（陳冠廷）**
- 國立台灣大學 機械工程學系
