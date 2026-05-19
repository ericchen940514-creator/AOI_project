

# OCD 矽光柵代理模型開發專案 (Full Pipeline: Generation to Prediction)

本專案結合了 **RCWA (Rigorous Coupled-Wave Analysis)** 物理模擬與 **深度學習 (Deep Learning)** 技術，實現奈米級半導體結構的即時光學關鍵尺寸 (OCD) 檢測。

## 📌 專案核心目標

利用 **CPU 多核心平行運算** 大量產出物理資料，並透過訓練神經網路 (ANN) 將原本需數秒的模擬壓縮至 **1 毫秒 (ms)** 以內，實現工廠端的高速自動化檢測。

---

## 🛠️ 環境安裝

請確保您的 Python 環境為 3.9 以上版本，並安裝以下必要套件：

```bash
# 1. 基礎運算與畫圖
pip install pandas numpy scikit-learn matplotlib joblib

# 2. 物理模擬與平行運算 (RCWA 腳本需求)
# 請確保已安裝您使用的 RCWA 函式庫 (例如 S4 或客製化封裝)

# 3. 深度學習框架 PyTorch (建議使用支援 GPU 的版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

---

## 📂 檔案結構說明

* **`rcwa_data_generator.py`**: **資料生產線**。利用平行運算產生大量 $(w, h, p)$ 組合及其對應反射光譜。
* **`rcwa_spectra_pretrain.csv`**: 核心數據庫。存放 20,000 筆物理模擬結果。
* **`train_ann_pretrain.py`**: 深度神經網路訓練腳本。
* **`ocd_surrogate_pretrained.pth`**: 訓練完成的模型權重檔（神經網路的大腦）。
* **`scaler.pkl`**: 資料正規化器。
* **`predict_and_plot.py`**: 極速預測與畫圖腳本。
* **`inverse_predict.py`**: 由目標 `spectrum` 反推 `(w, h, p)` 的 inverse prediction 腳本。

---

## 🚀 標準作業流程 (SOP)

### 第 0 階段：資料生成 (Data Generation)

如果你需要重新產出原始資料或擴充資料庫：

1. 打開 `rcwa_data_generator.py`。
2. 確認 `TEST_MODE = False` 以啟動全量生產（預設為 20,000 組）。
3. 執行腳本：`python rcwa_data_generator.py`

* **技術亮點**：腳本會自動偵測電腦核心數（預設使用 19 核心），透過平行運算大幅縮短 RCWA 矩陣求解時間。

### 第一階段：大規模預訓練 (Pre-training)

當 CSV 檔案準備好後，開始訓練神經網路：

1. 執行腳本：`python train_ann_pretrain.py`

* **產出物**：`ocd_surrogate_pretrained.pth`、`scaler.pkl` 以及訓練紀錄圖 `loss_curve.png`。

### 第二階段：即時預測與應用 (Inference)

只需輸入尺寸，即可瞬間生成預測光譜：

1. 修改 `predict_and_plot.py` 中的 `test_w`, `test_h`, `test_p`。
2. 執行腳本：`python predict_and_plot.py`

* **結果**：產出 `predicted_spectrum.png`。

### 第二點五階段：由 Spectrum 反推結構參數 (Inverse Prediction)

當你手上有一條目標反射光譜，想估計最可能的 `w / h / p`，可以直接使用 `inverse_predict.py`。

1. 開啟 `inverse_predict.py`
2. 設定 `TARGET_MODE`
3. 執行腳本：`python inverse_predict.py`

目前支援三種模式：

* **`demo_forward`**：先用已知 `(w, h, p)` 經由模型生成目標光譜，再反推回來，適合驗證 inverse 流程。
* **`csv_row`**：從 `rcwa_spectra_TEST.csv` 或 `rcwa_spectra_pretrain.csv` 讀一列資料當目標光譜。
* **`manual_array`**：手動提供 100 個波長點的反射率資料。

* **結果**：終端會輸出估計的 `w`、`h`、`p`、對應 loss 與誤差，並儲存 `inverse_predicted_result.png`。

### 第三階段：Tidy3D 高精度校準 (Fine-tuning)

將 ANN 預測結果與 **Tidy3D (FDTD)** 的模擬結果進行對照，並根據 Tidy3D 的黃金資料進行模型微調，以符合更複雜的 3D 邊界效應。

---

## 🧪 物理模型設定參考 (Tidy3D)

* **邊界條件 (Boundary)**: $x, y$ 設為 `Periodic`；$z$ 設為 `PML`。
* **光源 (Source)**: 平面波 (`Plane Wave`)，波長範圍 `400nm - 700nm`。
* **監測器 (Monitor)**: `FluxMonitor` 取樣點數設定為 `100` 以上以獲得平滑曲線。

---

## 👨‍🔬 開發者

* **Chen Guan-ting (陳冠廷)**
* 國立台灣大學 機械工程學系 (NTUME)

