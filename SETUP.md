# 復現環境設定

## 1. Clone repo
```bash
git clone https://github.com/ericchen940514-creator/AOI_project.git
cd AOI_project
```

## 2. 安裝 Python 套件

需要 Python 3.11。

**先裝 PyTorch（依據 GPU 版本選）：**
```bash
# CUDA 12.1（推薦，RTX 系列）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 沒有 GPU
pip install torch torchvision
```

**再裝其他套件：**
```bash
pip install torcwa numpy scipy matplotlib h5py pyyaml tqdm pandas
```

## 3. 下載 refractiveindex 資料庫

torcwa 需要 SiO2 的 n,k 數據（Franta）：

```bash
git clone https://github.com/polyanskiy/refractiveindex.info-database.git %USERPROFILE%\.refractiveindex.info-database
```

Linux/Mac：
```bash
git clone https://github.com/polyanskiy/refractiveindex.info-database.git ~/.refractiveindex.info-database
```

確認路徑：`~/.refractiveindex.info-database/data/main/SiO2/nk/Franta.yml` 存在。

## 4. 驗證環境
```bash
python -c "import torch, torcwa, h5py; print('GPU:', torch.cuda.is_available())"
```

## 5. 跑完整 pipeline

```bash
# 生成 5000 筆資料 + 訓練 300 epochs
python pipeline.py --nrows 5000 --workers 3 --epochs 300
```

跑完後：
- 資料：`data_generation/output/rcwa_sio2_train_ng9.csv`
- 模型：`ann/ann_sio2_ng9.pth`

## 6. 驗證結果（對比 Mattila 2024 量測）

```bash
python validation/simulate_samples_AB.py
```

結果圖：`validation/simulate_samples_AB.png`
預期 MAE：Sample A ~0.36%、Sample B ~0.03%

## 參數範圍

| 參數 | 範圍 |
|------|------|
| period | 600–760 nm |
| pillar | 150–500 nm |
| height | 100–400 nm |
| fill factor | 0.20–0.75 |
| wavelength | 400–750 nm（256 點） |
| NG | 9 |
