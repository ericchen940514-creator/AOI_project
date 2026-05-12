import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 系統與硬體設定
# ==========================================
# 檢查是否有 GPU 可用！這就是你顯卡發威的時候了
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"目前使用的運算設備: {device}")

# ==========================================
# 1. 資料讀取與前處理 (ETL)
# ==========================================
print("正在讀取 20,000 筆預訓練資料...")
df = pd.read_csv("rcwa_spectra_pretrain.csv")

# 萃取特徵 X (w, h, p) 與 標籤 Y (100 個波長的反射率)
X_raw = df[['w', 'h', 'p']].values
Y_raw = df.loc[:, 'R_wl_0':'R_wl_99'].values

# 【關鍵步驟】特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 切分訓練集 (80%) 與驗證集 (20%)
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_raw, test_size=0.2, random_state=42)

# 轉換為 PyTorch Tensor，並放到 GPU 上
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)

# 建立 DataLoader (批次訓練，Batch Size 設為 256 是滿有效率的大小)
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 2. 定義神經網路架構 (MLP)
# ==========================================
class OCD_Surrogate_Model(nn.Module):
    def __init__(self):
        super(OCD_Surrogate_Model, self).__init__()
        # 漏斗型的網路設計，逐漸把 3 個維度的幾何特徵，解壓縮成 100 維的光譜
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
            nn.Sigmoid() # 反射率 R(lambda) 物理上一定介於 0 到 1 之間
        )

    def forward(self, x):
        return self.network(x)

model = OCD_Surrogate_Model().to(device)

# ==========================================
# 3. 訓練設定
# ==========================================
epochs = 300
criterion = nn.MSELoss() # 使用均方誤差作為 Loss 函數
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 加入 Learning Rate Scheduler (當 Validation Loss 不再下降時，自動把 lr 砍半)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

train_losses = []
val_losses = []

# ==========================================
# 4. 開始訓練 (Training Loop)
# ==========================================
print("\n開始訓練神經網路...")
for epoch in range(epochs):
    model.train() # 設定為訓練模式
    epoch_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()           # 清空上一步的梯度
        predictions = model(batch_x)    # 前向傳播 (預測)
        loss = criterion(predictions, batch_y) # 計算誤差
        loss.backward()                 # 反向傳播 (計算梯度)
        optimizer.step()                # 更新權重
        epoch_loss += loss.item()
        
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 驗證階段 (不計算梯度)
    model.eval() 
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = criterion(val_predictions, Y_val_tensor).item()
        val_losses.append(val_loss)
        
    # 更新 Scheduler
    scheduler.step(val_loss)
    
    # 每 10 個 Epoch 印出一次進度
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

# ==========================================
# 5. 儲存模型與繪製 Loss 曲線
# ==========================================
# 儲存訓練好的網路權重與正規化器 (之後 Fine-tune 和預測時會用到)
torch.save(model.state_dict(), "ocd_surrogate_pretrained.pth")
import joblib
joblib.dump(scaler, "scaler.pkl")

print("\n✅ 訓練完成！模型已儲存為 ocd_surrogate_pretrained.pth")

# 畫圖看看 Loss 曲線有沒有漂亮地收斂
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Curve')
plt.yscale('log') # 用 Log scale 看得比較清楚
plt.legend()
plt.savefig("loss_curve.png")
print("Loss 曲線已儲存為 loss_curve.png")