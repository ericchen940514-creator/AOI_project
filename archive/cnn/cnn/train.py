"""
CNN Inverse — Direct: spectrum (100 points) → (w, h, p)
Train: python cnn/train.py
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "data/rcwa_spectra_pretrain.csv"
MODEL_SAVE = "cnn/model.pth"
SCALER_SAVE = "cnn/param_scaler.pkl"
PARAM_NAMES = ["w", "h", "p"]
EPOCHS = 300
BATCH_SIZE = 256
LR = 1e-3


class CNN1D(nn.Module):
    """
    Input:  (B, 1, 100) — 100-point reflectance spectrum
    Output: (B, 3)      — normalised w, h, p in [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 1, 100) → (B, 32, 98)
            nn.Conv1d(1, 32, kernel_size=3, padding=0), nn.ReLU(),
            # → (B, 32, 49)
            nn.MaxPool1d(2),
            # → (B, 64, 47)
            nn.Conv1d(32, 64, kernel_size=3, padding=0), nn.ReLU(),
            # → (B, 64, 23)
            nn.MaxPool1d(2),
            # → (B, 128, 21)
            nn.Conv1d(64, 128, kernel_size=3, padding=0), nn.ReLU(),
            # → (B, 128, 10)
            nn.MaxPool1d(2),
        )
        # flatten → 128*10 = 1280
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 3),    nn.Sigmoid(),   # output in [0,1]
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_PATH)
    X_raw = df.loc[:, "R_wl_0":"R_wl_99"].to_numpy(dtype=np.float32)   # spectra
    Y_raw = df[PARAM_NAMES].to_numpy(dtype=np.float32)                  # params

    # scale params to [0,1] so Sigmoid output maps directly
    param_scaler = MinMaxScaler()
    Y_scaled = param_scaler.fit_transform(Y_raw).astype(np.float32)
    joblib.dump(param_scaler, SCALER_SAVE)

    X_tr, X_val, Y_tr, Y_val = train_test_split(X_raw, Y_scaled, test_size=0.2, random_state=42)

    # add channel dim: (N, 100) → (N, 1, 100)
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    X_tr_t  = to_tensor(X_tr[:, None, :])
    X_val_t = to_tensor(X_val[:, None, :])
    Y_tr_t  = to_tensor(Y_tr)
    Y_val_t = to_tensor(Y_val)

    loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    model = CNN1D().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15)

    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(loader))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), Y_val_t).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  train={train_losses[-1]:.6f}  val={val_loss:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"Model saved to {MODEL_SAVE}")

    # evaluate in original units (nm)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_val_t).cpu().numpy()
    pred = param_scaler.inverse_transform(pred_scaled)
    true = param_scaler.inverse_transform(Y_val)
    mae_nm = np.abs(pred - true).mean(axis=0) * 1000
    print(f"Val MAE  w={mae_nm[0]:.2f}nm  h={mae_nm[1]:.2f}nm  p={mae_nm[2]:.2f}nm")

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("CNN Training Loss (spectrum → params)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn/loss_curve.png", dpi=150)
    print("Loss curve saved to cnn/loss_curve.png")


if __name__ == "__main__":
    main()
