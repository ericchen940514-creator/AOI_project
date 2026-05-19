"""
ANN Surrogate — Forward: (w, h, p) → spectrum (100 points)
Train: python ann/train.py
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "data/rcwa_spectra_pretrain.csv"
MODEL_SAVE = "ann/model.pth"
SCALER_SAVE = "ann/scaler.pkl"
EPOCHS = 300
BATCH_SIZE = 256
LR = 0.001


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 100), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_PATH)
    X_raw = df[["w", "h", "p"]].values
    Y_raw = df.loc[:, "R_wl_0":"R_wl_99"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    joblib.dump(scaler, SCALER_SAVE)

    X_tr, X_val, Y_tr, Y_val = train_test_split(X_scaled, Y_raw, test_size=0.2, random_state=42)
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    X_tr, X_val, Y_tr, Y_val = map(to_tensor, [X_tr, X_val, Y_tr, Y_val])

    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)

    model = ANN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15)

    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = sum(
            (lambda loss: (loss.backward(), optimizer.step(), loss.item()))(
                (optimizer.zero_grad(), criterion(model(bx), by))[1]
            )[-1]
            for bx, by in loader
        )
        train_losses.append(epoch_loss / len(loader))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), Y_val).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}  train={train_losses[-1]:.6f}  val={val_loss:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"Model saved to {MODEL_SAVE}")

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("ANN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ann/loss_curve.png", dpi=150)
    print("Loss curve saved to ann/loss_curve.png")


if __name__ == "__main__":
    main()
