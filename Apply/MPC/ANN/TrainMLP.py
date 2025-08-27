import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# datasets
# files = ['../u0_day30_to_day90_results.csv', '../u1_day30_to_day90_results.csv',
#          '../uPI_day30_to_day90_results.csv']

files = ['../../Action_0_dataset.csv', '../../Action_1_dataset.csv',
         '../../Baseline_dataset.csv']


dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# input 5 dims
Ti = df['Indoor_Temperature'].values
Ta = df['Ambient_Temperature'].values
Isol = df['Solar_Radiation'].values
Qh = df['Q_Heat_Pump'].values
Qint = df['Occupancy_Gain'].values

# input-output pairs
X = np.stack([Ti[:-1], Ta[:-1], Isol[:-1], Qh[:-1], Qint[:-1]], axis=1)
y = Ti[1:]  # Next time step indoor temperature

# Split into training and test sets 8:2
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Normalize input and output
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_train_norm = x_scaler.fit_transform(X_train)
X_test_norm = x_scaler.transform(X_test)
y_train_norm = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_norm = y_scaler.transform(y_test.reshape(-1, 1))

joblib.dump(x_scaler, "x_scaler.save")
joblib.dump(y_scaler, "y_scaler.save")

# print("\n Scaler params:")
# print(f" (data_min_): {x_scaler.data_min_}")
# print(f" (data_max_): {x_scaler.data_max_}")
# print(f"(data_min_): {y_scaler.data_min_}")
# print(f"(data_max_): {y_scaler.data_max_}")

# Define MLP model with two hidden layers
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)

# Convert to Tensor
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = MLP(input_size=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_losses = []
val_losses = []

# Early stopping parameters
patience = 60
best_loss = np.inf
counter = 0

for epoch in range(500):
    model.train()
    epoch_train_loss = 0
    batch_count = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        batch_count += 1
    avg_train_loss = epoch_train_loss / batch_count
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_tensor)
        val_loss = criterion(val_pred, y_test_tensor)
    val_losses.append(val_loss.item())
    print(f"Epoch {epoch}, train_loss: {avg_train_loss:.6f}, val_loss: {val_loss.item():.6f}")
    # Early stopping check
    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        counter = 0
        torch.save(model.state_dict(), "best_mlp.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered, training stopped.")
            break

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.show()
