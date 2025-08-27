import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import mean_absolute_error
import json

# Load scalers and model
x_scaler = joblib.load("x_scaler.save")
y_scaler = joblib.load("y_scaler.save")
df = pd.read_csv('../../evaluation_data.csv')

Ti = df['Indoor_Temperature'].values
Ta = df['Ambient_Temperature'].values
Isol = df['Solar_Radiation'].values
Qh = df['Q_Heat_Pump'].values
Qint = df['Occupancy_Gain'].values

class MLP(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)

model = MLP(input_size=5)
model.load_state_dict(torch.load("best_mlp.pth"))
model.eval()

# Multistep  predict
steps = 24
num_samples = len(Ti) - steps

all_preds = np.zeros((num_samples, steps))
all_trues = np.zeros((num_samples, steps))

for n in range(num_samples):
    Ti_pred = Ti[n]
    for i in range(steps):
        input_vec = np.array([
            Ti_pred,
            Ta[n+i],
            Isol[n+i],
            Qh[n+i],
            Qint[n+i]
        ]).reshape(1, -1)
        input_norm = x_scaler.transform(input_vec)
        input_tensor = torch.tensor(input_norm, dtype=torch.float32)
        with torch.no_grad():
            y_pred_norm = model(input_tensor).numpy()
            y_pred = y_scaler.inverse_transform(y_pred_norm)[0, 0]
        all_preds[n, i] = y_pred
        all_trues[n, i] = Ti[n+i+1]
        Ti_pred = y_pred

# Calculate MAE and stds for each step
maes = []
stds = []
for i in range(steps):
    mae = mean_absolute_error(all_trues[:, i], all_preds[:, i])
    std = np.std(all_preds[:, i] - all_trues[:, i])  # 标准差
    maes.append(mae)
    stds.append(std)

# Save to JSON
result = {
    "Step": list(range(1, steps+1)),
    "MAE": maes,
    "STD": stds
}
with open('ANN_multistep_metrics.json', 'w') as f:
    json.dump(result, f, indent=4)

print("MultiStep Prediction Metrics saved to ANN_multistep_metrics.json")













