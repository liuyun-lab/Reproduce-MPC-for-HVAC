import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

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

# Load the scalers for input and output normalization
x_scaler = joblib.load("x_scaler.save")
y_scaler = joblib.load("y_scaler.save")

#df = pd.read_csv('../uPI_day30_to_day90_results.csv')
df = pd.read_csv('../../evaluation_data.csv')


Ti = df['Indoor_Temperature'].values
Ta = df['Ambient_Temperature'].values
Isol = df['Solar_Radiation'].values
Qh = df['Q_Heat_Pump'].values
Qint = df['Occupancy_Gain'].values

# Construct input-output pairs for one-step prediction
X = np.stack([Ti[:-1], Ta[:-1], Isol[:-1], Qh[:-1], Qint[:-1]], axis=1)
y = Ti[1:]

# Normalize input and output
X_norm = x_scaler.transform(X)
y_norm = y_scaler.transform(y.reshape(-1, 1))

X_tensor = torch.tensor(X_norm, dtype=torch.float32)

# Load the trained model
model = MLP(input_size=5)
model.load_state_dict(torch.load("best_mlp.pth"))
model.eval()

# Predict
with torch.no_grad():
    y_pred_norm = model(X_tensor).numpy()
    y_pred = y_scaler.inverse_transform(y_pred_norm)
    y_true = y_scaler.inverse_transform(y_norm)

# Plot the true and predicted indoor temperature
plt.plot(y_true, label="True")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Indoor Temperature (°C)")
plt.title("Measured vs ANN onestep")
plt.show()

#store predict Temperature
time = df['Time'].values[1:]  # Skip the first row to match y_pred/y_true

# Create a DataFrame for export
result_df = pd.DataFrame({
    'Time': time,
    'True_Temperature': y_true.flatten(),
    'Predicted_Temperature': y_pred.flatten()
})

# Export to CSV
result_df.to_csv('ANN_temperature_prediction.csv', index=False)
print("OneStep Prediction results saved to ANN_temperature_prediction.csv")