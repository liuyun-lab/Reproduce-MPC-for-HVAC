import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from RC_model import simulate_3r2c_onestep
from RC_model import simulate_3r2c_multistep
from sklearn.metrics import mean_absolute_error

# Load identified parameters
with open("identified_params.json", "r") as f:
    params_dict = json.load(f)
param_names = ["Rwin", "Rea", "Rie", "Ci", "Ce", "Ai", "Ae", "K"]
theta = [params_dict[name] for name in param_names]


# Data preparation
df = pd.read_csv("../../Baseline_dataset.csv")
Ta = df["Ambient_Temperature"].values
Isol = df["Solar_Radiation"].values
Qh = df["Q_Heat_Pump"].values
Qint = df["Occupancy_Gain"].values
Ti_measured = df["Indoor_Temperature"].values
u_seq = np.stack([Ta, Isol, Qh, Qint], axis=1)


# --------- Onestep Predict---------
Ti_sim = simulate_3r2c_onestep(u_seq, theta, Ti_measured)

# Store predicted temperature
time = df['Time'].values[1:]  # Skip the first row to match y_pred/y_true
result_df = pd.DataFrame({
    'Time': time,
    'True_Temperature': Ti_measured[1:],
    'Predicted_Temperature': Ti_sim[1:]
})
result_df.to_csv('RC_temperature_prediction.csv', index=False)
print("OneStep Prediction results saved to RC_temperature_prediction.csv")

# --------- Multistep Predict---------
steps = 24
num_samples = len(Ti_measured) - steps

all_preds = np.zeros((num_samples, steps))
all_trues = np.zeros((num_samples, steps))

for n in range(num_samples):
    Ti0 = Ti_measured[n]
    u_window = u_seq[n:n+steps]
    Ti_pred = simulate_3r2c_multistep(u_window, theta, Ti0)
    all_preds[n, :] = Ti_pred
    all_trues[n, :] = Ti_measured[n+1:n+1+steps]

# Calculate MAE and STD for each step
maes = []
stds = []
for i in range(steps):
    mae = mean_absolute_error(all_trues[:, i], all_preds[:, i])
    std = np.std(all_preds[:, i] - all_trues[:, i])
    maes.append(mae)
    stds.append(std)

# Save to JSON
result = {
    "Step": list(range(1, steps+1)),
    "MAE": maes,
    "STD": stds
}
with open('RC_multistep_metrics.json', 'w') as f:
    json.dump(result, f, indent=4)
print("Metrics saved to RC_multistep_metrics.json")

