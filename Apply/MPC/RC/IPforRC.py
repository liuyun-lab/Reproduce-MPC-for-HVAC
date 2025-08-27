import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import json
from RC_model import simulate_3r2c_onestep

#Loss function
def model_func(u_flat, *theta):
    # u_flat: flattened input array [Ta, Isol, Qh, Qint]
    N = len(u_flat) // 4
    u_seq = u_flat.reshape(N, 4)
    Ti_sim = simulate_3r2c_onestep(u_seq, theta, Ti_measured)
    return Ti_sim

# Data preparation
#df = pd.read_csv("../uPI_day30_to_day90_results.csv")
df = pd.read_csv("../../Baseline_dataset.csv")

Ta = df["Ambient_Temperature"].values  # Outdoor temperature
Isol = df["Solar_Radiation"].values    # Solar radiation
Qh = df["Q_Heat_Pump"].values          # Floor heating
Qint = df["Occupancy_Gain"].values     # Internal heat gain
Ti_measured = df["Indoor_Temperature"].values  # Measured indoor temperature
Ti0 = Ti_measured[0]
u_seq = np.stack([Ta, Isol, Qh, Qint], axis=1)
N = len(Ta)

#Initial parameter guess and bounds
lower_bounds = np.array([1e-6, 1e-6, 1e-6, 1e6, 1e8, 1, 1, 0.01])
upper_bounds = np.array([1, 1, 1, 1e10, 1e13, 1e3, 1e3, 1e3])
theta_bounds = (lower_bounds, upper_bounds)

theta_init = np.sqrt(lower_bounds * upper_bounds)

#Parameter identification using curve_fit
[popt, pcov] = curve_fit(
    model_func,
    u_seq.flatten(),
    Ti_measured,
    p0=theta_init,
    bounds=theta_bounds,
    maxfev=10000
)

# Output the optimal parameters
print("Optimal parameters:")
param_names = ["Rwin", "Rea", "Rie", "Ci", "Ce", "Ai", "Ae", "K"]
for name, val in zip(param_names, popt):
    print(f"{name}: {val:.6g}")

# store as json file
params_dict = {name: float(val) for name, val in zip(param_names, popt)}
with open("identified_params.json", "w") as f:
    json.dump(params_dict, f, indent=4)
print("Optimal parameters saved to identified_params.json")



# #Visualization and error metrics
# Ti_sim = simulate_3r2c_onestep(u_seq, popt, Ti_measured)
# plt.plot(Ti_measured, label="Measured")
# plt.plot(Ti_sim, label="Simulated")
# plt.legend()
# plt.xlabel("Time step")
# plt.ylabel("Indoor Temperature (°C)")
# plt.title("Measured vs 3R2C onestep")
# plt.show()
#
#
# #store predict Temperature
# time = df['Time'].values[1:]  # Skip the first row to match y_pred/y_true
#
# # Create a DataFrame for export
# result_df = pd.DataFrame({
#     'Time': time,
#     'True_Temperature': Ti_measured[1:] ,
#     'Predicted_Temperature': Ti_sim[1:]
# })
#
# # Export to CSV
# result_df.to_csv('3R2C_temperature_prediction.csv', index=False)