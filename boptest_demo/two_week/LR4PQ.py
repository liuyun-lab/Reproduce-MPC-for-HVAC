import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data

# #df = pd.read_csv('uPI_typical_heat_day.csv')
# df = pd.read_csv('uPI_peak_heat_day.csv')

df_typical = pd.read_csv('uPI_typical_heat_day.csv')
df_peak = pd.read_csv('uPI_peak_heat_day.csv')
df = pd.concat([df_typical, df_peak], ignore_index=True)

# Extract relevant columns
u = df['Action'].values.reshape(-1, 1)
Ph = df['Heat_Pump_Power'].values
Qh = df['Q_Heat_Pump'].values

# Filter out u=0 (no action)
mask = (u.flatten() >= 0.01)
u_nonzero = u[mask].reshape(-1, 1)
Ph_nonzero = Ph[mask]
Qh_nonzero = Qh[mask]

# Linear regression: electrical power vs action
reg_ph = LinearRegression().fit(u_nonzero, Ph_nonzero)
print(f"Ph = {reg_ph.coef_[0]:.0f} * u + {reg_ph.intercept_:.0f}")

# Linear regression: thermal power vs action
reg_qh = LinearRegression().fit(u_nonzero, Qh_nonzero)
print(f"Qh = {reg_qh.coef_[0]:.0f} * u + {reg_qh.intercept_:.0f}")

Ph_pred = reg_ph.predict(u_nonzero)
Qh_pred = reg_qh.predict(u_nonzero)

def plot_regression(y_true, y_pred, ylabel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, edgecolors='b', facecolors='none', alpha=0.7)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='1:1')
    ax.plot([min_val, max_val], [min_val*1.15, max_val*1.15], 'k--', lw=1, alpha=0.5)
    ax.plot([min_val, max_val], [min_val*0.85, max_val*0.85], 'k--', lw=1, alpha=0.5)
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'$R^2$={r2:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.set_xlabel(f'Simulated {ylabel}')
    ax.set_ylabel(f'Predicted {ylabel}')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    return ax

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

plot_regression(Ph_nonzero, Ph_pred, 'electrical power (W)', ax=axs[0])
axs[0].set_title('Electrical Power Regression')
axs[0].text(0.7, 0.15, '+15%', transform=axs[0].transAxes, fontsize=12)
axs[0].text(0.7, 0.85, '-15%', transform=axs[0].transAxes, fontsize=12)

plot_regression(Qh_nonzero, Qh_pred, 'thermal power (W)', ax=axs[1])
axs[1].set_title('Thermal Power Regression')
axs[1].text(0.7, 0.15, '+15%', transform=axs[1].transAxes, fontsize=12)
axs[1].text(0.7, 0.85, '-15%', transform=axs[1].transAxes, fontsize=12)

plt.tight_layout()
plt.show()