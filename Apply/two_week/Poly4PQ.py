import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Load data
df_typical = pd.read_csv('uPI_typical_heat_day.csv')
df_peak = pd.read_csv('uPI_peak_heat_day.csv')
df = pd.concat([df_typical, df_peak], ignore_index=True)

# Extract relevant columns
u = df['Action'].values.reshape(-1, 1)
Ph = df['Heat_Pump_Power'].values
Qh = df['Q_Heat_Pump'].values

# Filter out u=0 (no action)
mask = (u.flatten() >= 0.001)
u_nonzero = u[mask].reshape(-1, 1)
Ph_nonzero = Ph[mask]
Qh_nonzero = Qh[mask]

# Create polynomial regression models (degree=2)
degree = 2

# Polynomial regression for electrical power vs action
polyreg_ph = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('linear', LinearRegression())
])
polyreg_ph.fit(u_nonzero, Ph_nonzero)

# Get polynomial coefficients
ph_coef = polyreg_ph.named_steps['linear'].coef_
ph_intercept = polyreg_ph.named_steps['linear'].intercept_
print(f"Ph = {ph_coef[2]:.2f} * u^2 + {ph_coef[1]:.2f} * u + {ph_intercept:.2f}")

# Polynomial regression for thermal power vs action
polyreg_qh = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('linear', LinearRegression())
])
polyreg_qh.fit(u_nonzero, Qh_nonzero)

# Get polynomial coefficients
qh_coef = polyreg_qh.named_steps['linear'].coef_
qh_intercept = polyreg_qh.named_steps['linear'].intercept_
print(f"Qh = {qh_coef[2]:.2f} * u^2 + {qh_coef[1]:.2f} * u + {qh_intercept:.2f}")

# Make predictions
Ph_pred = polyreg_ph.predict(u_nonzero)
Qh_pred = polyreg_qh.predict(u_nonzero)

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
axs[0].set_title('Electrical Power Polynomial Regression (degree=2)')
axs[0].text(0.7, 0.15, '+15%', transform=axs[0].transAxes, fontsize=12)
axs[0].text(0.7, 0.85, '-15%', transform=axs[0].transAxes, fontsize=12)

plot_regression(Qh_nonzero, Qh_pred, 'thermal power (W)', ax=axs[1])
axs[1].set_title('Thermal Power Polynomial Regression (degree=2)')
axs[1].text(0.7, 0.15, '+15%', transform=axs[1].transAxes, fontsize=12)
axs[1].text(0.7, 0.85, '-15%', transform=axs[1].transAxes, fontsize=12)

plt.tight_layout()
plt.show()