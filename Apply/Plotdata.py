import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

def plot_1():
    df_u0 = pd.read_csv("Action_0_dataset.csv")
    df_u1 = pd.read_csv("Action_1_dataset.csv")
    df_pi = pd.read_csv("Baseline_dataset.csv")
    df_pi2 = pd.read_csv("evaluation_data.csv")
    plt.figure(figsize=(12, 6))
    plt.plot(df_u0["Time"], df_u0["Indoor_Temperature"], color='green', label=r"$u_{hp}=0$")
    plt.plot(df_u1["Time"], df_u1["Indoor_Temperature"], color='blue', label=r"$u_{hp}=1$")
    plt.plot(df_pi["Time"], df_pi["Indoor_Temperature"], color='gray', label="Baseline")
    plt.plot(df_pi2["Time"], df_pi2["Indoor_Temperature"], color='red', label="paper data")
    plt.xlabel("Day of Year")
    plt.ylabel("Room Temperature (°C)")
    plt.title("Room Temperature vs Time under Different Control Signals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_2():
    df_3r2c = pd.read_csv('MPC/RC/RC_temperature_prediction.csv')
    df_ann = pd.read_csv('MPC/ANN/ANN_temperature_prediction.csv')
    # Number of hours to plot (last 7 days)
    N = 7 * 24

    df_3r2c_last = df_3r2c.tail(N).reset_index(drop=True)
    df_ann_last = df_ann.tail(N).reset_index(drop=True)

    end_time = pd.Timestamp('2020-03-31 00:00:00')
    time = pd.date_range(end=end_time, periods=N, freq='h')
    df_3r2c_last['Time'] = time
    df_ann_last['Time'] = time

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_3r2c_last['Time'], df_3r2c_last['True_Temperature'], color='black', label='Measurement', linewidth=2)
    ax.plot(df_3r2c_last['Time'], df_3r2c_last['Predicted_Temperature'], color='deepskyblue', label='3R2C Onestep', linewidth=1)
    ax.plot(df_ann_last['Time'], df_ann_last['Predicted_Temperature'], color='orange', label='ANN Onestep', linewidth=1)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.set_xlim(time[0], time[-1])

    plt.xlabel('Day of Year (2020)')
    plt.ylabel('Temperature (°C)')
    plt.title('(a)')
    #plt.ylim(20, 23)
    plt.legend(loc='upper right', frameon=False)
    plt.tight_layout()
    plt.show()


    y_true = df_3r2c['True_Temperature'].values
    y_pred_3r2c = df_3r2c['Predicted_Temperature'].values
    y_pred_ann = df_ann['Predicted_Temperature'].values

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100

    def cv_rmse(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse / np.mean(y_true) * 100

    # Calculate metrics for 3R2C
    mae_3r2c = mean_absolute_error(y_true, y_pred_3r2c)
    rmse_3r2c = np.sqrt(mean_squared_error(y_true, y_pred_3r2c))
    mape_3r2c = mape(y_true, y_pred_3r2c)
    cvrmse_3r2c = cv_rmse(y_true, y_pred_3r2c)

    # Calculate metrics for ANN
    mae_ann = mean_absolute_error(y_true, y_pred_ann)
    rmse_ann = np.sqrt(mean_squared_error(y_true, y_pred_ann))
    mape_ann = mape(y_true, y_pred_ann)
    cvrmse_ann = cv_rmse(y_true, y_pred_ann)

    # Data for plotting
    metrics1 = [mae_3r2c, rmse_3r2c]
    metrics2 = [mae_ann, rmse_ann]
    metrics1_si = [mape_3r2c, cvrmse_3r2c]
    metrics2_si = [mape_ann, cvrmse_ann]

    labels1 = ['MAE', 'RMSE']
    labels2 = ['MAPE', 'CV-RMSE']

    x = np.arange(len(labels1))  # the label locations
    width = 0.35  # the width of the bars

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # (b1) Scale-dependent metrics
    rects1 = axs[0].bar(x - width/2, metrics1, width, label='3R2C', color='deepskyblue')
    rects2 = axs[0].bar(x + width/2, metrics2, width, label='ANN', color='orange')
    axs[0].set_ylabel('Value (°C)')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels1)
    axs[0].set_title('(b1)')
    axs[0].legend()
    axs[0].set_ylim(0, max(metrics1 + metrics2) * 1.5)
    axs[0].set_xlabel('Scale-dependent Metrics')

    # Annotate bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        axs[0].annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # (b2) Scale-independent metrics
    x2 = np.arange(len(labels2))
    rects3 = axs[1].bar(x2 - width/2, metrics1_si, width, label='3R2C', color='deepskyblue')
    rects4 = axs[1].bar(x2 + width/2, metrics2_si, width, label='ANN', color='orange')
    axs[1].set_ylabel('Value (%)')
    axs[1].set_xticks(x2)
    axs[1].set_xticklabels(labels2)
    axs[1].set_title('(b2)')
    axs[1].legend()
    axs[1].set_ylim(0, max(metrics1_si + metrics2_si) * 1.5)
    axs[1].set_xlabel('Scale-independent Metrics')

    # Annotate bars
    for rect in rects3 + rects4:
        height = rect.get_height()
        axs[1].annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def plot_3():
    with open('MPC/ANN/ANN_multistep_metrics.json', 'r') as f:
        ann_result = json.load(f)
    ann_mae = ann_result["MAE"]
    ann_std = ann_result["STD"]

    with open('MPC/RC/RC_multistep_metrics.json', 'r') as f:
        r3c2_result = json.load(f)
    r3c2_mae = r3c2_result["MAE"]
    r3c2_std = r3c2_result["STD"]

    steps = range(1, 25)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    # 3R2C
    axs[0].plot(steps, r3c2_mae, label='3R2C MAE', color='orange')
    axs[0].fill_between(steps, np.array(r3c2_mae) - np.array(r3c2_std), np.array(r3c2_mae) + np.array(r3c2_std),
                        color='orange', alpha=0.2)
    axs[0].set_title('3R2C Multi-step MAE')
    axs[0].set_xlabel('Prediction Horizon (hours)')
    axs[0].set_ylabel('Mean absolute error(°C)')
    axs[0].set_xticks(steps)
    axs[0].legend()
    # ANN
    axs[1].plot(steps, ann_mae, label='ANN MAE', color='blue')
    axs[1].fill_between(steps, np.array(ann_mae) - np.array(ann_std), np.array(ann_mae) + np.array(ann_std),
                        color='blue', alpha=0.2)
    axs[1].set_title('ANN Multi-step MAE')
    axs[1].set_xlabel('Prediction Horizon (hours)')
    axs[1].set_xticks(steps)
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    # --------- the mean absolute error with error bars --------
    idxs = [0, 3, 7, 11, 15, 19, 23]
    steps = np.array([1, 4, 8, 12, 16, 20, 24])
    ann_mae = np.array(ann_mae)
    ann_std = np.array(ann_std)
    r3c2_mae = np.array(r3c2_mae)
    r3c2_std = np.array(r3c2_std)

    ann_mae_plot = ann_mae[idxs]
    ann_std_plot = ann_std[idxs]
    r3c2_mae_plot = r3c2_mae[idxs]
    r3c2_std_plot = r3c2_std[idxs]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # 3R2C
    axs[0].errorbar(steps, r3c2_mae_plot, yerr=r3c2_std_plot, fmt='s--', color='#3682be', ecolor='#3682be',
                    elinewidth=2, capsize=4, alpha=0.7, label='3R2C')
    axs[0].set_ylabel('Mean absolute error (°C)')
    axs[0].set_xlabel('Prediction horizon (hours)')
    axs[0].set_xticks(steps)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', linewidth=0.7)
    # ANN
    axs[1].errorbar(steps, ann_mae_plot, yerr=ann_std_plot, fmt='s--', color='#f28e2b', ecolor='#f28e2b',
                    elinewidth=2, capsize=4, alpha=0.7, label='ANN')
    axs[1].set_xlabel('Prediction horizon (hours)')
    axs[1].set_xticks(steps)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.show()

def plot_4():
    files = [
        "MPC/Data/KPI_ann_mpc_peak_w=2.2.json",
        "MPC/Data/KPI_rc_mpc_peak_w=2.2.json",
        "MPC/Data/KPI_Baseline_peak_w=2.2.json",
        "MPC/Data/KPI_ann_mpc_typical_w=2.2.json",
        "MPC/Data/KPI_rc_mpc_typical_w=2.2.json",
        "MPC/Data/KPI_Baseline_typical_w=2.2.json"
    ]
    colors = {
        'Baseline': '#2ca02c',  # green
        'rc_mpc': '#1f77b4',  # blue
        'ann_mpc': '#ff7f0e' # orange
    }
    controller_order = ['Baseline', 'rc_mpc', 'ann_mpc']
    scenarios = ['peak', 'typical']

    data = {
        'energy': {scenario: [] for scenario in scenarios},  # Operational cost
        'thermal': {scenario: [] for scenario in scenarios},  # Thermal discomfort
        'time': {scenario: [] for scenario in scenarios}  # Computational time
    }
    for scenario in scenarios:
        for ctrl_type in controller_order:
            match = [f for f in files
                     if scenario.lower() in f.lower()
                     and ctrl_type.lower() in f.lower()]

            if not match:
                print(f"Warning: Missing {ctrl_type} data for {scenario}")
                data['energy'][scenario].append(0)
                data['thermal'][scenario].append(0)
                data['time'][scenario].append(0)
                continue
            try:
                with open(match[0], "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                data['energy'][scenario].append(file_data["Energy cost"])
                data['thermal'][scenario].append(file_data["Thermal discomfort"])
                data['time'][scenario].append(file_data["Time cost"])
            except Exception as e:
                print(f"Error loading {match[0]}: {str(e)}")
                data['energy'][scenario].append(0)
                data['thermal'][scenario].append(0)
                data['time'][scenario].append(0)

    x = np.arange(len(scenarios))
    width = 0.25
    offset = [-width, 0, width]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    plot_configs = [
        ('energy', "Operational cost", "K$_{cost}$ (€/m²)"),
        ('thermal', "Thermal discomfort", "K$_{dis}$ (Kh/zone)"),
        ('time', "Computational time",  "K$_{time}$ (sec)")
    ]
    for ax, (cost_type, title, ylabel) in zip(axs, plot_configs):
        for i, ctrl_type in enumerate(controller_order):
            values = [data[cost_type][scenario][i] for scenario in scenarios]

            display_name = {
                'ann_mpc': 'ANN MPC',
                'rc_mpc': 'RC MPC',
                'Baseline': 'Baseline'
            }[ctrl_type]
            ax.bar(x + offset[i], values, width,
                   color=colors[ctrl_type], label=display_name)
            for j, val in enumerate(values):
                ax.text(x[j] + offset[i], val,
                        f"{val:.2f}", ha='center', va='bottom')
        ax.set_xticks(x)
        ax.set_xticklabels([scenario.title() for scenario in scenarios])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if cost_type == 'energy':
            ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_3()