import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
def eval_R3C2_onestep(inputs, output, params):
    Rie = params['Rie']
    Rea = params['Rea']
    Rwin = params['Rwin']
    Ci = params['Ci']
    Ce = params['Ce']
    Ai = params['Ai']
    Ae = params['Ae']
    a=params['a']

    ta = inputs[:, 0] # ambient temperature
    Q_h = inputs[:, 1] # heat gain delivered by heater
    I_s = inputs[:, 2] # global solar irradiation
    Q_int = inputs[:, 3] # internal heat gain

    ti = np.zeros(len(inputs))
    te = np.zeros(len(inputs))
    ti[0] = output[0]
    te[0] = (Rie * ta[0] + Rea * ti[0]) / (Rie + Rea)
    dt = 3600
    for t in range(1, len(output)):
        ti_init = output[t-1]
        te_init = (Rie * ta[t-1] + Rea * ti_init) / (Rie + Rea)
        ti[t] = ti_init + dt / Ci * ((ta[t-1] - ti_init) / Rwin + (te[t-1] - ti_init) / Rie + Q_h[t-1] + Ai * I_s[t-1] + a*Q_int[t-1])
        te[t] = te_init + dt / Ce * ((ti_init - te[t-1]) / Rie + (ta[t-1] - te_init) / Rea + Ae * I_s[t-1])
    return ti

def eval_R3C2_multistep(inputs, init_ti,params):
    Rie = params['Rie']
    Rea = params['Rea']
    Rwin = params['Rwin']
    Ci = params['Ci']
    Ce = params['Ce']
    Ai = params['Ai']
    Ae = params['Ae']
    a = params['a']

    ta = inputs[:, 0] # ambient temperature
    Q_h = inputs[:, 1] # heat gain delivered by heater
    I_s = inputs[:, 2] # global solar irradiation
    Q_int = inputs[:, 3]  # internal heat gain

    ti = np.zeros(len(inputs))
    te = np.zeros(len(inputs))

    # Initial temperatures
    ti[0] = init_ti
    te[0] = (Rie * ta[0] + Rea * ti[0]) / (Rie + Rea)
    dt = 3600
    # Loop for calculating all temperatures
    for t in range(1, len(inputs)):

        ti[t] = ti[t - 1] + dt / Ci * ( (ta[t-1]-ti[t-1])/Rwin+  (te[t - 1] - ti[t - 1]) / Rie + Q_h[t - 1] + Ai * I_s[t - 1]+a*Q_int[t - 1] )
        te[t] = te[t - 1] + dt / Ce * ((ti[t - 1] - te[t - 1]) / Rie + (ta[t - 1] - te[t - 1]) / Rea + Ae * I_s[t - 1])

    return ti


df = pd.read_csv('evaluation_data.csv')


last_7_days_inputs = df[['Ambient_Temperature', 'Q_Heat_Pump', 'Solar_Radiation', 'Occupancy_Gain']].values[-7*24-2:-1]
last_7_days_output = df['Indoor_Temperature'].values[-7*24-2:-1]


with open('R3C2_params.json', 'r') as file:
    params = json.load(file)


predicted_ti_last_7_days = eval_R3C2_onestep(last_7_days_inputs, last_7_days_output, params)


all_days_inputs = df[['Ambient_Temperature', 'Q_Heat_Pump', 'Solar_Radiation', 'Occupancy_Gain']].values
all_days_output = df['Indoor_Temperature'].values


with open('R3C2_params.json', 'r') as file:
    params = json.load(file)


all_predicted_ti = eval_R3C2_onestep(all_days_inputs, all_days_output, params)


rmse = np.sqrt(mean_squared_error(all_days_output, all_predicted_ti))
cv_rmse = rmse / np.mean(all_days_output) * 100  
mae = mean_absolute_error(all_days_output, all_predicted_ti)
mape = np.mean(np.abs(all_days_output - all_predicted_ti) / all_days_output) * 100  

metrics = {'RMSE': rmse, 'CV-RMSE': cv_rmse, 'MAE': mae, 'MAPE': mape}
metric_labels = ['RMSE', 'CV-RMSE (%)', 'MAE', 'MAPE (%)'] 

file_path = 'performance_metrics.json'


with open(file_path, 'w') as file:
    json.dump(metrics, file, indent=4)

date_range = pd.date_range(start='2020-03-24', periods=169, freq='H')
# Convert the date range to the desired string format
date_labels = date_range.strftime('%b %d')

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.4)
label_size=14
legend_size=13

ax0 = fig.add_subplot(gs[0, :])  
ax0.plot(last_7_days_output[1:], label='Measurement',color='black',lw=1.5)
ax0.plot(predicted_ti_last_7_days[1:], label='3R2C Onestep',color='#3682be')
ax0.set_xticks(range(0, len(date_labels), 24))  # Set x-ticks to be every 24 hours
ax0.set_xticklabels(date_labels[::24], rotation=0, ha='center')  # Set x-tick labels with a 45 degree rotation
ax0.set_xlabel('Day of Year',fontsize=label_size)  
ax0.set_ylabel('Temperature (°C)',fontsize=label_size)  
ax0.legend(frameon=False,fontsize=legend_size, ncol=2)

data = {
    'DateTime': date_range[:-1],
    'Ti': last_7_days_output[1:],
    'Ti_pred': predicted_ti_last_7_days[1:]
}
df = pd.DataFrame(data)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.to_csv('weekly_temperature_predictions.csv', index=False)

bar_width = 0.1


ax1 = fig.add_subplot(gs[1, 0])
scale_dependent_metrics = {'MAE': mae, 'RMSE': rmse}
bars1 = ax1.bar(scale_dependent_metrics.keys(), scale_dependent_metrics.values(), color=['#3682be', '#3682be'], width=bar_width)
ax1.set_ylim([0, max(scale_dependent_metrics.values()) * 1.2])  
ax1.set_ylabel('Value (°C)',fontsize=label_size)  
ax1.set_xlabel('Scale-dependent Metrics',fontsize=label_size) 

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom')
ax1.set_xlim(-0.5, len(scale_dependent_metrics)-0.5)

ax2 = fig.add_subplot(gs[1, 1])
scale_independent_metrics = {'MAPE': mape,'CV-RMSE': cv_rmse }
bars2 = ax2.bar(scale_independent_metrics.keys(), scale_independent_metrics.values(), color=['#3682be', '#3682be'],width=bar_width)
# ax2.set_title('b2) Scale-independent Metrics')
ax2.set_ylim([0, max(scale_independent_metrics.values()) * 1.2])  
ax2.set_ylabel('Value (%)',fontsize=label_size) 
ax2.set_xlabel('Scale-independent Metrics',fontsize=label_size)  

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}%", ha='center', va='bottom')


ax2.set_xlim(-0.5, len(scale_independent_metrics)-0.5)

plt.tight_layout()
plt.show()


all_predictions = []
actual_targets = []
for i in range(len(all_days_inputs)-25):

    predictions_24_steps=eval_R3C2_multistep(all_days_inputs[i:i+25,:],all_days_output[i], params)
    all_predictions.append(predictions_24_steps[1:])
    actual_targets.append(all_days_output[i+1:i+25])
# for i in range(0,240*5,24):
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1,25),all_predictions[i], 'o-',label='Predicted Temperature', color='blue',)
#     plt.plot(range(1,25),actual_targets[i],'o-', label='Actual Temperature', color='orange')
#     plt.title('Comparison of Actual and Predicted Temperatures')
#     plt.xlabel('Time Step')
#     plt.ylabel('Temperature')
#     plt.ylim(19,25)
#     plt.legend()
#     plt.show()

mean_errors = []
stds = []
for i in range(24):
    errors = [abs(all_predictions[j][i] - actual_targets[j][i]) for j in range(len(all_predictions))]
    mean_errors.append(np.mean(errors))
    stds.append(np.std(errors))
multistep_metrics = {
    "mean_errors": mean_errors,
    "stds": stds
}
with open('multistep_mae_metrics.json', 'w') as json_file:
    json.dump(multistep_metrics, json_file, indent=4)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 25), mean_errors, label='Mean Absolute Error', color='blue')
plt.fill_between(range(1, 25), np.array(mean_errors) - np.array(stds), np.array(mean_errors) + np.array(stds), color='blue', alpha=0.2)
plt.title('Mean Absolute Error and Variance Over Prediction Horizon')
plt.xlabel('Prediction Horizon (Time Steps)')
plt.ylabel('Error')
plt.xticks(range(1, 25))
plt.legend()
plt.show()
