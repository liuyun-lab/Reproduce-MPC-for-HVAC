import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Union
import matplotlib.pyplot as plt
import random
import torch.optim as optim
from torch.autograd import Variable
# import joblib
import json
import requests
import argparse
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MLP(nn.Module):
    def __init__(self,input_size,input_normalization_minimum,input_normalization_maximum,output_low_limit,output_high_limit):
        super(MLP, self).__init__()


        self.input_normalization_minimum = torch.tensor(input_normalization_minimum, dtype=torch.float32)
        self.input_normalization_maximum = torch.tensor(input_normalization_maximum, dtype=torch.float32)
        self.output_high_limit=torch.tensor(output_high_limit, dtype=torch.float32)
        self.output_low_limit=torch.tensor(output_low_limit, dtype=torch.float32)

        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def load_model(self):
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'ann_model.pth')

        self.load_state_dict(torch.load(model_path)["model_state_dict"])
    def normalize_input(self, data):
        normalized_data = (data - self.input_normalization_minimum) / \
                          (self.input_normalization_maximum - self.input_normalization_minimum)

        return normalized_data
    def normalize_output(self, output):
        normalize_output = (output - self.output_low_limit) / \
                          (self.output_high_limit - self.output_low_limit)

        return normalize_output
    def inverse_normalize_data(self, normalized_data_tensor):
        original_data = normalized_data_tensor * (self.output_high_limit - self.output_low_limit) + self.output_low_limit
        return original_data

    def forward(self, x):
        x_normalized = self.normalize_input(x)
        return self.normalize_output(self.layers(x_normalized))
    def step(self, x):
        x_normalized = self.normalize_input(x)
        return self.layers(x_normalized)

    def step_with_u(self,x,u, aux):
        aux_part1, aux_part2 = aux[0:1], aux[1:]

        if u.dim() == 0:
            u = u.unsqueeze(0)
        if x.dim() == 0:
            x = x.unsqueeze(0)
        input=torch.cat((aux_part1, u, x,aux_part2), dim=0)
        return self.step(input)


model = MLP(5, [-9.1000, 0.0000, 4.9050, 0.0000, 0.0000],
          [29.85,  14187.3652, 48.78613256, 219, 862],4.904992,48.78613256)

model.load_model()




df = pd.read_csv('evaluation_data.csv')
df['Indoor_Temperature_lag1']=df['Indoor_Temperature'].shift(1)
df_ann=df[['Indoor_Temperature','Ambient_Temperature','Q_Heat_Pump','Indoor_Temperature_lag1','Occupancy_Gain','Solar_Radiation']].dropna()
test_target=df_ann['Indoor_Temperature']
test_features = df_ann.drop('Indoor_Temperature', axis=1)
last_7_days_X_test = torch.tensor(test_features.values[-7*24-1:-1], dtype=torch.float32)
last_7_days_output = torch.tensor(test_target.values[-7*24-1:-1], dtype=torch.float32)

model.eval()
with torch.no_grad():
    predicted_ti_last_7_days = model.step(last_7_days_X_test)




last_7_days_output = last_7_days_output.numpy()
predicted_ti_last_7_days = predicted_ti_last_7_days.squeeze().numpy()






df = pd.read_csv('evaluation_data.csv')
df['Indoor_Temperature_lag1']=df['Indoor_Temperature'].shift(1)
df_ann=df[['Indoor_Temperature','Ambient_Temperature','Q_Heat_Pump','Indoor_Temperature_lag1','Occupancy_Gain','Solar_Radiation']].dropna()
test_target=df_ann['Indoor_Temperature']
test_features_all = df_ann.drop('Indoor_Temperature', axis=1)
X_test = torch.tensor(test_features_all.values, dtype=torch.float32)
all_days_output = torch.tensor(test_target.values, dtype=torch.float32)

model.eval()
with torch.no_grad():
    all_predicted_ti = model.step(X_test)




all_days_output = all_days_output.numpy()
all_predicted_ti = all_predicted_ti.squeeze().numpy()





rmse = np.sqrt(mean_squared_error(all_days_output, all_predicted_ti))
cv_rmse = rmse / np.mean(all_days_output) * 100  # 转换为百分比
mae = mean_absolute_error(all_days_output, all_predicted_ti)
mape = np.mean(np.abs(all_days_output - all_predicted_ti) / all_days_output) * 100  # 转换为百分比



metrics = {'RMSE': float(rmse), 'CV-RMSE': float(cv_rmse), 'MAE': float(mae), 'MAPE': float(mape)}
metric_labels = ['RMSE', 'CV-RMSE (%)', 'MAE', 'MAPE (%)']  # 标签，区分百分比

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
ax0.plot(last_7_days_output, label='Measurement',color='black',lw=1.5)
ax0.plot(predicted_ti_last_7_days, label='3R2C Onestep',color='#3682be')
ax0.set_xticks(range(0, len(date_labels), 24))  # Set x-ticks to be every 24 hours
ax0.set_xticklabels(date_labels[::24], rotation=0, ha='center')  # Set x-tick labels with a 45 degree rotation
ax0.set_xlabel('Day of Year',fontsize=label_size)  # X轴标签
ax0.set_ylabel('Temperature (°C)',fontsize=label_size)  # Y轴标签
ax0.legend(frameon=False,fontsize=legend_size, ncol=2)


data = {
    'DateTime': date_range[:-1],
    'Ti': last_7_days_output,
    'Ti_pred': predicted_ti_last_7_days
}
df = pd.DataFrame(data)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.to_csv('weekly_temperature_predictions.csv', index=False)




from tqdm import tqdm
all_predictions = []
actual_targets = []
for i in tqdm(range(len(X_test)-24)):
    features_24_steps = test_features_all.iloc[i:i+24].copy()
    predictions_24_steps = []
    for j in range(24):
        current_input= features_24_steps.iloc[j:j+1].values
        current_input_tensor=torch.tensor(current_input, dtype=torch.float32)
        with torch.no_grad():
                predicted_temperature = model.step(current_input_tensor).item()

        predictions_24_steps.append(predicted_temperature)
        if j < 23:
            features_24_steps.iloc[j+1, 2] = predicted_temperature
    all_predictions.append(predictions_24_steps)
    actual_targets.append(test_target.iloc[i:i+24].to_list())
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



















bar_width = 0.1

# 添加底部左侧的柱状图子图 (尺度依赖指标)
ax1 = fig.add_subplot(gs[1, 0])
scale_dependent_metrics = {'MAE': mae, 'RMSE': rmse}
bars1 = ax1.bar(scale_dependent_metrics.keys(), scale_dependent_metrics.values(), color=['#3682be', '#3682be'], width=bar_width)
ax1.set_ylim([0, max(scale_dependent_metrics.values()) * 1.2])  # 设置Y轴最大值
ax1.set_ylabel('Value (°C)',fontsize=label_size)  # Y轴标签
ax1.set_xlabel('Scale-dependent Metrics',fontsize=label_size)  # X轴标签
# 添加每个柱子上的数值标签
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center', va='bottom')
ax1.set_xlim(-0.5, len(scale_dependent_metrics)-0.5)
# 添加底部右侧的柱状图子图 (尺度独立指标)·
ax2 = fig.add_subplot(gs[1, 1])
scale_independent_metrics = {'MAPE': mape,'CV-RMSE': cv_rmse }
bars2 = ax2.bar(scale_independent_metrics.keys(), scale_independent_metrics.values(), color=['#3682be', '#3682be'],width=bar_width)
# ax2.set_title('b2) Scale-independent Metrics')
ax2.set_ylim([0, max(scale_independent_metrics.values()) * 1.2])  # 设置Y轴最大值
ax2.set_ylabel('Value (%)',fontsize=label_size)  # Y轴标签，百分比
ax2.set_xlabel('Scale-independent Metrics',fontsize=label_size)  # X轴标签
# 添加每个柱子上的数值标签
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}%", ha='center', va='bottom')

# 为了在图的底部添加尺度依赖和尺度独立指标的描述，我们使用figtext
# ax1.text(0.5, -0.5, 'Scale-dependent Metrics', transform=ax1.transAxes,
#          size=16, ha='center')  # 添加(b1)标记，居中
# ax2.text(0.5, -0.5, 'Scale-independent Metrics', transform=ax2.transAxes,
#          size=16, ha='center')  # 添加(b2)标记，居中
ax2.set_xlim(-0.5, len(scale_independent_metrics)-0.5)

plt.tight_layout()
plt.show()


