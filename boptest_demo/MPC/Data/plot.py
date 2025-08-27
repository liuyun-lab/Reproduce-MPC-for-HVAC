import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# 文件名与标签、颜色对应
files = {
    'Baseline_peak': ('Disturbances_Baseline_peak_w=2.2.csv', 'Baseline', 'green'),
    'RC-MPC_peak': ('Disturbances_rc_mpc_peak_w=2.2.csv', 'RC-MPC', 'dodgerblue'),
    'ANN-MPC_peak': ('Disturbances_ann_mpc_peak_w=2.2.csv', 'ANN-MPC', 'orange'),
    'Baseline_typical': ('Disturbances_Baseline_typical_w=2.2.csv', 'Baseline', 'green'),
    'RC-MPC_typical': ('Disturbances_rc_mpc_typical_w=2.2.csv', 'RC-MPC', 'dodgerblue'),
    'ANN-MPC_typical': ('Disturbances_ann_mpc_typical_w=2.2.csv', 'ANN-MPC', 'orange'),
}

# 读取所有数据
data = {k: pd.read_csv(v[0]) for k, v in files.items()}

# 创建大图 - 使用GridSpec控制布局
fig = plt.figure(figsize=(18, 18))
gs = GridSpec(4, 2, figure=fig, height_ratios=[1.3, 1, 1, 1.2], hspace=0.20, wspace=0.10)

# 设置全局字体
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# ===== 第1行：室内温度 =====
ax_temp_peak = fig.add_subplot(gs[0, 0])
ax_temp_typical = fig.add_subplot(gs[0, 1])

# Peak 室内温度
for key, (fname, label, color) in files.items():
    if 'peak' in key:
        df = data[key]
        ax_temp_peak.plot(df['Time'], df['Indoor_Temperature'], label=label, color=color, linewidth=1.5)

# 添加设定温度
df = data['Baseline_peak']
ax_temp_peak.plot(df['Time'], df['Set_Heat_Temperature'], '-', label='Heat setpoint', color='gray', linewidth=1)
ax_temp_peak.plot(df['Time'], df['Set_Cool_Temperature'], '-', label='Cool setpoint', color='gray', linewidth=1)

# 添加价格轴
ax_price_peak = ax_temp_peak.twinx()
ax_price_peak.plot(df['Time'], df['Price'], color='purple', linestyle='-', alpha=0.7, label='Electricity Price')
# ax_price_peak.set_ylabel('Price (EUR/kWh)', color='purple', fontsize=12)
ax_price_peak.tick_params(axis='y', labelcolor='purple')
ax_price_peak.set_ylim(0, 0.32)
ax_price_peak.set_yticks([])

# 设置图表属性
ax_temp_peak.set_title('(a) Peak Heating Period', pad=14)
ax_temp_peak.set_ylabel('Operative Temperature (°C)', fontsize=14)
ax_temp_peak.set_ylim(14.5, 30.5)
ax_temp_peak.grid(True, linestyle='--', alpha=0.5)

# Typical 室内温度
for key, (fname, label, color) in files.items():
    if 'typical' in key:
        df = data[key]
        ax_temp_typical.plot(df['Time'], df['Indoor_Temperature'], label=label, color=color, linewidth=1.5)

# 添加设定温度
df = data['Baseline_typical']
ax_temp_typical.plot(df['Time'], df['Set_Heat_Temperature'], '-', label='Heat setpoint', color='gray', linewidth=1)
ax_temp_typical.plot(df['Time'], df['Set_Cool_Temperature'], '-', label='Cool setpoint', color='gray', linewidth=1)

# 添加价格轴
ax_price_typical = ax_temp_typical.twinx()
ax_price_typical.plot(df['Time'], df['Price'], color='purple', linestyle='-', alpha=0.7, label='Electricity Price')
ax_price_typical.set_ylabel('Price (EUR/kWh)', color='purple', fontsize=12)
ax_price_typical.tick_params(axis='y', labelcolor='purple')
ax_price_typical.set_ylim(0, 0.32)

# 设置图表属性
ax_temp_typical.set_title('(b) Typical Heating Period', pad=14)
ax_temp_typical.set_ylim(14.5, 30.5)
ax_temp_typical.grid(True, linestyle='--', alpha=0.5)
ax_temp_typical.set_yticks([])

# ===== 第2行：热不适 =====
ax_dis_peak = fig.add_subplot(gs[1, 0])
ax_dis_typical = fig.add_subplot(gs[1, 1])

# Peak 热不适
for key, (fname, label, color) in files.items():
    if 'peak' in key:
        df = data[key]
        ax_dis_peak.plot(df['Time'], df['Thermal discomfort'], label=label, color=color, linewidth=1.5)

# ax_dis_peak.set_title('(c) Thermal Discomfort - Peak Heating Period', pad=12)
ax_dis_peak.set_ylabel('Thermal Discomfort (Kh/zone)', fontsize=14)
ax_dis_peak.set_ylim(-0.2, 10)
ax_dis_peak.grid(True, linestyle='--', alpha=0.5)

# Typical 热不适
for key, (fname, label, color) in files.items():
    if 'typical' in key:
        df = data[key]
        ax_dis_typical.plot(df['Time'], df['Thermal discomfort'], label=label, color=color, linewidth=1.5)

# ax_dis_typical.set_title('(d) Thermal Discomfort - Typical Heating Period', pad=12)
ax_dis_typical.set_ylim(-0.2, 10)
ax_dis_typical.grid(True, linestyle='--', alpha=0.5)
ax_dis_typical.set_yticks([])

# ===== 第3行：能源成本 =====
ax_cost_peak = fig.add_subplot(gs[2, 0])
ax_cost_typical = fig.add_subplot(gs[2, 1])

# Peak 能源成本
for key, (fname, label, color) in files.items():
    if 'peak' in key:
        df = data[key]
        ax_cost_peak.plot(df['Time'], df['Energy cost'], label=label, color=color, linewidth=1.5)

# ax_cost_peak.set_title('(e) Energy Cost - Peak Heating Period', pad=12)
ax_cost_peak.set_ylabel('Operational Cost (€/m²)', fontsize=14)
ax_cost_peak.set_ylim(-0.01, 0.8)
ax_cost_peak.grid(True, linestyle='--', alpha=0.5)


# Typical 能源成本
for key, (fname, label, color) in files.items():
    if 'typical' in key:
        df = data[key]
        ax_cost_typical.plot(df['Time'], df['Energy cost'], label=label, color=color, linewidth=1.5)

# ax_cost_typical.set_title('(f) Energy Cost - Typical Heating Period', pad=12)
ax_cost_typical.set_ylim(-0.01, 0.8)
ax_cost_typical.grid(True, linestyle='--', alpha=0.5)
ax_cost_typical.set_yticks([])

# ===== 第4行：环境条件 =====
ax_env_peak = fig.add_subplot(gs[3, 0])
ax_env_typical = fig.add_subplot(gs[3, 1])

# Peak 环境条件
for key, (fname, label, color) in files.items():
    if 'peak' in key:
        df = data[key]
        # 环境温度（左侧轴）
        ax_env_peak.plot(df['Time'], df['Ambient_Temperature'], color='blue', linewidth=1.5,
                         label='Ambient Temperature')

# 太阳辐射（右侧轴）
ax_solar_peak = ax_env_peak.twinx()
ax_solar_peak.plot(df['Time'], df['Solar_Radiation'], color='red', linewidth=1.5, label='Solar Radiation')

# 设置图表属性
# ax_env_peak.set_title('(g) Environmental Conditions - Peak Heating Period', pad=12)
ax_env_peak.set_ylabel('Ambient Temp (°C)', color='blue', fontsize=14)
ax_env_peak.tick_params(axis='y', labelcolor='blue')
ax_env_peak.set_ylim(-5, 24)
ax_env_peak.grid(True, linestyle='--', alpha=0.5)

#ax_solar_peak.set_ylabel('Solar Irrad (W/m²)', color='red', fontsize=14)
ax_solar_peak.tick_params(axis='y', labelcolor='red')
ax_solar_peak.set_ylim(-10, 850)
ax_solar_peak.set_yticks([])

# Typical 环境条件
for key, (fname, label, color) in files.items():
    if 'typical' in key:
        df = data[key]
        # 环境温度（左侧轴）
        ax_env_typical.plot(df['Time'], df['Ambient_Temperature'], color='blue', linewidth=1.5,
                            label='Ambient Temperature')

# 太阳辐射（右侧轴）
ax_solar_typical = ax_env_typical.twinx()
ax_solar_typical.plot(df['Time'], df['Solar_Radiation'], color='red', linewidth=1.5, label='Solar Radiation')

# 设置图表属性
# ax_env_typical.set_title('(h) Environmental Conditions - Typical Heating Period', pad=12)
ax_env_typical.set_ylim(-5, 24)
ax_env_typical.grid(True, linestyle='--', alpha=0.5)
ax_env_typical.set_yticks([])

ax_solar_typical.set_ylabel('Solar radiation (W/m²)', color='red', fontsize=14)
ax_solar_typical.tick_params(axis='y', labelcolor='red')
ax_solar_typical.set_ylim(-10, 850)


# ===== 统一处理时间轴 =====
for ax in [ax_temp_peak, ax_temp_typical, ax_dis_peak, ax_dis_typical,
           ax_cost_peak, ax_cost_typical, ax_env_peak, ax_env_typical]:
    # 设置时间格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 仅底部图表显示X轴标签
    if ax not in [ax_env_peak, ax_env_typical]:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time', fontsize=14)

# ===== 添加图例 =====
# 控制器图例（前6个图表）
handles, labels = ax_temp_peak.get_legend_handles_labels()
# 过滤掉设定温度线（只保留控制器线）
controller_handles = [h for h, l in zip(handles, labels) if l not in ['Heat setpoint', 'Cool setpoint']]
controller_labels = [l for l in labels if l not in ['Heat setpoint', 'Cool setpoint']]

all_handles = [
    *controller_handles,  # 控制器句柄（展开列表）
    handles[labels.index('Heat setpoint')],  # 加热设定点
    ax_env_peak.get_lines()[0],  # 环境温度
    ax_solar_peak.get_lines()[0],  # 太阳辐射
    ax_price_peak.get_lines()[0]   # 电价
]

all_labels = [
    *controller_labels,  # 控制器标签（展开列表）
    'Comfort setpoint',
    'Ambient Temperature',
    'Solar Radiation',
    'Electricity Price'
]

# 创建统一图例（通过ncol实现分组效果）
fig.legend(
    handles=all_handles,
    labels=all_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.92),  # 调整垂直位置
    ncol=7,  # 总列数（控制器3列 + 设定点2列 + 环境2列 + 电价1列）
    frameon=True,
    title_fontsize=14,
    fontsize=12
)

# ===== 最终调整和保存 =====
plt.subplots_adjust(top=0.85, bottom=0.08)
plt.suptitle("MPC Controller Performance Analysis", fontsize=20, y=0.95)
# plt.tight_layout(rect=[0, 0, 1, 0.85])  # 为图例留出空间

# 保存高分辨率图像
plt.savefig("combined_analysis.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()













# # 文件名与标签、颜色对应
# files = {
#     'Baseline_peak':    ('Disturbances_Baseline_peak_w=2.2.csv',    'Baseline',   'green'),
#     'RC-MPC_peak':      ('Disturbances_rc_mpc_peak_w=2.2.csv',      'RC-MPC',     'dodgerblue'),
#     'ANN-MPC_peak':     ('Disturbances_ann_mpc_peak_w=2.2.csv',     'ANN-MPC',    'orange'),
#     'Baseline_typical': ('Disturbances_Baseline_typical_w=2.2.csv', 'Baseline',   'green'),
#     'RC-MPC_typical':   ('Disturbances_rc_mpc_typical_w=2.2.csv',   'RC-MPC',     'dodgerblue'),
#     'ANN-MPC_typical':  ('Disturbances_ann_mpc_typical_w=2.2.csv',  'ANN-MPC',    'orange'),
# }
#
# # 读取所有数据
# data = {k: pd.read_csv(v[0]) for k, v in files.items()}
#
# fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
#
# # ----------- Peak -----------
# ax = axs[0]
# for key, (fname, label, color) in files.items():
#     if 'peak' in key:
#         df = data[key]
#         ax.plot(df['Time'], df['Indoor_Temperature'], label=label, color=color)
# # setpoint - 加热设定温度
# df = data['Baseline_peak']
# ax.plot(df['Time'], df['Set_Heat_Temperature'], label='Heat setpoint', color='gray', linewidth=1)
# ax.plot(df['Time'], df['Set_Cool_Temperature'], label='Cool setpoint', color='gray', linewidth=1)
# # 双y轴画价格
# ax2 = ax.twinx()
# ax2.plot(df['Time'], df['Price'], color='mediumpurple', linestyle=':', alpha=0.7)
# ax2.set_ylabel('')
# ax2.tick_params(axis='y', labelcolor='mediumpurple')
# ax.set_title('Peak')
# ax.set_xlabel('Time')
# ax.set_ylabel('Operative temperature (°C)')
# ax.set_ylim(14.5, 30.5)
# ax2.set_ylim(0, 0.3)
# ax2.set_yticks([])
#
# # ----------- Typical -----------
# ax = axs[1]
# for key, (fname, label, color) in files.items():
#     if 'typical' in key:
#         df = data[key]
#         ax.plot(df['Time'], df['Indoor_Temperature'], label=label, color=color)
# df = data['Baseline_typical']
# ax.plot(df['Time'], df['Set_Heat_Temperature'], label='Heat setpoint', color='gray', linewidth=1)
# ax.plot(df['Time'], df['Set_Cool_Temperature'], label='Cool setpoint', color='gray', linewidth=1)
# ax2 = ax.twinx()
# ax2.plot(df['Time'], df['Price'], color='mediumpurple', linestyle=':', alpha=0.7)
# ax2.set_ylabel('Price (EUR/kWh)', color='mediumpurple')
# ax2.tick_params(axis='y', labelcolor='mediumpurple')
# ax.set_title('Typical')
# ax.set_xlabel('Time')
# ax.set_ylim(14.5, 30.5)
# ax2.set_ylim(0, 0.3)
# ax.set_yticks([])
#
# # 图例
# lines, labels = axs[0].get_legend_handles_labels()
# fig.legend(lines, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.08))
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
#
#
#
# fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
#
# # ----------- Peak -----------
# ax = axs[0]
# for key, (fname, label, color) in files.items():
#     if 'peak' in key:
#         df = data[key]
#         ax.plot(df['Time'], df['Thermal discomfort'], label=label, color=color)
#
# ax.set_title('Peak')
# ax.set_xlabel('Time')
# ax.set_ylabel('Thermal discomfort (Kh/zone)')
# ax.set_ylim(-0.2, 10)
#
# # ----------- Typical -----------
# ax = axs[1]
# for key, (fname, label, color) in files.items():
#     if 'typical' in key:
#         df = data[key]
#         ax.plot(df['Time'], df['Thermal discomfort'], label=label, color=color)
#
# ax.set_title('Typical')
# ax.set_xlabel('Time')
# ax.set_ylim(-0.2, 10)
#
# lines, labels = axs[0].get_legend_handles_labels()
# fig.legend(lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08))
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
#
#
#
# fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
#
# # ----------- Peak -----------
# ax = axs[0]
# for key, (fname, label, color) in files.items():
#     if 'peak' in key:
#         df = data[key]
#         ax.plot(df['Time'], df['Energy cost'], label=label, color=color)
#
# ax.set_title('Peak')
# ax.set_xlabel('Time')
# ax.set_ylabel('Operational cost (€/m²)')
# ax.set_ylim(-0.01, 0.8)
#
# # ----------- Typical -----------
# ax = axs[1]
# for key, (fname, label, color) in files.items():
#     if 'typical' in key:
#         df = data[key]
#         ax.plot(df['Time'], df['Energy cost'], label=label, color=color)
#
# ax.set_title('Typical')
# ax.set_xlabel('Time')
# ax.set_ylim(-0.01, 0.8)
#
# lines, labels = axs[0].get_legend_handles_labels()
# fig.legend(lines, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08))
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
#
#
# fig, axs = plt.subplots(1, 2, figsize=(16, 5))
#
# # ----------- Peak -----------
# ax = axs[0]
# for key, (fname, label, color) in files.items():
#     if 'peak' in key:
#         df = data[key]
#         ax.plot(df['Time'], df['Ambient_Temperature'], color='cyan', linewidth=1)
#
# ax.set_ylabel('Ambient temperature (°C)', color='cyan')
# ax.tick_params(axis='y', labelcolor='cyan')
# ax.set_ylim(-5, 24)
#
# ax2 = ax.twinx()
# ax2.plot(df['Time'], df['Solar_Radiation'], color='red', linewidth=1)
# ax2.set_ylabel('')
# ax2.tick_params(axis='y', labelcolor='red')
# ax2.set_ylim(-10, 850)
# ax2.set_yticks([])
#
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
#
# ax.set_title('(a) Peak heating period')
#
# # ----------- Typical -----------
# ax = axs[1]
# for key, (fname, label, color) in files.items():
#     if 'typical' in key:
#         df = data[key]
#         ax.plot(df['Time'], df['Ambient_Temperature'], color='cyan', linewidth=1)
#
# ax.set_ylabel('')
# ax.tick_params(axis='y', labelcolor='cyan')
# ax.set_ylim(-5, 24)
# ax.set_yticks([])
#
# # 右y轴：太阳辐射（红色）
# ax2 = ax.twinx()
# ax2.plot(df['Time'], df['Solar_Radiation'], color='red', linewidth=1)
# ax2.set_ylabel('Solar irradiation (W)', color='red')
# ax2.tick_params(axis='y', labelcolor='red')
# ax2.set_ylim(-10, 850)
#
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
# ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
#
# ax.set_title('(b) Typical heating period')
#
# plt.tight_layout()
# plt.show()
# fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)