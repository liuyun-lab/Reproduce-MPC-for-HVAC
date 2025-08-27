import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def pareto_frontier(X, Y, maxX = False, maxY = False):
    myList = sorted([[X[i], Y[i]] for i in range(len(X))], reverse=maxX)
    p_front = [myList[0]]
    for pair in myList[1:]:
        if maxY:
            if pair[1] >= p_front[-1][1]:
                p_front.append(pair)
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair)
    return np.array(p_front)

# Load data for all four cases
data_files = {
    'RC_peak': 'RC/KPI/KPI_peak.json',
    'ANN_peak': 'ANN/KPI/KPI_peak.json',
    'RC_typical': 'RC/KPI/KPI_typical.json',
    'ANN_typical': 'ANN/KPI/KPI_typical.json'
}

data = {}
for name, path in data_files.items():
    with open(path, 'r', encoding='utf-8') as f:
        data[name] = json.load(f)

# Create figure with 2x2 grid
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1],
              wspace=0.15, hspace=0.2)

# Reference points for markers
markers = {
    'peak': (0.91, 8.4),
    'typical': (0.41, 9.4)
}

# Plot each subfigure
for i, (model, day_type) in enumerate([('RC', 'peak'), ('ANN', 'peak'),
                                       ('RC', 'typical'), ('ANN', 'typical')]):
    key = f"{model}_{day_type}"
    ax = fig.add_subplot(gs[i // 2, i % 2])  # 2x2 grid positioning

    # Extract data
    energy_cost = np.array([d["Energy cost"] for d in data[key]])
    thermal_discomfort = np.array([d["Thermal discomfort"] for d in data[key]])
    weights = np.array([d["weight"] for d in data[key]])

    # Compute Pareto frontier
    pareto = pareto_frontier(energy_cost, thermal_discomfort, maxX=False, maxY=False)

    # Create scatter plot
    sc = ax.scatter(energy_cost, thermal_discomfort, c=weights,
                    cmap='rainbow', s=80, edgecolors='k')

    # Add Pareto frontier line
    ax.plot(pareto[:, 0], pareto[:, 1], color='r', linewidth=2)

    # Add reference marker
    marker_point = markers[day_type]
    ax.scatter(marker_point[0], marker_point[1], color='green',
               s=100, marker='s', zorder=5)

    # Add text annotation with adjusted position
    ax.text(marker_point[0] - 0.005, marker_point[1] + 20,
            f'({marker_point[0]}, {marker_point[1]})',
            color='green', fontsize=10, ha='center', va='bottom')

    # Set labels and grid
    # ax.set_ylabel('Thermal discomfort', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f'{model} Model - {"Peak" if day_type == "peak" else "Typical"} Day',
                 fontsize=14, pad=12)
# Add common colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label('Weight', fontsize=12)

# plt.suptitle('Energy Cost vs Thermal Discomfort with Pareto Frontiers',
#              fontsize=16, y=0.95)

fig.text(0.5, 0.05, 'Operational cost (€/m²)', ha='center', va='center', fontsize=20)
fig.text(0.05, 0.5, 'Thermal discomfort (Kh/zone)', ha='center', va='center', rotation='vertical', fontsize=20)

#plt.tight_layout(rect=[0.03, 0.03, 0.92, 0.95])  # Make room for colorbar and suptitle
plt.subplots_adjust(bottom=0.1)
plt.show()





# with open('RC/KPI/KPI_peak.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
# energy_cost = np.array([d["Energy cost"] for d in data])
# thermal_discomfort = np.array([d["Thermal discomfort"] for d in data])
# weight = np.array([d["weight"] for d in data])
#
# pareto = pareto_frontier(energy_cost, thermal_discomfort, maxX=False, maxY=False)
#
# plt.figure(figsize=(8,6))
# sc = plt.scatter(energy_cost, thermal_discomfort, c=weight, cmap='rainbow', s=80, edgecolors='k')
# plt.plot(pareto[:,0], pareto[:,1], color='r')
# plt.scatter(0.91, 8.4, color='green', s=100, marker='s')
# plt.text(0.91-0.005, 8.4+20, '(0.91, 8.4)', color='green', fontsize=10, ha='center', va='bottom')
# plt.xlabel('Energy cost')
# plt.ylabel('Thermal discomfort')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.colorbar(sc, label='Weight')
# plt.tight_layout()
# plt.show()
#
# with open('ANN/KPI/KPI_peak.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# energy_cost = np.array([d["Energy cost"] for d in data])
# thermal_discomfort = np.array([d["Thermal discomfort"] for d in data])
# weight = np.array([d["weight"] for d in data])
#
# pareto = pareto_frontier(energy_cost, thermal_discomfort, maxX=False, maxY=False)
#
# plt.figure(figsize=(8,6))
# sc = plt.scatter(energy_cost, thermal_discomfort, c=weight, cmap='rainbow', s=80, edgecolors='k')
# plt.plot(pareto[:,0], pareto[:,1], color='r')
# plt.scatter(0.91, 8.4, color='green', s=100, marker='s')
# plt.text(0.91-0.005, 8.4+20, '(0.91, 8.4)', color='green', fontsize=10, ha='center', va='bottom')
# plt.xlabel('Energy cost')
# plt.ylabel('Thermal discomfort')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.colorbar(sc, label='Weight')
# plt.tight_layout()
# plt.show()
#
# with open('RC/KPI/KPI_typical.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
# energy_cost = np.array([d["Energy cost"] for d in data])
# thermal_discomfort = np.array([d["Thermal discomfort"] for d in data])
# weight = np.array([d["weight"] for d in data])
#
# pareto = pareto_frontier(energy_cost, thermal_discomfort, maxX=False, maxY=False)
#
# plt.figure(figsize=(8,6))
# sc = plt.scatter(energy_cost, thermal_discomfort, c=weight, cmap='rainbow', s=80, edgecolors='k')
# plt.plot(pareto[:,0], pareto[:,1], color='r')
# plt.scatter(0.41, 9.4, color='green', s=100, marker='s')
# plt.text(0.41-0.005, 9.4+20, '(0.41, 9.4)', color='green', fontsize=10, ha='center', va='bottom')
# plt.xlabel('Energy cost')
# plt.ylabel('Thermal discomfort')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.colorbar(sc, label='Weight')
# plt.tight_layout()
# plt.show()
#
#
# with open('ANN/KPI/KPI_typical.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# energy_cost = np.array([d["Energy cost"] for d in data])
# thermal_discomfort = np.array([d["Thermal discomfort"] for d in data])
# weight = np.array([d["weight"] for d in data])
# pareto = pareto_frontier(energy_cost, thermal_discomfort, maxX=False, maxY=False)
# plt.figure(figsize=(8,6))
# sc = plt.scatter(energy_cost, thermal_discomfort, c=weight, cmap='rainbow', s=80, edgecolors='k')
# plt.plot(pareto[:,0], pareto[:,1], color='r')
# plt.scatter(0.41, 9.4, color='green', s=100, marker='s')
# plt.text(0.41-0.005, 9.4+20, '(0.41, 9.4)', color='green', fontsize=10, ha='center', va='bottom')
# plt.xlabel('Energy cost')
# plt.ylabel('Thermal discomfort')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.colorbar(sc, label='Weight')
# plt.tight_layout()
# plt.show()
#
