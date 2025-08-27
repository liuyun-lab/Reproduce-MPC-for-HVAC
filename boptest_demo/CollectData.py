import requests
import pandas as pd
import matplotlib.pyplot as plt

BASE_URL = "http://127.0.0.1:80"
CASE_NAME = "bestest_hydronic_heat_pump"

# 1. select testcase and get test_id
resp = requests.post(f"{BASE_URL}/testcases/{CASE_NAME}/select")
testid = resp.json()["testid"]

# 2. initial para
start_time = 31 * 24 * 3600-3600   #seconds
warmup_period = 7*24*3600
requests.put(f"{BASE_URL}/initialize/{testid}", json={
    "start_time": start_time,
    "warmup_period": warmup_period
})

# 3. set time step
step = 3600
requests.put(f"{BASE_URL}/step/{testid}", json={"step": step})
data = []
point_names = ["TDryBul", "HGloHor", "InternalGainsRad[1]", "Occupancy[1]"]  # forecast_points
horizon = 3600
interval = 3600

# inputs
action_value = 0
advance_data = {
    "oveHeaPumY_activate": 1,
    "oveHeaPumY_u": action_value,
}

# Collect data  two months
time_period = 24*60+1
for i in range(time_period):
    #resp = requests.post(f"{BASE_URL}/advance/{testid}", json=advance_data)
    resp = requests.post(f"{BASE_URL}/advance/{testid}")
    payload = resp.json()["payload"]

    resp_forecast = requests.put(
        f"{BASE_URL}/forecast/{testid}",
        json={
            "point_names": point_names,
            "horizon": horizon,
            "interval": interval
        }
    )
    forecast = resp_forecast.json()["payload"]

    # collect data including outputs and forecasts
    row = {
        "Time": payload.get("time") / 3600 / 24,  # hour
        "Action": payload.get("oveHeaPumY_u"),
        "Indoor_Temperature": payload.get("reaTZon_y") - 273.15 if payload.get("reaTZon_y") else None,
        "Ambient_Temperature": payload.get("weaSta_reaWeaTDryBul_y") - 273.15 if payload.get("weaSta_reaWeaTDryBul_y") else None,

        #"Solar_Radiation": payload.get("weaSta_reaWeaHDirNor_y"),
        "Solar_Radiation": payload.get("weaSta_reaWeaHGloHor_y"),
        #"Solar_Radiation": forecast["HGloHor"][0],

        "Set_Heat_Temperature": payload.get("reaTSetHea_y")- 273.15 if payload.get("reaTSetHea_y") else None,
        "Set_Cool_Temperature": payload.get("reaTSetCoo_y") - 273.15 if payload.get("reaTSetCoo_y") else None,
        "Cop": payload.get("reaCOP_y"),
        "Heat_Pump_Power": payload.get("reaPHeaPum_y"),
        "Emission_circuit_pump_electrical_power": payload.get("reaPPumEmi_y"),
        "Electrical_power_heat_pump_evaporator_fan": payload.get("reaPFan_y"),

        "Occupants": forecast["Occupancy[1]"][0],
        "Occupancy_Gain": forecast["InternalGainsRad[1]"][0],

        "reaQHeaPumEva_y": payload.get("reaQHeaPumEva_y"),
        "Q_Heat_Pump": payload.get("reaQFloHea_y"),
        "reaQHeaPumCon_y": payload.get("reaQHeaPumCon_y"),
    }
    data.append(row)

columns = [
    "Time","Action","Indoor_Temperature","Ambient_Temperature","Solar_Radiation",
    "Set_Heat_Temperature","Set_Cool_Temperature","Cop","Heat_Pump_Power",
    "Emission_circuit_pump_electrical_power","Electrical_power_heat_pump_evaporator_fan",
    "Occupants","Occupancy_Gain","reaQHeaPumEva_y",
    "Q_Heat_Pump","reaQHeaPumCon_y"
]
df = pd.DataFrame(data, columns=columns)
df.to_csv("uPI_day30_to_day90_results2.csv", index=False)


# 4. stop test_id，release source
resp = requests.put(f"{BASE_URL}/stop/{testid}")
print("stop：", resp.status_code, resp.text)



# ## plot
# df = pd.read_csv("uPI_day30_to_day90_results.csv")
#
# plt.figure(figsize=(12, 6))
# plt.plot(df["Time"], df["Indoor_Temperature"], marker='o', color='b', label="Indoor Temperature")
# plt.xlabel("Time (days)")
# plt.ylabel("Indoor Temperature (°C)")
# plt.title("Indoor Temperature vs Time")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

