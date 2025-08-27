import pandas as pd
import requests
import json
import time
import numpy as np
from tqdm import trange

def Collect_data(
    BASE_URL = "http://127.0.0.1:80",
    CASE_NAME = "bestest_hydronic_heat_pump",
    start_time=31 * 24 * 3600 - 3600,  # seconds
    warmup_period = 7 * 24 * 3600,
    time_period= (60 *24 + 1) * 3600,
    step=3600,
    horizon=3600,
    interval = 3600,
    Action = None,
    output_csv = None,
):

    # 1. select testcase and get test_id
    resp = requests.post(f"{BASE_URL}/testcases/{CASE_NAME}/select")
    testid = resp.json()["testid"]

    # 2. initial para
    requests.put(f"{BASE_URL}/initialize/{testid}", json={
        "start_time": start_time,
        "warmup_period": warmup_period
    })

    # 3. set time step
    requests.put(f"{BASE_URL}/step/{testid}", json={"step": step})
    data = []
    point_names = ["TDryBul", "HGloHor", "InternalGainsRad[1]", "Occupancy[1]"]  # forecast_points

    # Collect data  two months
    iter_num = int(time_period/step)
    for i in trange(iter_num, desc="Simulating"):
        if Action is not None:
            payload = requests.post(f"{BASE_URL}/advance/{testid}",
                json={
                    "oveHeaPumY_activate": 1,
                    "oveHeaPumY_u": Action,
                }).json()["payload"]
        else:
            payload = requests.post(f"{BASE_URL}/advance/{testid}").json()["payload"]

        forecast = requests.put(f"{BASE_URL}/forecast/{testid}",
            json={
                "point_names": point_names,
                "horizon": horizon,
                "interval": interval
            }).json()["payload"]

        # collect data including outputs and forecasts
        if output_csv is not None:
            row = {
                "Time": payload.get("time") / 3600 / 24,  # day
                "Action": payload.get("oveHeaPumY_u"),
                "Indoor_Temperature": payload.get("reaTZon_y") - 273.15 if payload.get("reaTZon_y") else None,
                "Ambient_Temperature": payload.get("weaSta_reaWeaTDryBul_y") - 273.15 if payload.get("weaSta_reaWeaTDryBul_y") else None,
                "Solar_Radiation": payload.get("weaSta_reaWeaHDirNor_y"),
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

    if output_csv is not None:
        columns = [
            "Time","Action","Indoor_Temperature","Ambient_Temperature","Solar_Radiation",
            "Set_Heat_Temperature","Set_Cool_Temperature","Cop","Heat_Pump_Power",
            "Emission_circuit_pump_electrical_power","Electrical_power_heat_pump_evaporator_fan",
            "Occupants","Occupancy_Gain","reaQHeaPumEva_y",
            "Q_Heat_Pump","reaQHeaPumCon_y"
        ]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_csv, index=False)

    # 4. stop test_id，release source
    resp = requests.put(f"{BASE_URL}/stop/{testid}")
    print("stop：", resp.status_code, resp.text)
    print(f"system disturbances saved to {output_csv}")
    return 0


Actions = [0, 1, None]
for Action in Actions:
    if Action is None:
        Action_name = "Baseline"
    elif Action == 0:
        Action_name = "Action_0"
    elif Action == 1:
        Action_name = "Action_1"
    output_csv = f"{Action_name}" + "_dataset.csv"
    Collect_data(Action=Action,  output_csv=output_csv)
