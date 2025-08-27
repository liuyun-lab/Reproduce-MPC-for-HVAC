import requests
import json
import time
import numpy as np
from tqdm import trange
from boptest_demo.MPC.RC_MPC import rc_mpc

def run_boptest_RC_MPC(
    base_url="http://127.0.0.1:80",
    case_name="bestest_hydronic_heat_pump",
    test_period="peak_heat_day",
    ele_price="highly_dynamic",
    step=3600,
    days=14,
    output_json=None,
    horizon=12,
    weight=5.0,
    epsilon=0.5,
):
    # 1. Select testcase and get test_id
    resp = requests.post(f"{base_url}/testcases/{case_name}/select")
    testid = resp.json()["testid"]

    data = []

    # 2. Choose scenario
    resp = requests.put(f"{base_url}/scenario/{testid}",
                     json={'time_period': test_period,
                           'electricity_price': ele_price}).json()['payload']

    # 3. Set time step
    requests.put(f"{base_url}/step/{testid}", json={"step": step})


    # 5. Get initial temperature
    y = resp
    current_temp = y['time_period']['reaTZon_y'] - 273.15

    time_period = 24 * days
    start_time = time.time()
    for i in trange(time_period, desc="Simulating"):
        # 5.1 Get forecast for next horizon steps
        points = ['TDryBul', 'HDirNor', 'InternalGainsRad[1]', 'Occupancy[1]', 'PriceElectricPowerHighlyDynamic', 'UpperSetp[1]', 'LowerSetp[1]']
        w = requests.put(f"{base_url}/forecast/{testid}",
                         json={'point_names': points, 'horizon': horizon*step, 'interval': step}).json()['payload']
        Ta = np.array(w['TDryBul'][1:]) - 273.15
        Isol = np.array(w['HDirNor'][1:])
        Qint = np.array(w['InternalGainsRad[1]'][1:])
        price = np.array(w['PriceElectricPowerHighlyDynamic'][1:])
        UpperSetp = np.array(w['UpperSetp[1]'][1:]) - 273.15
        LowerSetp = np.array(w['LowerSetp[1]'][1:]) - 273.15

        # Prepare disturbances and setpoints for ann_mpc
        disturbances = {'Ta': Ta, 'Isol': Isol, 'Qint': Qint}
        setpoints = {'Lower': LowerSetp, 'Upper': UpperSetp}

        # 5.2 Call ANN-MPC to get optimal control action
        u = rc_mpc(
            current_temp=current_temp,
            disturbances=disturbances,
            setpoints=setpoints,
            price=price,
            weight=weight,
            horizon=horizon,
            epsilon=epsilon,
        )

        # 5.3 Advance simulation with ANN-MPC control
        y = requests.post(f"{base_url}/advance/{testid}", json={'oveHeaPumY_u': float(u), 'oveHeaPumY_activate': 1}).json()['payload']
        current_temp = y['reaTZon_y'] - 273.15

        # 5.4 Collect KPIs
        kpis = requests.get(f"{base_url}/kpi/{testid}").json()['payload']
        row = {
            "Energy cost": kpis.get("cost_tot"),
            "Thermal discomfort": kpis.get("tdis_tot"),
        }
        data.append(row)

    end_time = time.time()
    time_cost = end_time - start_time

    # Save the last record
    last_row = data[-1]
    result = {
        "Time cost": time_cost,
        "Energy cost": last_row["Energy cost"],
        "Thermal discomfort": last_row["Thermal discomfort"]
    }

    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=4)

    # 6. Stop test_id and release resources
    resp = requests.put(f"{base_url}/stop/{testid}")
    print("stop:", resp.status_code, resp.text)
    print(f"KPI saved to {output_json}")
    return result

if __name__ == "__main__":
    weights = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9, 1.0,
               1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
               2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    weights = [w / 10.0 for w in weights]
    days=14

    # --------- peak_heat_day---------
    test_period="peak_heat_day"
    all_results = []

    for weight in weights:
        result  = run_boptest_RC_MPC(test_period=test_period, weight=weight,days =days)
        result_record = {
            "weight": weight,
            "Energy cost": result['Energy cost'],
            "Thermal discomfort": result['Thermal discomfort'],
            "Time cost": result['Time cost']
        }
        all_results.append(result_record)

        print(f"weight: {weight:.2f}, Energy cost: {result['Energy cost']:.4f}, "
              f"Thermal discomfort: {result['Thermal discomfort']:.4f}, "
              f"runtime: {result['Time cost']:.2f} seconds")

        with open("RC/KPI/KPI_peak.json", "w") as f:
            json.dump(all_results, f, indent=4)

    # --------- typical_heat_day---------
    test_period = "typical_heat_day"
    all_results = []


    for weight in weights:
        result = run_boptest_RC_MPC(test_period=test_period, weight=weight,days =days)
        result_record = {
            "weight": weight,
            "Energy cost": result['Energy cost'],
            "Thermal discomfort": result['Thermal discomfort'],
            "Time cost": result['Time cost']
        }
        all_results.append(result_record)

        print(f"weight: {weight:.2f}, Energy cost: {result['Energy cost']:.4f}, "
              f"Thermal discomfort: {result['Thermal discomfort']:.4f}, "
              f"runtime: {result['Time cost']:.2f} seconds")

        with open("RC/KPI/KPI_typical.json", "w") as f:
            json.dump(all_results, f, indent=4)


