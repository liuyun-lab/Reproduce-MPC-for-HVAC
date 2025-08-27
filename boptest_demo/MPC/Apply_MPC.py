import pandas as pd
import requests
import json
import time
import numpy as np
from tqdm import trange
from boptest_demo.MPC.ANN_MPC import ann_mpc
from boptest_demo.MPC.RC_MPC import rc_mpc

def run_MPC(
    base_url="http://127.0.0.1:80",
    case_name="bestest_hydronic_heat_pump",
    controller =None,
    test_period="peak_heat_day",
    ele_price="highly_dynamic",
    step=3600,
    days=14,
    output_json=None,
    output_csv=None,
    horizon=12,
    weight=5.0,
    epsilon=0.5,
):
    # 1. Select testcase and get test_id
    resp = requests.post(f"{base_url}/testcases/{case_name}/select")
    testid = resp.json()["testid"]

    data = []
    data2 = []

    # 2. Choose scenario
    resp = requests.put(f"{base_url}/scenario/{testid}",
                     json={'time_period': test_period,
                           'electricity_price': ele_price}).json()['payload']

    # 3. Set time step
    requests.put(f"{base_url}/step/{testid}", json={"step": step})

    # 4. Set Forecasts data for next horizon steps
    points = ['TDryBul', 'HDirNor', 'InternalGainsRad[1]', 'Occupancy[1]',
              'PriceElectricPowerHighlyDynamic', 'UpperSetp[1]', 'LowerSetp[1]']

    # 5. Get initial temperature
    y = resp
    current_temp = y['time_period']['reaTZon_y'] - 273.15

    time_period = 24 * days
    start_time = time.time()
    for i in trange(time_period, desc="Simulating"):

        forecast = requests.put(f"{base_url}/forecast/{testid}",
                json={'point_names': points, 'horizon': horizon*step, 'interval': step}).json()['payload']
        Ta = np.array(forecast['TDryBul'][1:]) - 273.15
        Isol = np.array(forecast['HDirNor'][1:])
        Qint = np.array(forecast['InternalGainsRad[1]'][1:])
        price = np.array(forecast['PriceElectricPowerHighlyDynamic'][1:])
        UpperSetp = np.array(forecast['UpperSetp[1]'][1:]) - 273.15
        LowerSetp = np.array(forecast['LowerSetp[1]'][1:]) - 273.15

        # Prepare disturbances and setpoints for controller
        disturbances = {'Ta': Ta, 'Isol': Isol, 'Qint': Qint}
        setpoints = {'Lower': LowerSetp, 'Upper': UpperSetp}

        # 5.2 Advance simulation with Controller  Default PI
        if controller is None:
            payload = requests.post(f"{base_url}/advance/{testid}").json()['payload']
        else:
            u = controller(
                current_temp=current_temp,
                disturbances=disturbances,
                setpoints=setpoints,
                price=price,
                weight=weight,
                horizon=horizon,
                epsilon=epsilon,
            )
            payload = requests.post(f"{base_url}/advance/{testid}",
                    json={'oveHeaPumY_u': float(u), 'oveHeaPumY_activate': 1}).json()['payload']

        current_temp = payload['reaTZon_y'] - 273.15

        # 5.3 Collect KPIs
        kpis = requests.get(f"{base_url}/kpi/{testid}").json()['payload']
        row = {
            "Energy cost": kpis.get("cost_tot"),
            "Thermal discomfort": kpis.get("tdis_tot"),
        }
        data.append(row)

        # 5.4 Collect system disturbances
        if output_csv is not None:
            row2 = {
                "Time": payload.get("time") / 3600 / 24,  # day
                "Action": payload.get("oveHeaPumY_u"),
                "Indoor_Temperature": payload.get("reaTZon_y") - 273.15 if payload.get("reaTZon_y") else None,
                "Ambient_Temperature": payload.get("weaSta_reaWeaTDryBul_y") - 273.15 if payload.get("weaSta_reaWeaTDryBul_y") else None,
                "Solar_Radiation": payload.get("weaSta_reaWeaHGloHor_y"),
                "Set_Heat_Temperature": payload.get("reaTSetHea_y") - 273.15 if payload.get("reaTSetHea_y") else None,
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

                "Price": forecast['PriceElectricPowerHighlyDynamic'][0],
                "Energy cost": kpis.get("cost_tot"),
                "Thermal discomfort": kpis.get("tdis_tot"),
            }
            data2.append(row2)

    end_time = time.time()
    time_cost = end_time - start_time

    # Save the last record KPI
    last_row = data[-1]
    result = {
        "Time cost": time_cost,
        "Energy cost": last_row["Energy cost"],
        "Thermal discomfort": last_row["Thermal discomfort"]
    }
    if output_json is not None:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=4)
            print(f"KPI saved to {output_json}")

    # Save the system disturbances
    if output_csv is not None:
        columns = [
            "Time", "Action", "Indoor_Temperature", "Ambient_Temperature", "Solar_Radiation",
            "Set_Heat_Temperature", "Set_Cool_Temperature", "Cop", "Heat_Pump_Power",
            "Emission_circuit_pump_electrical_power", "Electrical_power_heat_pump_evaporator_fan",
            "Occupants", "Occupancy_Gain", "reaQHeaPumEva_y",
            "Q_Heat_Pump", "reaQHeaPumCon_y", "Price","Energy cost", "Thermal discomfort"
        ]
        df = pd.DataFrame(data2, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"system disturbances saved to {output_csv}")

    # 6. Stop test_id and release resources
    resp = requests.put(f"{base_url}/stop/{testid}")
    print("stop:", resp.status_code, resp.text)
    return result


if __name__ == "__main__":

    weight=2.2
    days=14
    #test_period="peak_heat_day"
    #test_period="typical_heat_day"

    #controller =ann_mpc
    #controller = None
    #controller = rc_mpc

    controllers = [ann_mpc, rc_mpc, None]
    test_periods = ["peak_heat_day", "typical_heat_day"]
    for controller in controllers:
        for test_period in test_periods:
            if controller is None:
                controller_name = "Baseline"
            else:
                controller_name = controller.__name__

            output_json = (f"Data/KPI_{controller_name}_{test_period.replace('_heat_day', '')}"
                           f"_w={weight:.1f}.json")
            output_csv = (f"Data/Disturbances_{controller_name}"
                          f"_{test_period.replace('_heat_day', '')}_w={weight:.1f}.csv")

            result = run_MPC(controller = controller,test_period=test_period,weight=weight,
                             output_json=output_json, output_csv=output_csv,days=days)

    # print(f"weight: {weight:.2f}, Energy cost: {result['Energy cost']:.4f}, "
    #       f"Thermal discomfort: {result['Thermal discomfort']:.4f}, "
    #       f"runtime: {result['Time cost']:.2f} seconds")