import pandas as pd
import requests

def Get_2week_data(
    base_url="http://127.0.0.1:80",
    case_name="bestest_hydronic_heat_pump",
    test_period="peak_heat_day",
    ele_price="highly_dynamic",
    step=3600,
    horizon=3600,
    interval=3600,
    days=14,
    output_csv="uPI_peak.csv"
):
    # 1. Select testcase and get test_id
    resp = requests.post(f"{base_url}/testcases/{case_name}/select")
    testid = resp.json()["testid"]

    # 2. Choose scenario
    resp = requests.put(f"{base_url}/scenario/{testid}",
                        json={'time_period': test_period,
                              'electricity_price': ele_price}).json()['payload']

    # 3. Set time step
    requests.put(f"{base_url}/step/{testid}", json={"step": step})

    data = []
    point_names = ["TDryBul", "HGloHor", "InternalGainsRad[1]", "Occupancy[1]"]

    # 4. Collect data for the specified period
    time_period = 24 * days
    for i in range(time_period):
        payload = requests.post(f"{base_url}/advance/{testid}",
                                json={'oveHeaPumY_activate': 0}).json()["payload"]
        kpis = requests.get(f"{base_url}/kpi/{testid}").json()['payload']

        resp_forecast = requests.put(
            f"{base_url}/forecast/{testid}",
            json={
                "point_names": point_names,
                "horizon": horizon,
                "interval": interval
            }
        )
        forecast = resp_forecast.json()["payload"]

        row = {
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
            "Energy cost": kpis.get("cost_tot"),
            "Thermal discomfort": kpis.get("tdis_tot"),
        }
        data.append(row)

    columns = [
        "Time", "Action", "Indoor_Temperature", "Ambient_Temperature", "Solar_Radiation",
        "Set_Heat_Temperature", "Set_Cool_Temperature", "Cop", "Heat_Pump_Power",
        "Emission_circuit_pump_electrical_power", "Electrical_power_heat_pump_evaporator_fan",
        "Occupants", "Occupancy_Gain", "reaQHeaPumEva_y",
        "Q_Heat_Pump", "reaQHeaPumCon_y", "Energy cost", "Thermal discomfort"
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)

    #Stop test_id and release resources
    resp = requests.put(f"{base_url}/stop/{testid}")
    print("stop:", resp.status_code, resp.text)
    return 0


if __name__ == "__main__":
    for period in ["peak_heat_day", "typical_heat_day"]:
        Get_2week_data(test_period=period, output_csv=f"uPI_{period}.csv")