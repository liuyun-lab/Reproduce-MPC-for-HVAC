import requests
import json
import time
def run_boptest_PI(
    base_url="http://127.0.0.1:80",
    case_name="bestest_hydronic_heat_pump",
    test_period="peak_heat_day",
    ele_price="highly_dynamic",
    step=3600,
    days=14,
    output_json="Performance_PI_peak.json"
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

    # 4. Collect data for the specified period
    time_period = 24 * days
    start_time = time.time()
    for i in range(time_period):
        payload = requests.post(f"{base_url}/advance/{testid}",
                                json={'oveHeaPumY_activate': 0}).json()["payload"]
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
        "Time cost": round(time_cost, 2),
        "Energy cost": round(last_row["Energy cost"], 2),
        "Thermal discomfort": round(last_row["Thermal discomfort"], 2)
    }

    with open(output_json, "w") as f:
        json.dump(result, f, indent=4)

    # 6. Stop test_id and release resources
    resp = requests.put(f"{base_url}/stop/{testid}")
    print("stop:", resp.status_code, resp.text)
    print(f" KPI saved to {output_json}")
    return result


#run_boptest_PI(test_period="typical_heat_day", days=14, output_json="Performance_PI_typical.json")
run_boptest_PI(test_period="peak_heat_day", days=14, output_json="Performance_PI_peak.json")