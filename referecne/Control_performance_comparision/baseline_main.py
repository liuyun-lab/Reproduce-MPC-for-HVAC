import time
import numpy as np
import json
import requests
import argparse
import os

if __name__=="__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('-port', '--port', type=int, default=5000, help='Port to listen')
    parser.add_argument('-p','--time_period',type=str,default='peak',help='Select test time period (peak_heat_day and typical_heat_day)')
    parser.add_argument('-ch','--control_horizon',type=int,default=12,help='Control horizon of mpc')
    parser.add_argument('-ci','--control_interval',type=int,default=3600,help='Control interval of mpc')
    parser.add_argument('-w', '--weight', type=float, default=2, help='Weight in the cost funtion')
    parser.add_argument("--plot",action='store_true',default=False,help="Plot the results")

    parser.add_argument('-dir', '--log_dir', type=str, required=False, help='Path to the log directory.')


    args=parser.parse_args()

    PORT=args.port
    TIME_PERIOD=args.time_period
    CONTROL_HORIZON=args.control_horizon
    CONTROL_INTERVAL=args.control_interval
    WEIGHT=args.weight
    PLOT=args.plot

    log_dir=args.log_dir

    url = f'http://127.0.0.1:{PORT}'

    y = requests.put('{0}/scenario'.format(url),
                         json={'time_period':f'{TIME_PERIOD}_heat_day',
                               'electricity_price':'highly_dynamic'}).json()['payload']

    requests.put('{0}/step'.format(url), json={'step':CONTROL_INTERVAL})
    current_temp=y['time_period']['reaTZon_y'] - 273.15

    cost_tot=[]
    tdist_tot=[]

    current_Qh=y['time_period']['reaPHeaPum_y']
    # print('Simulating...')
    start_time = time.time()  
    for i in range(int(24*14*(3600/CONTROL_INTERVAL))):

        y = requests.post('{0}/advance'.format(url), json={ 'oveHeaPumY_activate': 0}).json()['payload']

        kpis = requests.get('{0}/kpi'.format(url)).json()['payload']
        cost_tot.append(kpis['cost_tot'])
        tdist_tot.append(kpis['tdis_tot'])

    end_time = time.time()
    duration = end_time - start_time
    kpis = requests.get('{0}/kpi'.format(url)).json()['payload']


    formatted_cost = "{:.4f}".format(kpis.get('cost_tot', 0))
    formatted_tdis = "{:.4f}".format(kpis.get('tdis_tot', 0))

    # print(f"w={WEIGHT}, cost_tot: {formatted_cost}, tdis_tot: {formatted_tdis}")
    print(f"h={CONTROL_HORIZON}, cost_tot: {formatted_cost}, tdis_tot: {formatted_tdis}, runtime: {duration:.2f} seconds")

    data = {
        "h": CONTROL_HORIZON,
        "cost_tot": formatted_cost,
        "tdis_tot": formatted_tdis,
        "runtime": f"{duration:.2f}"
    }
    if TIME_PERIOD == 'peak':
        with open(os.path.join(log_dir, f'performance_16.json'), 'w') as f:
            json.dump(data, f, indent=4)
    else:

        with open(os.path.join(log_dir, f'performance_108.json'), 'w') as f:
            json.dump(data, f, indent=4)
