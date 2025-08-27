import time
import numpy as np
import json
import requests
import argparse
import os
from controller.mpc_ecos import MPC as MPC_Ecos
from controller.mpc_cplex import MPC as MPC_Cplex
from controller.mpc_ipopt import MPC as MPC_Ipopt
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-port', '--port', type=int, default=5500, help='Port to listen')
    parser.add_argument('-p', '--time_period', type=str, default='peak', help='Select test time period (peak_heat_day and typical_heat_day)')
    parser.add_argument('-ch', '--control_horizon', type=int, default=12, help='Control horizon of mpc')
    parser.add_argument('-ci', '--control_interval', type=int, default=3600, help='Control interval of mpc')
    parser.add_argument('-w', '--weight', type=float, default=2, help='Weight in the cost function')
    parser.add_argument("--plot", action='store_true', default=True, help="Plot the results")
    parser.add_argument('-dir', '--log_dir', type=str, required=False, help='Path to the log directory.')
    parser.add_argument('-solver', '--solver', type=str, choices=['ecos', 'cplex', 'ipopt'], default='cplex', help='Select the solver to use')

    args = parser.parse_args()

    PORT = args.port
    TIME_PERIOD = args.time_period
    CONTROL_HORIZON = args.control_horizon
    CONTROL_INTERVAL = args.control_interval
    WEIGHT = args.weight
    PLOT = args.plot
    SOLVER = args.solver
    log_dir = args.log_dir

    # Create log directory if it does not exist
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Initialize the MPC controller based on the selected solver
    if SOLVER == 'ecos':
        controller = MPC_Ecos(control_horizon=CONTROL_HORIZON, weight=WEIGHT)
    elif SOLVER == 'cplex':
        controller = MPC_Cplex(control_horizon=CONTROL_HORIZON, weight=WEIGHT)
    elif SOLVER == 'ipopt':
        controller = MPC_Ipopt(control_horizon=CONTROL_HORIZON, weight=WEIGHT)

    url = f'http://127.0.0.1:{PORT}'

    y = requests.put('{0}/scenario'.format(url),
                     json={'time_period': f'{TIME_PERIOD}_heat_day',
                           'electricity_price': 'highly_dynamic'}).json()['payload']

    requests.put('{0}/step'.format(url), json={'step': CONTROL_INTERVAL})
    current_temp = y['time_period']['reaTZon_y'] - 273.15

    cost_tot=[]
    tdist_tot=[]

    current_Qh=y['time_period']['reaPHeaPum_y']
    # print('Simulating...')
    start_time = time.time()  # 记录循环开始前的时间

    iterations=[]
    for i in range(int(24*14*(3600/CONTROL_INTERVAL))):
    # for i in range(int(24*1*(3600/CONTROL_INTERVAL))):

        points=['TDryBul','HGloHor','InternalGainsRad[1]','PriceElectricPowerHighlyDynamic','UpperSetp[1]','LowerSetp[1]']
        w = requests.put('{0}/forecast'.format(url),
                         json={'point_names': points, 'horizon': CONTROL_HORIZON*CONTROL_INTERVAL, 'interval': CONTROL_INTERVAL}).json()['payload']
        Ta = np.array(w['TDryBul'][1:])-273.15
        solar_radiation = np.array(w['HGloHor'][1:])
        internal_gains = np.array(w['InternalGainsRad[1]'][1:])
        price = w['PriceElectricPowerHighlyDynamic'][1:]
        UpperSetp = np.array(w['UpperSetp[1]'][1:])-273.15-0.5
        LowerSetp = np.array(w['LowerSetp[1]'][1:])-273.15+0.5


        disturbance=[Ta,internal_gains,solar_radiation]

        current_states=[current_temp,w['TDryBul'][0]-273.15,w['HGloHor'][0],w['InternalGainsRad[1]'][0],current_Qh]

        u,iteration=controller.forecast(current_states,disturbance,price,UpperSetp,LowerSetp)
        # if i%24==0:
        #     print(f'day:{i//24+1}')




        y = requests.post('{0}/advance'.format(url), json={'oveHeaPumY_u': u, 'oveHeaPumY_activate': 1}).json()['payload']
        current_temp=y['reaTZon_y'] - 273.15
        current_Qh=u*7221+2200

        kpis = requests.get('{0}/kpi'.format(url)).json()['payload']
        cost_tot.append(kpis['cost_tot'])
        tdist_tot.append(kpis['tdis_tot'])
        iterations.append(iteration)

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
        "runtime": f"{duration:.2f}",
        "iteration":np.mean(iterations)
    }
    if PLOT:
        if TIME_PERIOD=='peak':
            with open(os.path.join(log_dir, f'kpis_16_w={WEIGHT}.json'), 'w') as f:
                json.dump(kpis, f)

            with open(os.path.join(log_dir, f'performance_16_w={WEIGHT}.json'), 'w') as f:
                json.dump(data, f, indent=4)


            utils.get_and_plot_results(url=url, log_dir=log_dir, cost_tot=cost_tot, tdist_tot=tdist_tot, start_time=16 * 24 * 3600, final_time=(16 + 14) * 24 * 3600, w=WEIGHT, p=TIME_PERIOD, horizon=CONTROL_HORIZON, interval=CONTROL_INTERVAL, controller='MPC_Cplex')
        else:
            with open(os.path.join(log_dir, f'kpis_108_w={WEIGHT}'), 'w') as f:
                json.dump(kpis, f)

            with open(os.path.join(log_dir, f'performance_108_w={WEIGHT}.json'), 'w') as f:
                json.dump(data, f, indent=4)
            utils.get_and_plot_results(url=url, log_dir=log_dir, cost_tot=cost_tot, tdist_tot=tdist_tot, start_time=108 * 24 * 3600, final_time=(108 + 14) * 24 * 3600, w=WEIGHT, p=TIME_PERIOD, horizon=CONTROL_HORIZON, interval=CONTROL_INTERVAL, controller='MPC_Cplex')

