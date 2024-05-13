import numpy as np
import requests
import argparse
import time
from controller.ann_mpc import MPC
import utils
import os
import json
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-port', '--port', type=int, default=5000, help='Port to listen')
    parser.add_argument('-p', '--time_period', type=str, default='peak',
                        help='Select test time period (peak_heat_day and typical_heat_day)')
    parser.add_argument('-ch', '--control_horizon', type=int, default=12, help='Control horizon of mpc')
    parser.add_argument('-ci', '--control_interval', type=int, default=3600, help='Control interval of mpc')
    parser.add_argument('-w', '--weight', type=float, default=5, help='Weight in the cost funtion')
    parser.add_argument("--plot", action='store_true', default=True, help="Plot the results")

    parser.add_argument('-dir', '--log_dir', type=str, required=False, help='Path to the log directory.')

    args=parser.parse_args()

    PORT = args.port
    TIME_PERIOD = args.time_period
    CONTROL_HORIZON = args.control_horizon
    CONTROL_INTERVAL = args.control_interval
    WEIGHT = args.weight
    PLOT = args.plot
    log_dir = args.log_dir



    url = f'http://127.0.0.1:{PORT}'

    y = requests.put('{0}/scenario'.format(url),
                         json={'time_period':f'{TIME_PERIOD}_heat_day',
                               'electricity_price':'highly_dynamic'}).json()['payload']

    requests.put('{0}/step'.format(url), json={'step':CONTROL_INTERVAL})
    current_temp=y['time_period']['reaTZon_y'] - 273.15
    controller = MPC(control_horizon=CONTROL_HORIZON,weight=WEIGHT)
    cost_tot = []
    tdist_tot = []
    start_time = time.time() 
    for i in range(int(24*14*(3600/CONTROL_INTERVAL))):

        points=['TDryBul','HDirNor','InternalGainsRad[1]','PriceElectricPowerHighlyDynamic','UpperSetp[1]','LowerSetp[1]']
        w = requests.put('{0}/forecast'.format(url),
                         json={'point_names': points, 'horizon': CONTROL_HORIZON*3600, 'interval': CONTROL_INTERVAL}).json()['payload']
        Ta = np.array(w['TDryBul'][1:])-273.15
        solar_radiation = w['HDirNor'][1:]
        internal_gains = w['InternalGainsRad[1]'][1:]
        price = w['PriceElectricPowerHighlyDynamic'][1:]
        UpperSetp = np.array(w['UpperSetp[1]'][1:])-273.15-0.5
        LowerSetp = np.array(w['LowerSetp[1]'][1:])-273.15+0.5

        # temperature_setpoints=np.array(LowerSetp)-273.15+1
        temperature_setpoints= np.full(CONTROL_HORIZON, 22.5)

        disturbance=np.array([Ta,
                              internal_gains,
                              solar_radiation
                              ]).T

        u=controller.forecast(current_temp,disturbance,price,temperature_setpoints,UpperSetp,LowerSetp)

        # if i % 24 == 0:
        #     print(f'day:{i // 24 + 1}')


        y = requests.post('{0}/advance'.format(url), json={'oveHeaPumY_u': u, 'oveHeaPumY_activate': 1}).json()['payload']
        current_temp=y['reaTZon_y'] - 273.15

        kpis = requests.get('{0}/kpi'.format(url)).json()['payload']
        cost_tot.append(kpis['cost_tot'])
        tdist_tot.append(kpis['tdis_tot'])

    end_time = time.time()
    duration = end_time - start_time
    kpis = requests.get('{0}/kpi'.format(url)).json()['payload']

    formatted_cost = "{:.4f}".format(kpis.get('cost_tot', 0))
    formatted_tdis = "{:.4f}".format(kpis.get('tdis_tot', 0))

    
    print(f"h={CONTROL_HORIZON}, cost_tot: {formatted_cost}, tdis_tot: {formatted_tdis}, runtime: {duration:.2f} seconds")
    if PLOT:
        if TIME_PERIOD == 'peak':
            with open(os.path.join(log_dir, f'kpis_16_w={WEIGHT}.json'), 'w') as f:
                json.dump(kpis, f)
            utils.get_and_plot_results(url=url,log_dir=log_dir, cost_tot=cost_tot,tdist_tot=tdist_tot,start_time=16 * 24 * 3600,
                                       final_time=(16+14) * 24 * 3600,w=WEIGHT,p=TIME_PERIOD,horizon=CONTROL_HORIZON,interval=CONTROL_INTERVAL,controller='ann_mpc')
        else:
            with open(os.path.join(log_dir, f'kpis_108_w={WEIGHT}.json'), 'w') as f:
                json.dump(kpis, f)
            utils.get_and_plot_results(url=url, log_dir=log_dir, cost_tot=cost_tot, tdist_tot=tdist_tot,
                                       start_time=108 * 24 * 3600, final_time=(108 + 14) * 24 * 3600, w=WEIGHT,
                                       p=TIME_PERIOD, horizon=CONTROL_HORIZON, interval=CONTROL_INTERVAL,
                                       controller='ann_mpc')


