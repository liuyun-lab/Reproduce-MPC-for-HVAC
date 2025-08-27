import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
import requests
import matplotlib.dates as mdates
def get_and_plot_results(url, log_dir, cost_tot,tdist_tot,start_time, final_time,w,p,controller='MPC',horizon=12,interval=3600,shrinkage=0.5):
    '''
    Get and plot result key trajectories from the test case.
    The key trajectories are zone operative temperature, zone heating and
    cooling set points, heat pump modulation signal, outside air dry bulb
    temperature, and outside direct normal solar irradiation.

    Parameters
    ----------
    url : str
    The url for the BOPTEST service.
    start_time : float or int
    The start time in seconds from the beginning of the year for data.
    final_time : float or int
    The final time in seconds from the beginning of the year for data.

    Returns
    -------
    None
    '''

    df_res = pd.DataFrame()
    for point in ['reaTZon_y', 'reaTSetHea_y', 'reaTSetCoo_y', 'oveHeaPumY_u',
                  'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y']:
        args = {'point_names': [point],
                'start_time': start_time,
                'final_time': final_time}

        res = requests.put('{0}/results'.format(url), json=args).json()['payload']
        df_res = pd.concat((df_res, pd.DataFrame(data=res[point], index=res['time'], columns=[point])), axis=1)

    df_res.index.name = 'time'
    df_res.reset_index(inplace=True)
    df_res = reindex(df_res)

    test_episode_length=14 * 24 * 3600
    control_interval=3600

    requests.put('{0}/initialize'.format(url),
                 json={'start_time': df_res['time'].iloc[0],
                       'warmup_period': 0}).json()


    forecast = requests.put('{0}/forecast'.format(url),
                     json={'point_names': ['PriceElectricPowerHighlyDynamic'], 'horizon': test_episode_length ,
                           'interval': control_interval/ 10}).json()['payload']# Take 10 points per step.
    df_for = pd.DataFrame(forecast)
    df_for = reindex(df_for)
    df_for.drop('time', axis=1, inplace=True)

    df = pd.concat((df_res, df_for), axis=1)
    df = create_datetime(df)

    df.dropna(axis=0, inplace=True)



    plt.close()

    x_time = df_res.index / 3600. / 24.

    rewards_time_days = np.arange(df_res.index[0], start_time+test_episode_length, interval) / 3600. / 24.


    # 创建插值函数
    f2 = interpolate.interp1d(rewards_time_days, cost_tot, kind='zero', fill_value='extrapolate')
    f3 = interpolate.interp1d(rewards_time_days, tdist_tot, kind='zero', fill_value='extrapolate')

    res_time_days = np.array(df_res.index) / 3600. / 24.
    cost_tot_reindexed = f2(res_time_days)
    tdist_tot_reindexed = f3(res_time_days)

    columns = ["reaTZon_y", "reaTSetHea_y", "reaTSetCoo_y", 'oveHeaPumY_u', 'PriceElectricPowerHighlyDynamic',
               'weaSta_reaWeaHDirNor_y', 'weaSta_reaWeaTDryBul_y']
    newdf = pd.DataFrame(df, columns=columns)
    newdf['cost_tot'] = cost_tot_reindexed
    newdf['tdist_tot'] = tdist_tot_reindexed

    newdf.to_csv(os.path.join(log_dir, "result_figure_data_{}_sh={}.csv".format(str(int(res['time'][0] / 3600 / 24)),shrinkage)))



    # _, axs = plt.subplots(3, sharex=True, figsize=(10, 8))
    # # Plot operative temperature
    # axs[0].plot(x_time, df_res['reaTZon_y'] - 273.15, 'darkorange', linestyle='-', linewidth=0.8, label='$T_z$')
    # axs[0].plot(x_time, df_res['reaTSetHea_y'] - 273.15, 'gray', linewidth=0.8, label='Comfort setp.')
    # axs[0].plot(x_time, df_res['reaTSetCoo_y'] - 273.15, 'gray', linewidth=0.8, label='_nolegend_')
    # axs[0].set_yticks(np.arange(15, 31, 5))
    # axs[0].set_ylabel('Operative\ntemperature\n($^\circ$C)')
    #
    # # axt = axs[0].twinx()
    # # axt.plot(x_time, df_res['PriceElectricPowerHighlyDynamic'], color='dimgray', linestyle='dotted', linewidth=1,
    # #          label='Price')
    # # axs[0].plot([], [], color='dimgray', linestyle='-', linewidth=1, label='Price')
    # #
    # # axt.set_ylim(0, 0.3)
    # # axt.set_yticks(np.arange(0, 0.31, 0.1))
    # # axt.set_ylabel('(EUR/kWh)')
    # # axt.set_ylabel('Price\n(EUR/kWh)')
    #
    #
    #
    # axs[0].legend()
    #
    # # Plot heat pump modulation signal
    # axs[1].plot(x_time, df_res['oveHeaPumY_u'], 'darkorange',
    #             linestyle='-', linewidth=0.8, label='$u_{hp}$')
    # axs[1].set_ylabel('Heat pump\nmodulation\nsignal\n( - )')
    # axs[1].legend()
    # # axs[1].set_xlabel('Day of the year')
    #
    # #Plot disturbances
    # axs[2].plot(x_time, df_res['weaSta_reaWeaTDryBul_y'] - 273.15, 'royalblue', linestyle='-', linewidth=0.8,
    #             label='$T_a$')
    # axs[2].set_ylabel('Ambient\ntemperature\n($^\circ$C)')
    # axs[2].set_yticks(np.arange(-5, 16, 5))
    # axs[2].legend(loc='upper left')
    # axs[2].set_xlabel('Day of the year')
    # axt = axs[2].twinx()
    # axt.plot(x_time, df_res['weaSta_reaWeaHDirNor_y'], 'gold', linestyle='-', linewidth=0.8, label='$\dot{Q}_{rad}$')
    # axt.set_ylabel('Solar\nirradiation\n($W$)')
    # axt.legend(loc='upper right')
    #
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # figures_dir = os.path.join(base_dir, 'shell_scripts', 'results', f'{controller}')
    # os.makedirs(figures_dir, exist_ok=True)
    # plt.savefig(f'{figures_dir}/{p}_step={interval}_w={int(w)}_h={int(horizon)}_{controller}2.png')
    # plt.show()


    return None

def reindex(df, interval=60, start=None, stop=None):
    '''
    Define the index. Make sure last point is included if
    possible. If interval is not an exact divisor of stop,
    the closest possible point under stop will be the end
    point in order to keep interval unchanged among index.

    '''

    if start is None:
        start = df['time'][df.index[0]]
    if stop is None:
        stop = df['time'][df.index[-1]]
    index = np.arange(start, stop + 0.1, interval).astype(int)
    df_reindexed = df.reindex(index)

    # Avoid duplicates from FMU simulation. Duplicates lead to
    # extrapolation errors
    df.drop_duplicates('time', inplace=True)

    for key in df_reindexed.keys():
        # Use linear interpolation
        f = interpolate.interp1d(df['time'], df[key], kind='linear',
                                 fill_value='extrapolate')
        df_reindexed.loc[:, key] = f(index)

    return df_reindexed


def create_datetime(df):
    '''
    Create a datetime index for the data

    '''

    datetime = []
    for t in df['time']:
        datetime.append(pd.Timestamp('2020/1/1') + pd.Timedelta(t, 's'))
    df['datetime'] = datetime
    df.set_index('datetime', inplace=True)

    return df

def plot_results_heating(mode, x_time, df_baseline, df_rc_mpc, df_ann_mpc=None):
    # 关闭当前的图像，防止重叠绘图
    plt.close()
    fig = plt.figure(figsize=(14, 12))
    axs = fig.subplots(3)

    # 选择模式对应的数据集
    index = 0 if mode == "peak" else 1

    # 提取数据
    df_res = df_rc_mpc[index]

    # 绘图设置
    colors = ['darkorange', 'bisque', 'tan', 'darkgoldenrod', 'blue']

    label_size=12
    lw=1

    grid_color = '#cccccc'  # 网格颜色
    grid_style = '--'       # 网格线型
    grid_width = 0.8        # 网格线宽

    # 绘制基线和MPC控制结果
    axs[0].plot(x_time, df_res['reaTZon_y'] - 273.15, 'dodgerblue', linestyle='-', linewidth=lw, label='RC-MPC')
    axs[1].plot(x_time, df_res['tdist_tot'], color='dodgerblue', linestyle='-', linewidth=lw, label='_nolegend_')
    axs[2].plot(x_time, df_res['cost_tot'], color='dodgerblue', linestyle='-', linewidth=lw, label='_nolegend_')

    if df_ann_mpc is not None:
        axs[0].plot(x_time, df_ann_mpc[index]['reaTZon_y'] - 273.15, 'darkorange', linestyle='-', linewidth=lw, label='ANN-MPC')
        axs[1].plot(x_time, df_ann_mpc[index]['tdist_tot'], color='darkorange', linestyle='-', linewidth=lw, label='_nolegend_')
        axs[2].plot(x_time, df_ann_mpc[index]['cost_tot'], color='darkorange', linestyle='-', linewidth=lw, label='_nolegend_')

    # axs[0].plot(x_time, df_baseline[index]['reaTZon_y'] - 273.15, 'k', linestyle='-', linewidth=lw, label='Baseline')
    axs[0].plot(x_time, df_res['reaTSetCoo_y'] - 273.15, 'gray', linewidth=0.8, label='Comfort setpoint')
    axs[0].plot(x_time, df_res['reaTSetHea_y'] - 273.15, 'gray', linewidth=0.8, label='_nolegend_')

    # 格式化和标签
    axs[0].set_yticks(np.arange(15, 31, 5))
    axs[0].set_ylabel('Operative\ntemperature\n($^\circ$C)', size=label_size)
    axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)  # 只显示刻度，不显示X轴标签

    # axt = axs[0].twinx()
    # axt.set_ylim(0, 0.3)
    # axt.set_yticks(np.arange(0, 0.31, 0.1))
    # axt.set_ylabel('Price\n(EUR/kWh)', size=label_size)
    # axt.plot(x_time, df_res['PriceElectricPowerHighlyDynamic'], color='dimgray', linestyle='dotted', linewidth=1, label='Price')

    axs[1].set_ylabel('Thermal\ndiscomfort\n(Kh/zone)', size=label_size)
    axs[1].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)  # 只显示刻度，不显示X轴标签
    axs[1].plot(x_time, df_baseline[index]['tdist_tot'], 'k', linestyle='-', linewidth=lw, label='_nolegend_')

    axs[2].set_ylabel('Operational\ncost\n(EUR/$m^2$)', size=label_size)
    axs[2].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)  # 只显示刻度，不显示X轴标签
    axs[2].plot(x_time, df_baseline[index]['cost_tot'], 'k', linestyle='-', linewidth=lw, label='_nolegend_')

    # # 绘制环境扰动
    # axs[3].plot(x_time, df_res['weaSta_reaWeaTDryBul_y'] - 273.15, 'gold', linestyle='-', linewidth=lw, label='$T_a$')
    # axs[3].set_ylabel('Ambient\ntemperature\n($^\circ$C)', size=label_size)
    # axs[3].set_yticks(np.arange(-5, 16, 5))
    # axs[3].set_xlabel('Day of the year (2020)', size=label_size)

    # axt = axs[3].twinx()
    # axt.plot(x_time, df_res['weaSta_reaWeaHDirNor_y'], 'tomato', linestyle='-', linewidth=lw, label='$\dot{Q}_{rad}$')
    # axt.set_ylabel('Solar\nirradiation\n($W$)', size=label_size)

    locator = mdates.DayLocator(interval=2)
    formatter = mdates.DateFormatter('%b %d')
    start_date = x_time[0] - pd.Timedelta(days=0.5)
    end_date = x_time[-1] + pd.Timedelta(days=0.5)
    for ax in axs[:3]:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    # 为最后一个子图设置X轴刻度和标签
    # axs[3].xaxis.set_major_locator(locator)
    # axs[3].xaxis.set_major_formatter(formatter)
    # axs[3].tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)

    # 保证所有子图共享X轴
    for ax in axs:
        ax.set_xlim([start_date, end_date])
    # 添加图例
    lines, labels = [], []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.505, 0.01), fancybox=True, shadow=True, ncol=len(lines), prop={'size': 13})
    # for ax in axs[1:]:
    #     ax.grid(True, which='both', linestyle=grid_style, color=grid_color, linewidth=grid_width)
    plt.savefig(f"mpc_{mode}_comparison.png", bbox_inches='tight')
    plt.show()