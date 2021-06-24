import os
import pandas as pd
import json
import matplotlib as mpl
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import sqlite3
import math
from ..utils.configs import DpPolicyType

# sns.set_palette("viridis")
viridis = sns.color_palette("viridis")


# sns.palplot(custom_palette)


def parse_json(d):
    assert d['sim.exception'] is None
    res = {}
    c = d['config']
    res['blocks_mice_fraction'] = c["task.demand.num_blocks.mice_percentage"]
    res['epsilon_mice_fraction'] = c["task.demand.epsilon.mice_percentage"]
    res['arrival_interval'] = c["task.arrival_interval"]
    res['policy'] = c["resource_master.dp_policy"]
    res['lifetime'] = c["resource_master.block.lifetime"]
    res['sim_index'] = c["meta.sim.index"]

    res['task_amount_N'] = c["resource_master.dp_policy.denominator"]
    res['task_timeout'] = c['task.timeout.interval']
    res['is_rdp'] = c['resource_master.dp_policy.is_rdp']

    res['granted_tasks_total'] = d["succeed_tasks_total"]

    res['num_granted_tasks_l_dp_l_blk'] = d["succeed_tasks_l_dp_l_blk"]  # large dp large block #
    res['num_granted_tasks_l_dp_s_blk'] = d["succeed_tasks_l_dp_s_blk"]  # large dp small block #
    res['num_granted_tasks_s_dp_l_blk'] = d["succeed_tasks_s_dp_l_blk"]
    res['num_granted_tasks_s_dp_s_blk'] = d["succeed_tasks_s_dp_s_blk"]

    res['granted_tasks_per_10sec'] = d["succeed_tasks_total"] * 10 / d["sim.time"]
    res['sim_duration'] = d["sim.time"]
    res['grant_delay_P50'] = d.get("dp_allocation_duration_Median")
    res['grant_delay_P99'] = d.get("dp_allocation_duration_P99")  # may be None
    res['grant_delay_avg'] = d["dp_allocation_duration_avg"]
    res['grant_delay_max'] = d["dp_allocation_duration_max"]
    res['grant_delay_min'] = d["dp_allocation_duration_min"]
    # res['dp_allocation_duration_list'] = d['dp_allocation_duration']

    return res


def workspace2dataframe(workspace_dir):
    data_files = list(os.walk(workspace_dir))
    # raise(Exception(d))
    table_data = []
    # count = 0
    json_file = None
    sqlite_file = None
    # for file in data_files:
    #     if file == 'err.yaml':
    #         raise Exception(file)
    #     elif file.endswith('json'):
    #         json_file = file
    #     elif file.endswith('sqlite'):
    #         sqlite_file = file
    json_file = "result.json"
    sqlite_file = "sim.sqlite"
    for d in data_files:
        if not d[1]:  # subdirectory
            result_json = None
            for file in d[2]:
                if file.endswith('json'):
                    result_json = file
                elif file == 'err.yaml':
                    raise (Exception(d))
            # if result_json is None:
            #     # not found json
            #     print(d)
            #     continue
            # raise (Exception(d))

            with open(os.path.join(d[0], json_file)) as f:
                data = json.load(f)
                try:
                    parsed_d = parse_json(data)
                    with sqlite3.connect(os.path.join(d[0], sqlite_file)) as conn:
                        alloc_dur = conn.execute(
                            # "select (abs(dp_commit_timestamp) - start_timestamp) AS dp_allocation_duration  from tasks").fetchall()
                            "select start_timestamp,dp_commit_timestamp from tasks"
                        ).fetchall()
                        parsed_d['dp_allocation_duration_list'] = alloc_dur
                        err_alloc_dur = conn.execute(
                            "select start_timestamp,dp_commit_timestamp from tasks where abs(dp_commit_timestamp) < start_timestamp"
                        ).fetchall()
                        if not len(err_alloc_dur) == 0:
                            raise Exception(err_alloc_dur)

                except Exception as e:
                    print(e)
                    print(data['config']['resource_master.dp_policy'])
                    print(data['config']['meta.sim.index'])
                    print(data['config']["resource_master.block.lifetime"])
                    #                 pprint(data)
                    print('\n\n\n')

                    raise (e)
                #             if parsed_d['task_amount'] != 8192:
                table_data.append(parsed_d)
    #                 count += 1
    # print(count)
    table = pd.DataFrame(table_data)
    # table = table[table.epsilon_mice_fraction == 100]
    # table.replace({'policy': {'DPF_N_234': 'DPF-N', 'fcfs2342': 'FCFS', 'DPF_T_234': 'DPF-T', 'RR_NN': 'RR-N','RR_T':'RR-T'}},
    #               inplace=True)
    if len(table.columns) == 0:
        raise Exception("no data found")
    table['N_or_T_based'] = None
    is_n_based = lambda x: x in (
        DpPolicyType.DP_POLICY_DPF_N.value, DpPolicyType.DP_POLICY_RR_N.value, DpPolicyType.DP_POLICY_RR_N2.value)
    is_t_based = lambda x: x in (
        DpPolicyType.DP_POLICY_DPF_T.value, DpPolicyType.DP_POLICY_RR_T.value, DpPolicyType.DP_POLICY_DPF_NA.value)
    table.loc[table['policy'].apply(is_n_based), "N_or_T_based"] = 'N'
    table.loc[table['policy'].apply(is_t_based), "N_or_T_based"] = 'T'
    # table.loc[table['policy']=='FCFS',"N_or_T_based"] = 'T'
    #
    # tt = table[table['policy'] == 'FCFS'].copy()
    # tt['N_or_T_based'] = 'N'
    # table = table.append(tt, ignore_index=True)
    table["N_"] = table.apply(get_n, axis=1)
    table["N_or_T_"] = table["N_"]
    table["N_or_T_"].loc[table['N_or_T_based'] == 'T'] = table["lifetime"]

    return table


def get_n(row):
    if row['N_or_T_based'] == 'N':
        return row['task_amount_N']
    elif row['N_or_T_based'] == 'T':
        return round(row['lifetime'] / row['arrival_interval'])
    else:
        return -1  # for fcfs


def load_filter_by_dimension(df, dimension):
    load_dim = ['blocks_mice_fraction', 'epsilon_mice_fraction', 'arrival_interval']
    assert dimension in load_dim
    load_dim.remove(dimension)
    for dim in load_dim:
        df = df[df[dim] == max(df[dim])]
    return df


['blocks_mice_fraction', 'epsilon_mice_fraction', 'arrival_interval']
['rate123', 'DPF_N_234', 'DPF_T_234', 'fcfs2342']


# block mice : 1 block
# block elephant : 100 block

# e  mice : 1e-3 epsilon
# e  mice : 1e-1 epsilon

def plot_by_contention_dim(table, dimension, delay_metrics='grant_delay_avg'):
    single_load_factor_table = load_filter_by_dimension(table, dimension)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    markers = ['^', '*', 'o', ]
    for g in table.groupby('policy'):
        policy, samples = g
        ax.scatter(samples[delay_metrics], samples[dimension], samples['granted_tasks_total'], label=policy, alpha=0.5,
                   marker='o')  # markers.pop(0))
        for g2 in samples.groupby(dimension):
            dim, g2_sample = g2
            ax.plot(g2_sample[delay_metrics], g2_sample[dimension], g2_sample['granted_tasks_total'],
                    alpha=0.1)  # markers.pop(0))

    ax.set_xlabel('avg grant delay')
    #     ax.set_xscale('log')
    ax.set_ylabel(dimension)
    ax.set_zlabel('granted_tasks_total')
    #     ax.set_zscale('log')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    a = workspace2dataframe("../results/workspace_06-12-17H-53-31")
    print('hello world')
