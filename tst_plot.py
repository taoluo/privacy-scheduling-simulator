import pandas as pd

from analysis.data_collection import workspace2dataframe
from analysis.plot import plot_granted_tasks, plot_delay_cdf
from utils.configs import DpPolicyType

if __name__ == '__main__':

    # workspace_file = "workspace_06-13-11H-25-40"  # remote single
    workspace_file = "workspace_06-14-04H-16-34" # remote multi
    # workspace_file =  "workspace_05-21-10H-04-42" # exp_result/old multi
    # workspace_dir="./results/%s" % workspace_file
    workspace_dir = "./remote_data/%s" % workspace_file
    # workspace_dir = "./exp_result/%s" % workspace_file
    # fixme policy name is different than config
    # workspace_dir = f"../desmod/docs/examples/data_snapshot/exp_result/single_block/{workspace_file}"
    table = workspace2dataframe(workspace_dir)
    # table = table.loc[(table['blocks_mice_fraction'] == 75) & (table['epsilon_mice_fraction'] == 75)]
    title = "fig1: multi block, %s completed as a function of N/T"
    table1 = table.loc[(table['epsilon_mice_fraction'] == 75)]
    df_padding_list = []
    fcfs_padding = table1[table1.policy == DpPolicyType.DP_POLICY_FCFS.value]
    for N_or_T_based in ('N', 'T'):
        pad_nt = fcfs_padding.copy()
        pad_nt.N_or_T_ = -1
        pad_nt.N_or_T_based = N_or_T_based

        max_N_or_T = max(table1[table1.N_or_T_based == N_or_T_based].N_or_T_)
        right = fcfs_padding.copy()
        right.N_or_T_ = max_N_or_T
        right.N_or_T_based = N_or_T_based

        df_padding_list.append(pad_nt)
        df_padding_list.append(right)
    table1_1 = pd.concat([table1] + df_padding_list, axis=0)

    plot_granted_tasks(save_file_name="figure1.pdf", table=table1_1, title='titles', xaxis_col='N_or_T_',
                       yaxis_col='granted_tasks_total')
    title = 'fig2: Single block, multiple CDFs of delay from issue time with different values of N: FCFS, Sage, DPF-N'
    should_modify_alloc_duration = True  # immediate rejection's delay is treated as timeout
    should_plot_granted = False
    should_exclude_late_task = False  # exclude tasks arrived late
    delay_lst_column = 'dp_allocation_duration_list'
    plot_delay_cdf(table1, delay_lst_column, 'N_or_T_', 'figure2.pdf', should_exclude_late_task,
                   should_modify_timeout_duration=True, task_timeout=300, plot_title=title)

    title = \
        'fig3: Single block, number of tasks completed as function of Mice percentage: FCFS, Sage, DPF-N'
    # for single block
    fixed_N_or_T_dp = 125
    fixed_N_or_T_rdp = 25399
    is_fixed_n = lambda x: x in (-1, fixed_N_or_T_dp, fixed_N_or_T_rdp)
    table2 = table.loc[table.N_.apply(is_fixed_n)]

    df_padding_list = []
    fcfs_padding = table2[table2.policy == DpPolicyType.DP_POLICY_FCFS.value]
    # left = fcfs_padding.copy()
    # left.N_or_T_based = 'N'
    for N_or_T_based in ('N', 'T'):
        pad_nt = fcfs_padding.copy()
        pad_nt.N_or_T_based = N_or_T_based
        df_padding_list.append(pad_nt)

    table2_1 = pd.concat([table2] + df_padding_list, axis=0)

    plot_granted_tasks(save_file_name="figure3.pdf", table=table2_1, title='titles', xaxis_col='epsilon_mice_fraction',
                       yaxis_col='granted_tasks_total')
    plot_delay_cdf(table2, delay_lst_column, 'epsilon_mice_fraction', 'figure4.pdf', should_exclude_late_task,
                   should_modify_timeout_duration=True, task_timeout=300, plot_title=title)
