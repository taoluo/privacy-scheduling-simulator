
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def plot_granted_tasks(save_file_name, table, title, column_to_plot):
    facetgrid = sns.relplot(data=table, x='N_or_T_', y=column_to_plot, row='is_rdp', col='N_or_T_based',
                            kind='line', hue='policy', style='policy', facet_kws=dict(sharex=False, sharey=False))
    # facetgrid.axes[0,0].set_xlim(-5, 250)
    # facetgrid.axes[0,1].set_xlim(-5, 250)
    # facetgrid.axes[1,0].set_xlim(-50, 35000)
    # facetgrid.axes[1,1].set_xlim(-50, 35000)
    #
    # facetgrid.axes[0, 0].set_ylim(0, 110)
    # facetgrid.axes[0, 1].set_ylim(0, 110)
    # facetgrid.axes[1, 0].set_ylim(0, 16000)
    # facetgrid.axes[1, 1].set_ylim(0, 16000)
    # fixme fix each element in grid.
    # if plot_dp and plot_rdp:
    if False:
        y2_hline_val = table[((table.policy == 'FCFS') & (table.is_rdp == False))][column_to_plot].iloc[0]
        facetgrid.axes[0][0].axhline(y=y2_hline_val, color='red', linestyle=':')
        facetgrid.axes[0][1].axhline(y=y2_hline_val, color='red', linestyle=':')

        y2_hline_val2 = table[((table.policy == 'FCFS') & (table.is_rdp == True))][column_to_plot].iloc[0]
        facetgrid.axes[1][0].axhline(y=y2_hline_val2, color='red', linestyle=':')
        facetgrid.axes[1][1].axhline(y=y2_hline_val2, color='red', linestyle=':')
    plt.subplots_adjust(top=0.8)
    facetgrid.fig.suptitle(title)
    facetgrid.savefig(save_file_name)


def plot_delay_cdf(table, delay_lst_column, should_exclude_late_task, should_modify_timeout_duration, task_timeout, plot_title):
    table_delay1 = pd.DataFrame({
        col: np.repeat(table[col].values, table[delay_lst_column].str.len())
        for col in table.columns.difference([delay_lst_column])
    })
    table_delay2 = pd.DataFrame(np.concatenate(table[delay_lst_column].values), columns=['commit_time', 'start_time'])
    table_delay = table_delay1.assign(commit_time=table_delay2['commit_time'],
                                      start_time=table_delay2['start_time'])  # [table1.columns.tolist()]
    table_delay.loc[table_delay['commit_time'].isna(), 'commit_time'] = np.Inf
    table_delay = table_delay.assign(
        dp_allocation_duration=lambda x: np.abs(x['commit_time']) - x['start_time'])  # [table1.columns.tolist()]
    if should_modify_timeout_duration == True:
        if isinstance(task_timeout, int):
            table_delay.loc[
                (table_delay['commit_time'] < 0) | table_delay['commit_time'].isin(
                    [np.Inf]), 'dp_allocation_duration'] = task_timeout
        else:
            table_delay.loc[(table_delay['commit_time'] < 0) | table_delay['commit_time'].isin(
                [np.Inf]), 'dp_allocation_duration'] = table_delay.loc[
                (table_delay['commit_time'] < 0) | table_delay['commit_time'].isna(), 'task_timeout']

    if should_exclude_late_task == True:
        table_delay = table_delay.loc[
            ~ ((table_delay['commit_time'] < 0) & (table_delay['dp_allocation_duration'] == 0))]
    #     table_delay.loc[ table_delay['commit_time']<0,'dp_allocation_duration'  ] = 1*table_delay.loc[ table_delay['commit_time']<0,'task_timeout'  ]
    # table_delay = table_delay.loc[  (table_delay['N_or_T'] < 500)]
    table_delay['log10_N_or_T'] = table_delay['N_or_T']
    table_delay.loc[table_delay['N_or_T'] > 0, 'log10_N_or_T'] = table_delay.loc[table_delay['N_or_T'] > 0][
        'log10_N_or_T'].apply(lambda x: math.log10(x))
    g = sns.FacetGrid(data=table_delay, row='is_rdp', col="policy", hue='log10_N_or_T', sharex=False,
                      palette='viridis')  # , palette=viridis )
    g.map(sns.ecdfplot, "dp_allocation_duration")  # ,stat="count") #  )# ,palette=viridis  )
    g.add_legend()
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle('CDF of task waittime// ' + plot_title)
    return g, table_delay

