

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plot_rdp = False
plot_dp = True

def prep_cdf_per_fraction2(table_delay, policy, is_rdp, N_or_T, mice_fractions):
    # print(table_delay, policy, is_rdp, N_or_T, mice_fractions)
    if N_or_T is not None:
        delay_by_policy = table_delay.loc[
        (table_delay['policy'] == policy) & (table_delay['N_or_T'] == N_or_T) & (table_delay['is_rdp'] == is_rdp)][['epsilon_mice_fraction', 'dp_allocation_duration']]
    else:
        assert policy == 'FCFS'
        delay_by_policy = table_delay.loc[(table_delay['policy'] == policy)& (table_delay['is_rdp'] == is_rdp)&(table_delay['N_or_T_based'] == 'N')][
            ['epsilon_mice_fraction', 'dp_allocation_duration']]

    sorted_delays = []
    dur_len = None
    total_task_amount = None
    for mf in mice_fractions:
        delay_by_fraction = delay_by_policy.loc[(delay_by_policy['epsilon_mice_fraction'] == mf)][
            'dp_allocation_duration']
        if total_task_amount is not None:
            assert total_task_amount == len(delay_by_fraction)
        else:
            total_task_amount = len(delay_by_fraction)
        # delay_by_fraction = delay_by_fraction.append(pd.Series([timeout] * padding_size), ignore_index=True)
        delay_by_fraction = delay_by_fraction.sort_values(ascending=True).reset_index(drop=True)
        delay_by_fraction.name = mf
        sorted_delays.append(delay_by_fraction)
    assert (total_task_amount is not None) and ( total_task_amount != 0)
    # print("total_task_amount %d" % total_task_amount)
    sorted_delays.append(
        pd.Series([1 / total_task_amount] * total_task_amount, name='prob').cumsum().reset_index(drop=True))
    sorted_delays_cdf = pd.concat(sorted_delays, axis=1)
    return sorted_delays_cdf



def prep_cdf_per_fraction(delay_by_policy,mice_fractions, total_task_amount, timeout):
    sorted_delays = []
    dur_len = None
    for mf in mice_fractions:

        delay_by_fraction = delay_by_policy.loc[(delay_by_policy['epsilon_mice_fraction'] == mf)]['dp_allocation_duration']
        padding_size = total_task_amount - len(delay_by_fraction)
        delay_by_fraction = delay_by_fraction.append(pd.Series([timeout] * padding_size), ignore_index=True)
        delay_by_fraction = delay_by_fraction.sort_values(ascending=True).reset_index(drop=True)
        delay_by_fraction.name = mf
        sorted_delays.append(delay_by_fraction)

    sorted_delays.append(pd.Series([1 /total_task_amount] * total_task_amount, name='prob').cumsum().reset_index(drop=True))
    sorted_delays_cdf = pd.concat(sorted_delays, axis=1)
    return sorted_delays_cdf




def plot_graph4(table,save_file_name,task_timeout):
    title = "fig4 Single block, CDF of delay from issue time as function of Mice percentage: FCFS, Sage, DPF-N"
    table = table.copy()
    table.loc[table['policy']=='FCFS',"N_or_T_based"] = 'T'

    tt = table[table['policy'] == 'FCFS'].copy()
    tt['N_or_T_based'] = 'N'
    table = table.append(tt, ignore_index=True)

    # filter granted task


    # data filter logic for plot
    task_arrival_itvl_light = 0.078125
    task_arrival_itvl_heavy = 0.004264781
    # aligned dp rdp
    fixed_N_or_T_dp = 297
    fixed_N_or_T_rdp = 8875  # fixme !!

    # N param on the plateau
    fixed_N_or_T_dp = 175
    fixed_N_or_T_rdp = 14514  # fixme !!


    # fixed_N_or_T_dp_T = 375 * task_arrival_itvl_light
    fixed_N_or_T_dp_T = fixed_N_or_T_dp * task_arrival_itvl_light

    fixed_N_or_T_rdp_T =  fixed_N_or_T_rdp * task_arrival_itvl_light

    should_filter_granted = False
    # table1 = table.loc[(table['blocks_mice_fraction'] == 100) & (table['epsilon_mice_fraction'] == 75) &  (table['policy'] != 'DPF_T_234')]
    # table1 = table.loc[(table['blocks_mice_fraction'] == 100) & ( (table['N_or_T'] == fixed_N_or_T) |(table['policy'] == 'FCFS'  )) &  (table['policy'] != 'DPF-T')]
    table1 = table.loc[(table['blocks_mice_fraction'] == 75) & ( (table['N_or_T'] == fixed_N_or_T_dp) |(table['N_or_T'] == fixed_N_or_T_dp_T) |(table['N_or_T'] == fixed_N_or_T_rdp)|(table['N_or_T'] == fixed_N_or_T_rdp_T)| (table['N_or_T'] == -1  ))]

    lst_col = 'dp_allocation_duration_list'
    table_delay1 = pd.DataFrame({
               col:np.repeat(table1[col].values, table1[lst_col].str.len())
              for col in table1.columns.difference([lst_col])
          })
    table_delay2 = pd.DataFrame(np.concatenate(table1[lst_col].values),columns= ['commit_time','start_time'])
    table_delay = table_delay1.assign(commit_time=table_delay2['commit_time'],start_time=table_delay2['start_time'] )# [table1.columns.tolist()]

    table_delay.loc[table_delay['commit_time'].isna(), 'commit_time'] = np.Inf

    table_delay = table_delay.assign(dp_allocation_duration= lambda x: np.abs(x['commit_time']) -x['start_time'])# [table1.columns.tolist()]
    if isinstance(task_timeout, int):
        table_delay.loc[
            (table_delay['commit_time'] < 0) | table_delay[
                'commit_time'].isna(), 'dp_allocation_duration'] = task_timeout
    else:
        table_delay.loc[ table_delay['commit_time']<0|table_delay['commit_time'].isna(),'dp_allocation_duration'  ] = table_delay.loc[ table_delay['commit_time']<0|table_delay['commit_time'].isna(),'task_timeout'  ]


    g = sns.FacetGrid(data = table_delay, row='is_rdp',col="policy",  hue='epsilon_mice_fraction',sharex=False,palette='viridis_r' )#, palette=viridis )
    g.map(sns.ecdfplot, "dp_allocation_duration" )#, stat="count") # ,palette=viridis  )

    g.add_legend()
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle('CDF of task waittime, fixed_N_or_T == %d or %d//' % (fixed_N_or_T_dp ,fixed_N_or_T_rdp )+ title)
    g.savefig(save_file_name)
    # for ax in g.axes[0, :]:
    #     ax.set_xlim(-5, table_delay[table_delay['is_rdp']==False].iloc[0]['task_timeout'])
    #
    #
    # for ax in g.axes[1, :]:
    #     ax.set_xlim(-5, table_delay[table_delay['is_rdp']==True].iloc[0]['task_timeout'])


    if should_filter_granted:
        table_delay_filter = table_delay.loc[ table_delay['commit_time']>0]
        g = sns.FacetGrid(data = table_delay_filter, col="policy",  hue='epsilon_mice_fraction' )#, palette=viridis )
        g.map(sns.ecdfplot, "dp_allocation_duration" , stat="count")# ,palette=viridis  )
        g.add_legend()
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle('granted tasks count VS task waitime, fixed_N_or_T == %d //' % (fixed_N_or_T_dp ,fixed_N_or_T_rdp ) + title)


    # fcfs_dur = table_delay.loc[(table_delay['policy'] == 'FCFS')][['epsilon_mice_fraction', 'dp_allocation_duration']]
    # mice_fractions = sorted(fcfs_dur['epsilon_mice_fraction'].unique())
    # no_mice_delay = fcfs_dur.loc[table_delay['epsilon_mice_fraction'] == 0][['epsilon_mice_fraction','dp_allocation_duration']]
    # assert no_mice_delay.shape[0] != 0
    # task_timeout = max(no_mice_delay['dp_allocation_duration'])
    # total_tasks = no_mice_delay.shape[0]
    #
    #
    #
    # fcfs_delays_cdf = prep_cdf_per_fraction(delay_by_policy=fcfs_dur, mice_fractions=mice_fractions, total_task_amount=total_tasks, timeout=task_timeout)
    # fcfs_delays_cdf.to_csv('graph4_data/fcfs_dur.csv',index=False,header=True)
    #
    # fixed_N_or_T = 125
    # dpf_dur = table_delay.loc[
    #     (table_delay['policy'] == 'DPF-N') & (table_delay['N_or_T'] == fixed_N_or_T)][['epsilon_mice_fraction', 'dp_allocation_duration']]
    #
    # dpf_delays_cdf = prep_cdf_per_fraction(delay_by_policy=dpf_dur, mice_fractions=mice_fractions,
    #                                         total_task_amount=total_tasks, timeout=task_timeout)
    # dpf_delays_cdf.to_csv('graph4_data/dpf_dur.csv', index=False, header=True)
    #
    #
    # sage_dur = table_delay.loc[
    #     (table_delay['policy'] == 'Sage') & (table_delay['N_or_T'] == fixed_N_or_T)][['epsilon_mice_fraction', 'dp_allocation_duration']]
    #
    # sage_delays_cdf = prep_cdf_per_fraction(delay_by_policy=sage_dur, mice_fractions=mice_fractions,
    #                                         total_task_amount=total_tasks, timeout=task_timeout)
    # sage_delays_cdf.to_csv('graph4_data/sage_dur.csv', index=False, header=True)

    mice_fractions = sorted(table_delay['epsilon_mice_fraction'].unique())

    dpf_delays_cdf = prep_cdf_per_fraction2(table_delay=table_delay, policy='FCFS', is_rdp=False, N_or_T=None, mice_fractions=mice_fractions)
        # delay_by_policy=dpf_dur, mice_fractions=mice_fractions,
        #                                     total_task_amount=total_tasks)
    dpf_delays_cdf.to_csv('graph4_data/dpf_dur_dp.csv', index=False, header=True)
    if plot_rdp:
        dpf_delays_cdf = prep_cdf_per_fraction2(table_delay=table_delay, policy='FCFS', is_rdp=True,
                                                N_or_T=None, mice_fractions=mice_fractions)
        dpf_delays_cdf.to_csv('graph4_data/dpf_dur_rdp.csv', index=False, header=True)
    # fixed_N_or_T = 125
    # dpf_dur = table_delay.loc[
    #     (table_delay['policy'] == 'DPF-N') & (table_delay['N_or_T'] == fixed_N_or_T_dp)][['epsilon_mice_fraction', 'dp_allocation_duration']]

    dpf_delays_cdf = prep_cdf_per_fraction2(table_delay=table_delay, policy='DPF-N', is_rdp=False, N_or_T=fixed_N_or_T_dp, mice_fractions=mice_fractions)
        # delay_by_policy=dpf_dur, mice_fractions=mice_fractions,
        #                                     total_task_amount=total_tasks)
    dpf_delays_cdf.to_csv('graph4_data/dpf_dur_dp.csv', index=False, header=True)
    if plot_rdp:
        dpf_delays_cdf = prep_cdf_per_fraction2(table_delay=table_delay, policy='DPF-N', is_rdp=True,
                                                N_or_T=fixed_N_or_T_rdp, mice_fractions=mice_fractions)
        dpf_delays_cdf.to_csv('graph4_data/dpf_dur_rdp.csv', index=False, header=True)



    dpf_delays_cdf = prep_cdf_per_fraction2(table_delay=table_delay, policy='RR-N', is_rdp=False, N_or_T=fixed_N_or_T_dp, mice_fractions=mice_fractions)
        # delay_by_policy=dpf_dur, mice_fractions=mice_fractions,
        #                                     total_task_amount=total_tasks)
    dpf_delays_cdf.to_csv('graph4_data/rr_dur_dp.csv', index=False, header=True)


