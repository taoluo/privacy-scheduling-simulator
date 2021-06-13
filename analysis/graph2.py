import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
viridis = sns.color_palette("viridis")

task_arrival_itvl_light = 0.078125
task_arrival_itvl_heavy = 0.004264781
task_arrival_itvl = task_arrival_itvl_heavy
aligned_rdp_dp = False
plot_rdp = True
plot_dp = False


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






def plot_graph2(table,save_file_name, task_timeout):
# figure2 Single block, multiple CDFs of delay from issue time with different values of N: FCFS, Sage, DPF-N

# filter granted task
    table = table.copy()
    title = 'fig2: Single block, multiple CDFs of delay from issue time with different values of N: FCFS, Sage, DPF-N'
    should_modify_alloc_duration = True  # immediate rejection's delay is treated as timeout
    should_plot_granted = False
    should_exclude_late_task = False # exclude tasks late-than-retirement arrived task.
    table1 = table.loc[
        (table['blocks_mice_fraction'] == 75) & (table['epsilon_mice_fraction'] == 75) ]# & (table['policy'] != 'DPF-T')]

    lst_col = 'dp_allocation_duration_list'
    g, table_delay = plot_delay_cdf(lst_col, should_exclude_late_task, should_modify_alloc_duration, table1, task_timeout,
                                title)

# for ax in g.axes[0, :]:
    #     ax.set_xlim(-5, table_delay[table_delay['is_rdp']==False].iloc[0]['task_timeout'])
    #
    #
    # for ax in g.axes[1, :]:
    #     ax.set_xlim(-5, table_delay[table_delay['is_rdp']==True].iloc[0]['task_timeout'])

    g.savefig(save_file_name)

    if should_plot_granted:
        table_delay_filter = table_delay.loc[table_delay['commit_time'] > 0]
        g = sns.FacetGrid(data=table_delay_filter, col="policy", hue='N_or_T')  # , palette=viridis )
        g.map(sns.ecdfplot, "dp_allocation_duration", stat="count")  # ,palette=viridis  )
        g.add_legend()
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle('granted tasks count VS task waittime//' + title)
        g.savefig('2-'+save_file_name)
    if plot_dp:
        fcfs_dur = table_delay.loc[(table_delay['policy'] == 'FCFS')&(table_delay['is_rdp'] == False), 'dp_allocation_duration']
        fcfs_dur.sort_values(ascending=True, inplace=True)
        fcfs_dur = pd.concat([fcfs_dur.reset_index(drop=True), pd.Series([1/len(fcfs_dur)]* len(fcfs_dur)).cumsum().reset_index(drop=True)] , axis=1, ignore_index=True)
        fcfs_dur.to_csv('graph2_data/fcfs_dur_dp.csv',index=False,header=False, sep=',')
        total_task = len(fcfs_dur[0])
        timeout = max(fcfs_dur[0])
        assert timeout > 0

    if plot_rdp:
        fcfs_dur = table_delay.loc[
            (table_delay['policy'] == 'FCFS') & (table_delay['is_rdp'] == True), 'dp_allocation_duration']
        fcfs_dur.sort_values(ascending=True, inplace=True)
        fcfs_dur = pd.concat(
            [fcfs_dur.reset_index(drop=True), pd.Series([1 / len(fcfs_dur)] * len(fcfs_dur)).cumsum().reset_index(drop=True)],
            axis=1, ignore_index=True)
        fcfs_dur.to_csv('graph2_data/fcfs_dur_rdp.csv', index=False, header=False, sep=',')
        total_task_rdp = len(fcfs_dur[0])



    if not aligned_rdp_dp:
        if plot_dp:
            dpf_l_dur = table_delay.loc[ (table_delay['policy'] == 'DPF-N') &(table_delay['is_rdp'] == False)& (table_delay['N_or_T'] == 50) , 'dp_allocation_duration']
            assert total_task == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True), pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)], axis=1, ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpfn_50_dur_dp.csv', index=False, header=False, sep=',')

            dpf_l_dur = table_delay.loc[ (table_delay['policy'] == 'DPF-N') &(table_delay['is_rdp'] == False)& (table_delay['N_or_T'] == 75) , 'dp_allocation_duration']
            assert total_task == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True), pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)], axis=1, ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpfn_75_dur_dp.csv', index=False, header=False, sep=',')


            dpf_h_dur = table_delay.loc[ (table_delay['policy'] == 'DPF-N') &(table_delay['is_rdp'] == False)& (table_delay['N_or_T'] == 125) , 'dp_allocation_duration']
            assert total_task == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True), pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1 , ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpfn_125_dur_dp.csv', index=False, header=False, sep=',')


            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == False) & (
                        table_delay['N_or_T'] == 175), 'dp_allocation_duration']
            assert total_task == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpfn_175_dur_dp.csv', index=False, header=False, sep=',')


            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == False) & (
                    table_delay['N_or_T'] == 375), 'dp_allocation_duration']
            assert total_task == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpfn_375_dur_dp.csv', index=False, header=False, sep=',')

            sage_dur = table_delay.loc[ (table_delay['policy'] == 'RR-N')&(table_delay['is_rdp'] == False) & (table_delay['N_or_T'] == 50) , 'dp_allocation_duration']
            assert total_task == len(sage_dur)
            sage_dur.sort_values(ascending=True, inplace=True)
            sage_dur = pd.concat([sage_dur.reset_index(drop=True), pd.Series([1 / len(sage_dur)] * len(sage_dur)).cumsum().reset_index(drop=True)], axis=1, ignore_index=True)
            sage_dur.to_csv('graph2_data/rrn_dur_dp.csv', index=False, header=False, sep=',')

            dpf_l_dur = table_delay.loc[ (table_delay['policy'] == 'DPF-T') &(table_delay['is_rdp'] == False)& (table_delay['N_or_T'] == 50*task_arrival_itvl) , 'dp_allocation_duration']
            assert total_task == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True), pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)], axis=1, ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpft_50_dur_dp.csv', index=False, header=False, sep=',')

            dpf_h_dur = table_delay.loc[ (table_delay['policy'] == 'DPF-T') &(table_delay['is_rdp'] == False)& (table_delay['N_or_T'] == 125*task_arrival_itvl) , 'dp_allocation_duration']
            assert total_task == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True), pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1 , ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpft_125_dur_dp.csv', index=False, header=False, sep=',')


            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-T') & (table_delay['is_rdp'] == False) & (
                        table_delay['N_or_T'] == 175 * task_arrival_itvl), 'dp_allocation_duration']
            assert total_task == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpft_175_dur_dp.csv', index=False, header=False, sep=',')


            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-T') & (table_delay['is_rdp'] == False) & (
                        table_delay['N_or_T'] == 375 * task_arrival_itvl), 'dp_allocation_duration']
            assert total_task == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpft_375_dur_dp.csv', index=False, header=False, sep=',')

            sage_dur = table_delay.loc[ (table_delay['policy'] == 'RR-T')&(table_delay['is_rdp'] == False) & (table_delay['N_or_T'] == 50*task_arrival_itvl) , 'dp_allocation_duration']
            assert total_task == len(sage_dur)
            sage_dur.sort_values(ascending=True, inplace=True)
            sage_dur = pd.concat([sage_dur.reset_index(drop=True), pd.Series([1 / len(sage_dur)] * len(sage_dur)).cumsum().reset_index(drop=True)], axis=1, ignore_index=True)
            sage_dur.to_csv('graph2_data/rrt_dur_dp.csv', index=False, header=False, sep=',')

        if plot_rdp:
            dpf_l_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 14514), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpfn_14514_dur_rdp.csv', index=False, header=False, sep=',')

            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 25399), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpfn_25399_dur_rdp.csv', index=False, header=False, sep=',')


            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 30479), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpfn_30479_dur_rdp.csv', index=False, header=False, sep=',')



        # if plot_rdp:
            dpf_l_dur = table_delay.loc[(table_delay['policy'] == 'DPF-T') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 14514*task_arrival_itvl), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpft_14514_dur_rdp.csv', index=False, header=False, sep=',')

            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-T') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 25399*task_arrival_itvl), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpft_25399_dur_rdp.csv', index=False, header=False, sep=',')



            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-T') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 30479*task_arrival_itvl), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpft_30479_dur_rdp.csv', index=False, header=False, sep=',')
        #
        # fcfs_dur = table_delay.loc[(table_delay['policy'] == 'FCFS'), 'dp_allocation_duration']
        # fcfs_dur.sort_values(ascending=True, inplace=True)
        # fcfs_dur = pd.concat([fcfs_dur.reset_index(drop=True), pd.Series([1/len(fcfs_dur)]* len(fcfs_dur)).cumsum().reset_index(drop=True)] , axis=1, ignore_index=True)
        # fcfs_dur.to_csv('graph2_data/fcfs_dur.csv',index=False,header=False, sep=',')
        #
        #
        # total_task = len(fcfs_dur[0])
        # timeout = max(fcfs_dur[0])
        # assert timeout > 0
        #
        # dpf_l_dur = table_delay.loc[ (table_delay['policy'] == 'DPF-N') & (table_delay['N_or_T'] == 50) , 'dp_allocation_duration']
        # dpf_l_pad = total_task - len(dpf_l_dur)
        # dpf_l_dur = dpf_l_dur.append(pd.Series([timeout]*dpf_l_pad ),ignore_index=True)
        #
        # dpf_l_dur.sort_values(ascending=True, inplace=True)
        # dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True), pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)], axis=1, ignore_index=True)
        # dpf_l_dur.to_csv('graph2_data/dpfn_l_dur.csv', index=False, header=False, sep=',')
        #
        # dpf_h_dur = table_delay.loc[ (table_delay['policy'] == 'DPF-N') & (table_delay['N_or_T'] == 150) , 'dp_allocation_duration']
        # dpf_h_pad = total_task - len(dpf_h_dur)
        # dpf_h_dur = dpf_h_dur.append(pd.Series([timeout]*dpf_h_pad ),ignore_index=True)
        #
        #
        # dpf_h_dur.sort_values(ascending=True, inplace=True)
        # dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True), pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1 , ignore_index=True)
        # dpf_h_dur.to_csv('graph2_data/dpfn_h_dur.csv', index=False, header=False, sep=',')
        #
        #
        # sage_dur = table_delay.loc[ (table_delay['policy'] == 'Sage') & (table_delay['N_or_T'] == 50) , 'dp_allocation_duration']
        # sage_pad = total_task - len(sage_dur)
        # sage_dur = sage_dur.append(pd.Series([timeout] * sage_pad), ignore_index=True)
        #
        # sage_dur.sort_values(ascending=True, inplace=True)
        # sage_dur = pd.concat([sage_dur.reset_index(drop=True), pd.Series([1 / len(sage_dur)] * len(sage_dur)).cumsum().reset_index(drop=True)], axis=1, ignore_index=True)
        # sage_dur.to_csv('graph2_data/sage_dur.csv', index=False, header=False, sep=',')
        #
        #
        #
        #



    else:
        if plot_dp:
            dpf_l_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == False) & (
                        table_delay['N_or_T'] == 31), 'dp_allocation_duration']


            assert total_task == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpfn_31_dur_dp.csv', index=False, header=False, sep=',')


            dpf_l_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == False) & (
                    table_delay['N_or_T'] == 297), 'dp_allocation_duration']

            assert total_task == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)],
                                  axis=1,
                                  ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpfn_297_dur_dp.csv', index=False, header=False, sep=',')

        if plot_rdp:
            dpf_l_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == True) & (
                    table_delay['N_or_T'] == 8875), 'dp_allocation_duration']

            assert total_task == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)],
                                  axis=1,
                                  ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpfn_8875_dur_rdp.csv', index=False, header=False, sep=',')





            dpf_l_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 14514), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpfn_14514_dur_rdp.csv', index=False, header=False, sep=',')

            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-N') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 27512), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpfn_27512_dur_rdp.csv', index=False, header=False, sep=',')











            dpf_l_dur = table_delay.loc[(table_delay['policy'] == 'DPF-T') & (table_delay['is_rdp'] == True) & (
                    table_delay['N_or_T'] == 8875*task_arrival_itvl_heavy), 'dp_allocation_duration']

            assert total_task == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)],
                                  axis=1,
                                  ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpft_8875_dur_rdp.csv', index=False, header=False, sep=',')

            dpf_l_dur = table_delay.loc[(table_delay['policy'] == 'DPF-T') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 14514*task_arrival_itvl_heavy), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_l_dur)
            dpf_l_dur.sort_values(ascending=True, inplace=True)
            dpf_l_dur = pd.concat([dpf_l_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_l_dur)] * len(dpf_l_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_l_dur.to_csv('graph2_data/dpft_14514_dur_rdp.csv', index=False, header=False, sep=',')

            dpf_h_dur = table_delay.loc[(table_delay['policy'] == 'DPF-T') & (table_delay['is_rdp'] == True) & (
                        table_delay['N_or_T'] == 27512*task_arrival_itvl_heavy), 'dp_allocation_duration']
            assert total_task_rdp == len(dpf_h_dur)
            dpf_h_dur.sort_values(ascending=True, inplace=True)
            dpf_h_dur = pd.concat([dpf_h_dur.reset_index(drop=True),
                                   pd.Series([1 / len(dpf_h_dur)] * len(dpf_h_dur)).cumsum().reset_index(drop=True)], axis=1,
                                  ignore_index=True)
            dpf_h_dur.to_csv('graph2_data/dpft_27512_dur_rdp.csv', index=False, header=False, sep=',')

