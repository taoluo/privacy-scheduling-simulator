import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot_graph3(table,save_file_name):
    table = table.copy()
    table.loc[table['policy']=='FCFS',"N_or_T_based"] = 'T'

    tt = table[table['policy'] == 'FCFS'].copy()
    tt['N_or_T_based'] = 'N'
    table = table.append(tt, ignore_index=True)

    title =\
        'fig3: Single block, number of tasks completed as function of Mice percentage: FCFS, Sage, DPF-N'
    fixed_N_or_T_dp = 375
    task_arrival_itvl_light = 0.078125
    task_arrival_itvl_heavy = 0.004264781
    fixed_N_or_T_dp2 = fixed_N_or_T_dp * task_arrival_itvl_heavy
    fixed_N_or_T_rdp = 14514 # fixme !!

    fixed_N_or_T_rdp2 =  fixed_N_or_T_rdp * task_arrival_itvl_heavy
    table1 = table.loc[(table['blocks_mice_fraction'] == 75) & ( (table['N_or_T'] == fixed_N_or_T_dp) |(table['N_or_T'] == fixed_N_or_T_dp2) |(table['N_or_T'] == fixed_N_or_T_rdp)|(table['N_or_T'] == fixed_N_or_T_rdp2)| (table['N_or_T'] == -1  ))]

    # table1.reset_index(inplace=True)
    # print(table1.columns)
    # table1.dtypes
    # table1['granted_tasks_total'] = table1['granted_tasks_total'].astype(float)
    # print('N = 2048')
    g = sns.relplot(data=table1, x = 'epsilon_mice_fraction', y = 'granted_tasks_total', row='is_rdp',col='N_or_T_based',style='policy', hue='policy', size='policy', kind='line',  facet_kws = dict(sharey=False))
    # for ax in g.axes[0, :]:
    #     ax.set_ylim(-5, 110 )
    #
    #
    # for ax in g.axes[1, :]:
    #     ax.set_ylim(-5, 17000)


    plt.subplots_adjust(top=0.8)
    g.fig.suptitle('fixed_N_or_T == %d or %d' % (fixed_N_or_T_dp ,fixed_N_or_T_rdp)+ title )
    g.savefig( save_file_name)
    table1 = table1[~((table1['policy']=='FCFS' )&( table1['N_or_T_based']=='T' ))]
    save_csv_df = table1[table1.is_rdp==True].pivot(index='epsilon_mice_fraction', values='granted_tasks_total', columns='policy')
    save_csv_df.to_csv('graph3_data/graph_rdp.csv',sep=',')

    save_csv_df = table1[table1.is_rdp == False].pivot(index='epsilon_mice_fraction', values='granted_tasks_total',
                                                      columns='policy')
    save_csv_df.to_csv('graph3_data/graph_dp.csv', sep=',')
