# from .data_collection import *
import seaborn as sns
import matplotlib.pyplot as plt
import sys
# sys.path.append('/foo/bar/mock-0.3.1')
# def _plot_and_save_subgraph(table,columns):
plot_rdp = True
plot_dp = False
def plot_graph1(table,save_file_name):
    should_plot_per_task_type = True
    table = table.copy()
    title = "fig1: multi block, # of tasks completed as a function of N/T"


    table1 = table.loc[(table['blocks_mice_fraction'] == 75) & (table['epsilon_mice_fraction'] == 75) ]
    # table1.reset_index(inplace=True)
    # print(table1.columns)
    # table1.dtypes
    # table1['granted_tasks_total'] = table1['granted_tasks_total'].astype(float)
    plot_and_save(save_file_name, table1, title,column_to_plot='granted_tasks_total')

    if should_plot_per_task_type:
        col = 'num_granted_tasks_l_dp_l_blk'
        plot_and_save(save_file_name.split('.')[0]+'-%s.pdf' % col, table1, title, column_to_plot=col)
        col = 'num_granted_tasks_l_dp_s_blk'
        plot_and_save(save_file_name.split('.')[0]+'-%s.pdf' % col, table1, title, column_to_plot=col)
        col = 'num_granted_tasks_s_dp_l_blk'
        plot_and_save(save_file_name.split('.')[0]+'-%s.pdf' % col, table1, title, column_to_plot=col)
        col = 'num_granted_tasks_s_dp_s_blk'
        plot_and_save(save_file_name.split('.')[0]+'-%s.pdf' % col, table1, title, column_to_plot=col)

    # save_csv_df_rdp = table1[table1.is_rdp==True].pivot(index='N_or_T', values='granted_tasks_total', columns='policy')
    # # padding
    # save_csv_df_rdp.loc[:, 'FCFS'] = save_csv_df_rdp['FCFS'].loc[-1]
    # save_csv_df_rdp.drop(labels=[-1], axis=0, inplace=True)
    # save_csv_df_rdp[save_csv_df_rdp.N_or_T_based == 'N'].to_csv('graph1_data/graph_rdp_N.csv',sep=',')
    # save_csv_df_rdp[save_csv_df_rdp.N_or_T_based == 'T'].to_csv('graph1_data/graph_rdp_T.csv', sep=',')
    # # with open('graph1_data/graph_rdp.csv','w') as f:
    # #     save_csv_df_rdp.to_csv(f,sep=',')
    # save_csv_df_dp = save_table_rdp_NT(table1,False, 'N')
    # save_csv_df_dp[save_csv_df_dp.N_or_T_based == 'N'].to_csv('graph1_data/graph_dp_N.csv', sep=',')
    # # with open('graph1_data/graph_dp.csv','w') as f:
    # #     save_csv_df_dp.to_csv(f,sep=',')
    if plot_rdp:
        save_table_rdp_NT(table1, True, 'T')
        save_table_rdp_NT(table1, True, 'N')
    if plot_dp:
        save_table_rdp_NT(table1, False, 'T')
        save_table_rdp_NT(table1, False, 'N')


def save_table_rdp_NT(table1,is_rdp,N_or_T_based):

    save_csv_df_dp = table1[(table1.is_rdp == is_rdp) & (table1.N_or_T_based == N_or_T_based)].pivot(index='N_or_T',
                                                                                           values='granted_tasks_total',
                                                                                           columns='policy')
    # padding
    save_csv_df_dp['FCFS'] = table1.loc[(table1.is_rdp == is_rdp)&(table1.policy=='FCFS'),'granted_tasks_total'].iloc[0]
    # save_csv_df_dp.drop(labels=[-1], axis=0, inplace=True)
    name_part = 'graph1_data/graph_rdp_' if is_rdp else 'graph1_data/graph_dp_'
    save_csv_df_dp.to_csv(name_part + N_or_T_based + '.csv', sep=',')
    return save_csv_df_dp


def plot_and_save(save_file_name, table1, title, column_to_plot):
    facetgrid = sns.relplot(data=table1, x='N_or_T', y=column_to_plot, row='is_rdp', col='N_or_T_based',
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
    if plot_dp and plot_rdp:
        y2_hline_val = table1[((table1.policy == 'FCFS') & (table1.is_rdp == False))][column_to_plot].iloc[0]
        facetgrid.axes[0][0].axhline(y=y2_hline_val, color='red', linestyle=':')
        facetgrid.axes[0][1].axhline(y=y2_hline_val, color='red', linestyle=':')

        y2_hline_val2 = table1[((table1.policy == 'FCFS') & (table1.is_rdp == True))][column_to_plot].iloc[0]
        facetgrid.axes[1][0].axhline(y=y2_hline_val2, color='red', linestyle=':')
        facetgrid.axes[1][1].axhline(y=y2_hline_val2, color='red', linestyle=':')
    plt.subplots_adjust(top=0.8)
    facetgrid.fig.suptitle(title)
    facetgrid.savefig(save_file_name)


