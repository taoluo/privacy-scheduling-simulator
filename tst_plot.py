from analysis.data_collection import workspace2dataframe
from analysis.plot import plot_granted_tasks

if __name__ == '__main__':
    workspace_file = "workspace_06-13-18H-02-36"
    workspace_dir="./results/%s" % workspace_file
    # workspace_dir = ""
    table = workspace2dataframe(workspace_dir)
    # table = table.loc[(table['blocks_mice_fraction'] == 75) & (table['epsilon_mice_fraction'] == 75)]
    title = "fig1: multi block, %s completed as a function of N/T"
    table1 = table.loc[(table['blocks_mice_fraction'] == 100) & (table['epsilon_mice_fraction'] == 75) ]
    plot_granted_tasks(save_file_name="test_plt.pdf", table=table1, title='titles', column_to_plot='granted_tasks_total')

