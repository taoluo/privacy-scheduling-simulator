from data_collection import workspace2dataframe
from graph1 import plot_graph1
from graph2 import plot_graph2
from graph3 import plot_graph3
from graph4 import plot_graph4

if __name__ == '__main__':
    # workspace_dir = "workspace_12-04-23H-41-58"
    # good single block snapshot
    # workspace_dir = "workspace_12-07-12H-56-59"
    # workspace_dir = "workspace_04-20-21H-56-57" # ticker = 5 task arrival
    # workspace_dir = "workspace_04-21-05H-51-59" # ticker = 1 task arrival
#     workspace_dir = "workspace_04-22-04H-03-19" # fix rdp task arrival rate

#    workspace_dir = "workspace_04-22-08H-23-18/" # fix rdp task arrival rate
#     workspace_dir = "workspace_04-23-22H-27-04"  # double sim duration for rdp
#     workspace_dir = "workspace_04-25-00H-15-48" # longer dp lifetime
#     workspace_dir = "workspace_04-25-21H-48-19" # longer dp lifetime
    workspace_dir = "workspace_04-28-12H-09-57" # longer dp lifetime, same/high arrival rate for all dp and rdp
    # ('./workspace_04-28-12H-09-57/242', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])
    # ('./workspace_04-28-12H-09-57/355', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])
    # ('./workspace_04-28-12H-09-57/244', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])
    # ('./workspace_04-28-12H-09-57/241', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])
    # ('./workspace_04-28-12H-09-57/243', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])
    # ('./workspace_04-28-12H-09-57/237', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])
    # ('./workspace_04-28-12H-09-57/240', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])
    # ('./workspace_04-28-12H-09-57/235', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])
    # ('./workspace_04-28-12H-09-57/236', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])
    # ('./workspace_04-28-12H-09-57/238', [], ['sim.sqlite-journal', 'sim.log', 'sim.sqlite', 'sim.gtkw'])

    # journal
    # │   └──./ 240 / sim.sqlite - journal
    # │   └──./ 241 / sim.sqlite - journal
    # │   └──./ 242 / sim.sqlite - journal
    # │   └──./ 243 / sim.sqlite - journal
    # │   └──./ 244 / sim.sqlite - journal
# high workload
    workspace_dir = "workspace_04-29-12H-21-39" # longer dp lifetime, same/high arrival rate for all dp and rdp
    # workspace_dir = "workspace_05-10-13H-44-35" # heavy_workload dp rdp
    # workspace_dir = "workspace_05-10-14H-30-32" # heavy_workload alighed N, RDP VS DP
    # workspace_dir = "workspace_05-10-22H-11-13" # more data sample for dp dpf

    # workspace_dir = "workspace_05-21-11H-11-02" # longer dp lifetime, same/high arrival rate for all dp and rdp

# low workload
    # for paper sub2 to shepherd
    # workspace_dir = "workspace_05-06-12H-41-39" # longer dp lifetime, same/high arrival rate for all dp and rdp
    workspace_dir="workspace_06-12-17H-53-31"

    table = workspace2dataframe(workspace_dir)
    # print(table.loc[(table['policy'] == 'DPF-T' )&( table['N_or_T'] >= 100)&table['is_rdp'], 'sim_index'])

    # print("graph0")
    plot_graph1(table, 'graph1.pdf')
    print("graph1")
    plot_graph2(table, 'graph2.pdf',task_timeout=300)
    print("graph2")
    plot_graph3(table, 'graph3.pdf')
    print("graph3")
    plot_graph4(table, 'graph4.pdf',task_timeout=300)
    print("graph4")




