from DP_simulator import *
import shutil
import os
import copy


if __name__ == '__main__':
    # workload with contention

    # rdp_arrival_itvl = 0.004264781

    # N_dp = 100
    # T_dp = rdp_arrival_itvl * N_dp
    #
    # N_rdp = 15000 # 14514
    # T_rdp = rdp_arrival_itvl * N_rdp
    eval_dp = False
    eval_rdp = True
    config = {
        'workload_test.enabled': False,
        'workload_test.workload_trace_file': '/home/tao2/desmod/docs/examples/DP_allocation/workloads.yaml',
        'task.demand.num_blocks.mice': 1,
        'task.demand.num_blocks.elephant': 10,
        'task.demand.num_blocks.mu': 20,
        'task.demand.num_blocks.sigma': 10,
        # num_blocks * 1/mean_tasks_per_block = 1/10
        'task.demand.epsilon.mean_tasks_per_block': 15,
        'task.demand.epsilon.mice': 1e-2,  # N < 100
        'task.demand.epsilon.elephant': 1e-1,
        'task.completion_time.constant': 0,  # finish immediately
        # max : half of capacity
        'task.completion_time.max': 100,
        # 'task.completion_time.min': 10,
        'task.demand.num_cpu.constant': 1,  # int, [min, max]
        'task.demand.num_cpu.max': 80,
        'task.demand.num_cpu.min': 1,
        'task.demand.size_memory.max': 412,
        'task.demand.size_memory.min': 1,
        'task.demand.num_gpu.max': 3,
        'task.demand.num_gpu.min': 1,
        'resource_master.block.init_epsilon': 1.0,  # normalized
        'resource_master.block.init_delta': 1.0e-6,  # only used for rdp should be small < 1
        'resource_master.dp_policy.is_rdp.mechanism': GAUSSIAN,
        'resource_master.dp_policy.is_rdp.rdp_bound_mode': PARTIAL_BOUNDED_RDP,  # PARTIAL_BOUNDED_RDP
        'resource_master.dp_policy.is_rdp.dominant_resource_share': DPS_RDP_alpha_all,
        'resource_master.block.arrival_interval': 10,
        # 'resource_master.dp_policy': DP_POLICY_RATE_LIMIT,
        # 'resource_master.dp_policy.dpf_family.dominant_resource_share': DRS_L_INF,  # DRS_L2
        'resource_master.dp_policy.dpf_family.dominant_resource_share': DPS_DP_L_Inf,  # DRS_L2
        'resource_master.dp_policy.dpf_family.grant_top_small': False,  # false: best effort alloc
        # only continous leading small tasks in queue are granted
        # DP_POLICY_FCFS
        # DP_POLICY_RR_T  DP_POLICY_RR_N DP_POLICY_RR_NN
        # DP_POLICY_DPF_N DP_POLICY_DPF_T  DP_POLICY_DPF_NA
        # todo fcfs rdp
        # 'resource_master.dp_policy': DP_POLICY_FCFS,
        # policy
        'sim.duration': '300 s',
        'task.timeout.interval': 21,
        'task.timeout.enabled': False,
        'task.arrival_interval': 1,
        'resource_master.dp_policy.is_admission_control_enabled': False,
        'resource_master.dp_policy.is_rdp': True,
        'resource_master.dp_policy': None,
        'resource_master.dp_policy.denominator': None,
        'resource_master.block.lifetime': None,  # policy level param
        # workload
        'resource_master.block.is_static': False,
        'resource_master.block.init_amount': 11,  # for block elephant demand
        'task.demand.num_blocks.mice_percentage': 75.0,
        'task.demand.epsilon.mice_percentage': 75.0,
        # https://cloud.google.com/compute/docs/gpus
        # V100 VM instance
        'resource_master.is_cpu_needed_only': True,
        'resource_master.cpu_capacity': sys.maxsize,  # number of cores
        'resource_master.memory_capacity': 624,  # in GB, assume granularity is 1GB
        'resource_master.gpu_capacity': 8,  # in cards
        'resource_master.clock.tick_seconds': 25,
        'resource_master.clock.dpf_adaptive_tick': True,
        'sim.db.enable': True,
        'sim.db.persist': True,
        'sim.dot.colorscheme': 'blues5',
        'sim.dot.enable': False,
        # 'sim.duration': '300000 s',  # for rdp
        'sim.runtime.timeout': 60 * 24,  # in min
        # 'sim.duration': '10 s',
        'sim.gtkw.file': 'sim.gtkw',
        'sim.gtkw.live': False,
        'sim.log.enable': True,
        "sim.log.level": "DEBUG",
        'sim.progress.enable': True,
        'sim.result.file': 'result.json',
        'sim.seed': 3345,  # 23338,
        'sim.timescale': 's',
        'sim.vcd.dump_file': 'sim_dp.vcd',
        'sim.vcd.enable': False,
        'sim.vcd.persist': False,
        'sim.workspace': 'exp_result/workspace_%s'
        % datetime.now().strftime("%m-%d-%HH-%M-%S"),
        'sim.workspace.overwrite': True,
    }
    ## single block setup for reproduce result of submission
    config['resource_master.block.is_static'] = False
    config['resource_master.block.init_amount'] = 11
    # config['task.arrival_interval'] = 1 # unit
    config['task.demand.num_blocks.mice_percentage'] = 75

    mice_fraction = [0, 25, 50, 75, 100]
    # timeout = 50

    from collections import namedtuple

    class Config(object):
        def __init__(self):
            pass
            # self.is_rdp =
            # self.sim_duration
            # self.dp_policy
            # self.denominator
            # self.block_lifetime

    config_list = []
    # pure DP
    dp_max_amount = 100
    dp_subconfig = Config()
    dp_subconfig.is_rdp = False
    aligned_N = None  # [int(10 * 3.1 ** x) for x in range(8)]
    dp_arrival_itvl_light = 0.078125
    dp_arrival_itvl_heavy = 0.004264781
    dp_subconfig.dp_arrival_itvl = 0.078125  # dp contention point
    dp_subconfig.dp_arrival_itvl = dp_arrival_itvl_heavy  # rdp contention point
    N_scale_factor = [0.10, 0.50, 0.75, 1.00, 1.25, 1.75, 2.1]  # for rdp and dp
    N_scale_factor_ext = [2.75, 3.25, 3.75, 7.5, 11.25, 15, 30, 60, 100]  # for dp only
    # num_arrivals_multiplier = 2.0 # for sim_duration actual arrived tasks / max allocable tasks
    # assert num_arrivals_multiplier in N_scale_factor
    dp_subconfig.dp_policy = [
        DP_POLICY_FCFS,
        DP_POLICY_DPF_T,
        DP_POLICY_DPF_N,
        DP_POLICY_RR_T,
        DP_POLICY_RR_NN,
    ]
    DP_N = dp_subconfig.denominator = [1] + [
        dp_max_amount * i for i in N_scale_factor + N_scale_factor_ext
    ]
    if aligned_N is not None:
        DP_N = dp_subconfig.denominator = aligned_N
    DP_T = dp_subconfig.block_lifetime = [
        N * dp_subconfig.dp_arrival_itvl for N in DP_N
    ]
    dp_subconfig.sim_duration = '%d s' % (
        config['resource_master.block.arrival_interval'] * 30
    )
    # dp_subconfig.sim_duration = '%d s' % (dp_subconfig.dp_arrival_itvl * dp_max_amount * num_arrivals_multiplier)
    # dp_subconfig.sim_duration = '11 s'
    # dp_timeout = 50 * dp_max_amount * config['task.arrival_interval'] * config[
    #     'task.demand.epsilon.mice']  # at max, 100 tasks waiting in the queue
    # 10 is the multiplier of dp between elephant and mice
    dp_timeout = (
        3 * (4 + 1) * 100 * dp_subconfig.dp_arrival_itvl
    )  # at max, 1500 tasks waiting in the queue
    if eval_dp:
        for p in dp_subconfig.dp_policy:
            if p == DP_POLICY_FCFS:
                config_list.extend(
                    list(
                        product(
                            [dp_subconfig.dp_arrival_itvl],
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            [None],
                            [None],
                        )
                    )
                )
            elif p == DP_POLICY_DPF_T:
                config_list.extend(
                    list(
                        product(
                            [dp_subconfig.dp_arrival_itvl],
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            [None],
                            DP_T,
                        )
                    )
                )
            elif p == DP_POLICY_DPF_N:
                config_list.extend(
                    list(
                        product(
                            [dp_subconfig.dp_arrival_itvl],
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            DP_N,
                            [None],
                        )
                    )
                )
            elif p == DP_POLICY_RR_T:
                config_list.extend(
                    list(
                        product(
                            [dp_subconfig.dp_arrival_itvl],
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            [None],
                            DP_T,
                        )
                    )
                )
            elif p == DP_POLICY_RR_NN:
                config_list.extend(
                    list(
                        product(
                            [dp_subconfig.dp_arrival_itvl],
                            [dp_timeout],
                            [dp_subconfig.is_rdp],
                            [dp_subconfig.sim_duration],
                            [p],
                            DP_N,
                            [None],
                        )
                    )
                )
            else:
                raise Exception()

    # RDP
    is_rdp = True
    rdp_max_amount = 14514
    # RDP_N = [int(n/100*max_amount) for n in DP_N]
    # RDP_T = [N * config['task.arrival_interval'] for N in RDP_N]
    # rdp_duration = max(RDP_T)
    # rdp_policy = [DP_POLICY_FCFS, DP_POLICY_DPF_T, DP_POLICY_DPF_N]
    #
    rdp_subconfig = Config()
    rdp_subconfig.is_rdp = True
    rdp_subconfig.rdp_arrival_itvl = dp_arrival_itvl_heavy  # contention point
    rdp_subconfig.dp_policy = [DP_POLICY_FCFS, DP_POLICY_DPF_T, DP_POLICY_DPF_N]
    RDP_N = rdp_subconfig.denominator = [1] + [
        int(rdp_max_amount * n) for n in N_scale_factor
    ]

    if aligned_N is not None:
        RDP_N = rdp_subconfig.denominator = aligned_N
    RDP_T = rdp_subconfig.block_lifetime = [
        N * rdp_subconfig.rdp_arrival_itvl for N in RDP_N
    ]
    rdp_subconfig.sim_duration = '%d s' % (
        config['resource_master.block.arrival_interval'] * 30
    )
    # rdp_subconfig.sim_duration = '%d s' % (rdp_subconfig.rdp_arrival_itvl* rdp_max_amount * num_arrivals_multiplier)
    # rdp_subconfig.sim_duration = '11 s'

    # 100 is the multiplier of dp between elephant and mice
    rdp_timeout = (
        3 * (4 + 1) * 100 * rdp_subconfig.rdp_arrival_itvl
    )  # at max, 1500 tasks waiting in the queue
    if eval_rdp:
        for p in rdp_subconfig.dp_policy:
            if p == DP_POLICY_FCFS:
                config_list.extend(
                    list(
                        product(
                            [rdp_subconfig.rdp_arrival_itvl],
                            [rdp_timeout],
                            [rdp_subconfig.is_rdp],
                            [rdp_subconfig.sim_duration],
                            [p],
                            [None],
                            [None],
                        )
                    )
                )
            elif p == DP_POLICY_DPF_T:
                config_list.extend(
                    list(
                        product(
                            [rdp_subconfig.rdp_arrival_itvl],
                            [rdp_timeout],
                            [rdp_subconfig.is_rdp],
                            [rdp_subconfig.sim_duration],
                            [p],
                            [None],
                            RDP_T,
                        )
                    )
                )
            elif p == DP_POLICY_DPF_N:
                config_list.extend(
                    list(
                        product(
                            [rdp_subconfig.rdp_arrival_itvl],
                            [rdp_timeout],
                            [rdp_subconfig.is_rdp],
                            [rdp_subconfig.sim_duration],
                            [p],
                            RDP_N,
                            [None],
                        )
                    )
                )
            # elif p == DP_POLICY_RR_T:
            #     config_list.append(product([rdp_subconfig.is_rdp],[rdp_subconfig.sim_duration],[p],[None],[RDP_T]))
            # elif p == DP_POLICY_RR_NN:
            #     config_list.append(product([rdp_subconfig.is_rdp],[rdp_subconfig.sim_duration],[p],[RDP_N],[None]))
            else:
                raise Exception()

    real_config_fields = [
        'task.arrival_interval',
        'task.timeout.interval',
        'resource_master.dp_policy.is_rdp',
        'sim.duration',
        'resource_master.dp_policy',
        'resource_master.dp_policy.denominator',
        'resource_master.block.lifetime',
    ]

    # mice_fraction = [0, 25, 50, 75, 100]
    # timeout = 50
    test_factors = [
        (real_config_fields, config_list),  # 250000
        (['task.demand.epsilon.mice_percentage'], [[pct] for pct in mice_fraction]),
    ]
    load_filter = lambda x: True
    simulate_factors(config, test_factors, Top, config_filter=load_filter)

    # shutil.copyfile(__file__, os.path.join(config['sim.workspace'], 'saved_config.py'))


