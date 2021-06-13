import DP_simulator
from DP_simulator import *
from utils.configs import *
from datetime import datetime

if __name__ == '__main__':

    # run_test_single = True
    # run_test_by_factor = False

    run_test_single = False
    run_test_by_factor = True

    dp_arrival_itvl = 0.078125
    rdp_arrival_itvl = 0.004264781
    N_dp = 100
    T_dp = rdp_arrival_itvl * N_dp

    N_rdp = 14514
    T_rdp = rdp_arrival_itvl * N_rdp
    T_rdp = dp_arrival_itvl * N_rdp

    run_test_many = False
    run_test_parallel = False
    run_factor = False
    is_factor_single_block = False
    is_factor_rdp = False



    config = {
        'workload_test.enabled': False,
        'workload_test.workload_trace_file': os.path.abspath('./workloads.yaml'),
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
        'resource_master.block.init_delta': 1.0e-6,  #  only used for rdp should be small < 1
        'resource_master.dp_policy.is_rdp.mechanism': RdpMechanism.GAUSSIAN,
        'resource_master.dp_policy.is_rdp.rdp_bound_mode': RdpBoundMode.PARTIAL_BOUNDED_RDP,  # PARTIAL_BOUNDED_RDP
        'resource_master.dp_policy.is_rdp.dominant_resource_share': DominantRdpShareType.DPS_RDP_alpha_positive_budget,
        'resource_master.block.arrival_interval': 10,
        # 'resource_master.dp_policy': DpPolicyType.DP_POLICY_RATE_LIMIT,
        # 'resource_master.dp_policy.dpf_family.dominant_resource_share': DRS_L_INF,  # DRS_L2
        'resource_master.dp_policy.dpf_family.dominant_resource_share': DominantDpShareType.DPS_DP_L_Inf,  # DRS_L2
        'resource_master.dp_policy.dpf_family.grant_top_small': False,  # false: best effort alloc
        # only continous leading small tasks in queue are granted
        # DpPolicyType.DP_POLICY_FCFS
        # DpPolicyType.DP_POLICY_RR_T  DpPolicyType.DP_POLICY_RR_N DpPolicyType.DP_POLICY_RR_NN
        # DpPolicyType.DP_POLICY_DPF_N DpPolicyType.DP_POLICY_DPF_T  DpPolicyType.DP_POLICY_DPF_NA
        # todo fcfs rdp
        # 'resource_master.dp_policy': DpPolicyType.DP_POLICY_FCFS,
        # policy
        'sim.main_file': __file__,
        'sim.duration': '12 s',
        'task.timeout.interval': 51,  # block arrival x2??
        'task.timeout.enabled': True,
        'task.arrival_interval': rdp_arrival_itvl,
        'resource_master.dp_policy.is_admission_control_enabled': False,
        'resource_master.dp_policy.is_rdp': False,
        'resource_master.dp_policy': DpPolicyType.DP_POLICY_DPF_N,
        'resource_master.dp_policy.denominator': N_rdp,
        'resource_master.block.lifetime': T_rdp,  # policy level param
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
        # 'resource_master.clock.tick_seconds': 25,
        'resource_master.clock.dpf_adaptive_tick': True,
        'sim.numerical_delta':1e-8,
        'sim.instant_timeout': 0.01,
        'sim.db.enable': True,
        'sim.db.persist': True,
        'sim.dot.colorscheme': 'blues5',
        'sim.dot.enable': False,
        # 'sim.duration': '300000 s',  # for rdp
        'sim.runtime.timeout': 60,  # in min
        # 'sim.duration': '10 s',
        'sim.gtkw.file': 'sim.gtkw',
        'sim.gtkw.live': False,
        'sim.log.enable': True,
        "sim.log.level": "DEBUG",
        'sim.progress.enable': True,
        'sim.result.file': 'result.json',
        'sim.seed': 23338,
        'sim.timescale': 's',
        'sim.vcd.dump_file': 'sim_dp.vcd',
        'sim.vcd.enable': True,
        'sim.vcd.persist': True,
        'sim.workspace': 'results/workspace_%s' % datetime.now().strftime("%m-%d-%HH-%M-%S"),
        'sim.workspace.overwrite': True,
    }
    sample_val = (
        1,
        2,
        3,
        4,
        5,
    )  # 6, 7,8,9,10)  # in 1 - 10
    b_genintvl = config['resource_master.block.arrival_interval']  # 10 sec
    # load: contention from low to high
    # 2 ^ (-1.5 ^ x)
    # mice >~= half

    # option 2
    if is_factor_single_block:
        blk_nr_mice_pct = [100]  # all block mice
    else:
        # 99.5 - 0.27 %
        # blk_nr_mice_pct = [ (1- 2 ** (- 1.5** i )) for i in (5,4,3,2,1,0,-1,-2)]
        # blk_nr_mice_pct = [100, 75, 50, 25, 0]  # all block mice
        blk_nr_mice_pct = [75]  # all block mice

    # epsilon_mice_pct = [ (1 - 2 ** (- 1.5** i ) )for i in (5,4,3,2,1,0,-1,-2)]

    # epsilon_mice_pct = [95, 75, 50, 25, 5]
    epsilon_mice_pct = [100, 75, 50, 25, 0]

    if is_factor_single_block:
        epsilon_mice_pct = [100, 75, 50, 25, 0]
    else:
        epsilon_mice_pct = [75]

    if is_factor_single_block:
        # option 2
        t_intvls = [
            1,
        ]  # treat as time unit
    else:  # ??
        # [16, 64, 128, 256, ] per b_genintvl (10s)
        # t_intvl = [b_genintvl * (2 ** -i) for i in ( 4, 6, 7, 8)]
        t_intvls = [b_genintvl * (2 ** -i) for i in (6, 7,)]

    def load_filter(conf):
        # assert stress_factor in ("blk_nr","epsilon","task_arrival" )
        blk_nr_filter = (
            lambda c: c['task.demand.epsilon.mice_percentage'] == epsilon_mice_pct[0]
            and c['task.arrival_interval'] == t_intvls[0]
        )
        epsilon_filter = (
            lambda c: c['task.demand.num_blocks.mice_percentage'] == blk_nr_mice_pct[0]
            and c['task.arrival_interval'] == t_intvls[0]
        )
        task_arrival_filter = (
            lambda c: c['task.demand.epsilon.mice_percentage'] == epsilon_mice_pct[0]
            and c['task.demand.num_blocks.mice_percentage'] == blk_nr_mice_pct[0]
        )

        # filters = {"blk_nr":blk_nr_filter, "epsilon":epsilon_filter, "task_arrival":task_arrival_filter}
        return blk_nr_filter(conf) or epsilon_filter(conf) or task_arrival_filter(conf)

    flip_coin = random.Random(x=23425453)

    def sparse_load_filter(conf):

        # idx_sum = sum(blk_nr_mice_pct.index(conf[ 'task.demand.num_blocks.mice_percentage' ]) + epsilon_mice_pct.index(conf[ 'task.demand.epsilon.mice_percentage' ]) + t_intvl.index(conf[ 'task.arrival_interval']))
        return flip_coin.randint(1, 10) in sample_val

    if is_factor_single_block:
        # assume 1 sec interarrival
        # option 2
        #   [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        # b_lifeintvl = [int(2 ** i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)]
        b_lifeintvl = [10, 50, 75, 100, 125, 150, 175, 200, 225]  # 256]

    else:
        # policy
        # [0.25, 1, 4, 16, 64, 256, // 1024]
        # b_lifeintvl = [b_genintvl * (4 ** i) for i in (-1, 0, 1, 2, 3, 4)]  # 5)]
        t_intvl_copy = t_intvls  # 2**-7 * 10
        b_lifeintvl = [b_genintvl * (2 ** -j) for j in [1, 2, 3, 4, 5, 6]] + [
            b_genintvl * i for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
        ]
        # b_lifeintvl = [30, 100]

    # if is_single_block:
    # option 2
    #   [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # b_N_total = [ int(i / t_intvl[0]) for i in b_lifeintvl]  # b_lifeintvl/1sec
    # b_N_total = [ 10, 50, 75, 100, 125, 150,175, 256]

    # else:
    # [1, 4, 16, 64, 256, 1024, 4096]
    # fixme
    # b_N_total = [ int(i/t_intvl_copy) for i in b_lifeintvl]
    # b_N_total = [ 10, 50, 75, 100, 125, 150,175, 256]

    # b_N_total = [(4 ** i) for i in (0, 1, 2, 3, 4, 5, 6)]

    if is_factor_single_block:
        is_static_block = True
        init_blocks = 1
    else:
        is_static_block = False
        init_blocks = config['task.demand.num_blocks.elephant']

    if is_factor_single_block:
        sim_duration = round(max(b_lifeintvl) * 1.05)  # long enough
        # task_timeout = [ max(i * 0.2, 3 * t_intvl[0]) for i in b_lifeintvl ]
        task_timeout_global = 50
        task_timeout = [task_timeout_global]
    else:
        #  10 * 20 * 10 sec
        sim_duration = 10 * config['task.demand.num_blocks.elephant'] * b_genintvl
        task_timeout_global = 50
        task_timeout = [task_timeout_global]

    # policy, T, N
    dpf_t_factors = product(
        t_intvls, [DpPolicyType.DP_POLICY_DPF_T], b_lifeintvl, [None], [task_timeout_global]
    )
    rate_limit_factors = product(
        t_intvls, [DpPolicyType.DP_POLICY_RR_T], b_lifeintvl, [None], [task_timeout_global]
    )
    # zip(repeat(DpPolicyType.DP_POLICY_RATE_LIMIT), b_lifeintvl, repeat(None)) , task_timeout)
    dpf_n_factors = product(
        t_intvls, [DpPolicyType.DP_POLICY_DPF_N], b_lifeintvl, [None], [task_timeout_global]
    )
    dpf_n_factors = filter(
        lambda x: x[3] > 0,
        map(lambda x: (x[0], x[1], x[2], int(x[2] / x[0]), x[4]), dpf_n_factors),
    )
    # for i in dpf_n_factors:
    #     if i[3] <= 0:
    #         print(i)
    if is_factor_single_block:
        load_filter = lambda x: True
    else:
        load_filter = sparse_load_filter
        load_filter = lambda x: True

    factors = [
        (['sim.duration'], [['%d s' % 10000]]),
        (['resource_master.block.is_static'], [[is_static_block]]),
        (['resource_master.block.init_amount'], [[init_blocks]]),
        (['task.demand.num_blocks.mice_percentage'], [[i] for i in blk_nr_mice_pct]),
        (['task.demand.epsilon.mice_percentage'], [[75]]),
        # (['task.arrival_interval'], [[i] for i in t_intvl]),
        (
            [
                'task.arrival_interval',
                'resource_master.dp_policy',
                'resource_master.block.lifetime',
                'resource_master.dp_policy.denominator',
                'task.timeout.interval',
            ],
            list(
                chain(
                    product(
                        t_intvls, [DpPolicyType.DP_POLICY_FCFS], [0.0], [0.0], [task_timeout_global]
                    ),
                    dpf_n_factors,
                    dpf_t_factors,
                    rate_limit_factors,
                )
            ),
        ),  # rate_limit_factors, dpf_t_factors, dpf_n_factors,
    ]

    factors_rdp = [
        (['sim.duration'], [['%d s' % 10000]]),
        (['resource_master.block.is_static'], [[is_static_block]]),
        (['resource_master.block.init_amount'], [[init_blocks]]),
        (
            ['task.demand.num_blocks.mice_percentage'],
            [[100,]],
        ),  # [[i] for i in blk_nr_mice_pct]),
        # (['task.demand.epsilon.mice_percentage'], [[i] for i in epsilon_mice_pct]), # [[75,]] ), #
        (['resource_master.dp_policy.is_rdp.mechanism'], [[RdpMechanism.GAUSSIAN,]]),
        (
            ['resource_master.dp_policy.is_rdp.rdp_bound_mode'],
            [[RdpBoundMode.PARTIAL_BOUNDED_RDP,]],
        ),  # [[PARTIAL_BOUNDED_RDP],[FULL_BOUNDED_RDP]]) ,  # PARTIAL_BOUNDED_RDP
        (
            ['resource_master.dp_policy.is_rdp.dominant_resource_share'],
            [[DominantRdpShareType.DPS_RDP_alpha_positive_budget, ]],
        ),
        # (['task.arrival_interval'], [[i] for i in t_intvl]),
        (
            [
                'resource_master.dp_policy.denominator',
                'task.demand.epsilon.mice_percentage',
            ],
            list(
                chain(
                    [[i, 75] for i in [1, 2500, 5000, 10000, 20000, 25000]],
                    [[15000, j] for j in epsilon_mice_pct],
                )
            ),
        ),
        (
            ['task.timeout.interval'],
            [[task_timeout_global]],
        ),  # fixme use filter is more concise
    ]

    test_factors = [
        (
            [
                'sim.duration',
                'task.timeout.interval',
                'task.timeout.enabled',
                'resource_master.dp_policy.is_admission_control_enabled',
            ],
            [['%d s' % 22, 21, False, False]],
        ),  # 250000
        (
            [
                'resource_master.block.is_static',
                'resource_master.block.init_amount',
                'task.demand.num_blocks.mice_percentage',
            ],
            [[True, 1, 100], [False, 11, 75]],  # single block  # dynamic block
        ),
        (
            [
                'resource_master.dp_policy.is_rdp',
                'resource_master.dp_policy',
                'resource_master.dp_policy.denominator',
                'resource_master.block.lifetime',
            ],
            [
                [False, DpPolicyType.DP_POLICY_FCFS, None, None],
                [False, DpPolicyType.DP_POLICY_RR_N, N_dp, None],
                [False, DpPolicyType.DP_POLICY_RR_T, None, T_dp],
                [False, DpPolicyType.DP_POLICY_DPF_N, N_dp, None],
                [False, DpPolicyType.DP_POLICY_DPF_T, None, T_dp],
                [True, DpPolicyType.DP_POLICY_FCFS, None, None],
                [True, DpPolicyType.DP_POLICY_DPF_N, N_rdp, None],
                [True, DpPolicyType.DP_POLICY_DPF_T, None, T_rdp],
            ],
        ),
    ]
    
    # factors = parse_user_factors(config, args.factors)
    if factors and run_factor:
        if not is_factor_rdp:
            simulate_factors(config, factors, Top, config_filter=load_filter)
        else:
            simulate_factors(config, factors_rdp, Top, config_filter=load_filter)

    if run_test_by_factor:
        simulate_factors(config, test_factors, Top, config_filter=load_filter)

    if run_test_single:
        pp.pprint(config)
        pp.pprint(simulate(copy.deepcopy(config), Top))

    task_configs = {}
    scheduler_configs = {}
    config1 = copy.deepcopy(config)
    # use rate limit by default
    config1["resource_master.dp_policy"] = DpPolicyType.DP_POLICY_RR_T
    # use random
    config1['task.completion_time.constant'] = None
    config1['task.demand.num_cpu.constant'] = None
    config1["resource_master.is_cpu_needed_only"] = False

    demand_block_num_baseline = (
        config1['task.demand.epsilon.mean_tasks_per_block']
        * config1['task.arrival_interval']
        / config1['resource_master.block.arrival_interval']
    )
    demand_block_num_low_factor = 1
    task_configs["high_cpu_low_dp"] = {
        'task.demand.num_cpu.max': config1["resource_master.cpu_capacity"],
        'task.demand.num_cpu.min': 2,
        'task.demand.epsilon.mean_tasks_per_block': 200,
        'task.demand.num_blocks.mu': demand_block_num_baseline
        * demand_block_num_low_factor,
        # 3
        'task.demand.num_blocks.sigma': demand_block_num_baseline
        * demand_block_num_low_factor,
    }
    task_configs["low_cpu_high_dp"] = {
        'task.demand.num_cpu.max': 2,
        'task.demand.num_cpu.min': 1,
        'task.demand.epsilon.mean_tasks_per_block': 8,
        'task.demand.num_blocks.mu': demand_block_num_baseline
        * demand_block_num_low_factor
        * 4,
        # 45
        'task.demand.num_blocks.sigma': demand_block_num_baseline
        * demand_block_num_low_factor
        * 4,
        # 5
    }

    scheduler_configs["fcfs_policy"] = {'resource_master.dp_policy': DpPolicyType.DP_POLICY_FCFS}

    scheduler_configs["rate_policy_slow_release"] = {
        'resource_master.dp_policy': DpPolicyType.DP_POLICY_RR_T,
        'resource_master.block.lifetime': config1['task.arrival_interval'] * 10 * 5
        # 500
    }
    scheduler_configs["rate_policy_fast_release"] = {
        'resource_master.dp_policy': DpPolicyType.DP_POLICY_RR_T,
        'resource_master.block.lifetime': config1['task.arrival_interval'] * 5,
    }  # 50
    dp_factor_names = set()
    for sched_conf_k, sched_conf_v in scheduler_configs.items():
        for conf_factor in sched_conf_v:
            dp_factor_names.add(conf_factor)

    for task_conf_k, task_conf_v in task_configs.items():
        for conf_factor in task_conf_v:
            dp_factor_names.add(conf_factor)
    dp_factor_names = list(dp_factor_names)
    dp_factor_values = []

    configs = []

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_fcfs"
    config2["resource_master.dp_policy"] = DpPolicyType.DP_POLICY_FCFS
    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_dpf"
    config2["resource_master.dp_policy"] = DpPolicyType.DP_POLICY_DPF_N
    config2["resource_master.dp_policy.denominator"] = 30  # inter arrival is 10 sec
    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_dpft"
    config2["resource_master.dp_policy"] = DpPolicyType.DP_POLICY_DPF_T
    config2["resource_master.block.lifetime"] = 300
    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_rate_limiting"
    config2["resource_master.dp_policy"] = DpPolicyType.DP_POLICY_RR_T
    config2["resource_master.block.lifetime"] = 300
    configs.append(config2)

    config2 = copy.deepcopy(config1)
    config2["sim.workspace"] = "workspace_dpfa"
    config2["resource_master.dp_policy"] = DpPolicyType.DP_POLICY_DPF_NA
    config2["resource_master.block.lifetime"] = 300
    configs.append(config2)

    if run_test_many:
        for c in configs:
            # c['sim.seed'] = time.time()
            # pp.pprint(c)
            pp.pprint(simulate(copy.deepcopy(c), Top))

    if run_test_parallel:
        simulate_many(copy.deepcopy(configs), Top)

    for sched_conf_k, sched_conf_v in scheduler_configs.items():
        for task_conf_k, task_conf_v in task_configs.items():
            new_config = copy.deepcopy(config1)
            new_config.update(sched_conf_v)
            new_config.update(task_conf_v)
            workspace_name = "workspace_%s-%s" % (sched_conf_k, task_conf_k)
            new_config["sim.workspace"] = workspace_name
            configs.append(new_config)

    for i, c in enumerate(configs):
        try:
            # simulate(c, Top)
            pass
        except:
            print(i)
            print(c)

    # debug a config
    # for cfg in configs:
    #     # if "slow_release-low_cpu_high_dp"in cfg["sim.workspace"]:
    #     #     simulate(cfg, Top)
    #
    #     if "fast_release-low_cpu_high_dp"in cfg["sim.workspace"]:
    #         simulate(cfg, Top)
