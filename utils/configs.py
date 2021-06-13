from dataclasses import dataclass
from enum import Enum
# import sys
# import datetime
ENABLE_OSDI21_ARTIFACT_ONLY = True
DISABLED_FLAG = "DISABLED_FEATURE"

class DpPolicyType(str,Enum):
    DP_POLICY_FCFS = "FCFS"  # fixme "fcfs2342"
    DP_POLICY_RR_T = "RR_T" # unlock/release quota evenly until lifetime in round robin fashion
    DP_POLICY_RR_N = "RR_N"  # rrt variant with unlock/release happens on task arrival
    DP_POLICY_DPF_N = "DPF_N"  # fixme "DPF_N_234" in  # task arrival-based quota accumulation
    DP_POLICY_DPF_T = "DPF_T"  # fixme  "DPF_T_234"  task time-based quota accumulation
    # for disable following feature for OSDI artifact release
    if not ENABLE_OSDI21_ARTIFACT_ONLY:
        DP_POLICY_DPF_NA = "DPF_N_A"  # fixme  "DPF_N_A_234"adaptive task arrival based quota accumulation
        DP_POLICY_RR_N2 = "RR_N2" # unlock dp quota to the top N smallest tasks upon block end of life
    else:
        DP_POLICY_DPF_NA = DISABLED_FLAG # fixme  "DPF_N_A_234"adaptive task arrival based quota accumulation
        DP_POLICY_RR_N2 = DISABLED_FLAG# unlock dp quota to the top N smallest tasks upon block end of life

# not included in OSDI paper therefore disable that feature
# disabled_DpPolicyType = [DpPolicyType.DP_POLICY_RR_N2, DpPolicyType.DP_POLICY_DPF_NA]

class DominantDpShareType(str,Enum):
    DPS_DP_L_Inf = 'L_inf_norm_def_DRS'
    # for disable following feature for OSDI artifact release
    if not ENABLE_OSDI21_ARTIFACT_ONLY:
        DPS_DP_DEFAULT = 'default_def_DRS'
        DPS_DP_L2 = 'L2_norm_def_DRS'
        DPS_DP_L1 = 'L1_norm_def_DRS'
    else:
        DPS_DP_DEFAULT = DISABLED_FLAG
        DPS_DP_L2 = DISABLED_FLAG
        DPS_DP_L1 = DISABLED_FLAG

# not included in OSDI paper therefore disable that feature
# disabled_DominantDpShareType = [DominantDpShareType.DPS_DP_DEFAULT, DominantDpShareType.DPS_DP_L2,DominantDpShareType.DPS_DP_L1]

class DominantRdpShareType(str, Enum):
    DPS_RDP_alpha_positive_budget = 'DPS_RDP_alpha_positive_budget'  # max over all alphas
    # for disable following feature for OSDI artifact release
    if not ENABLE_OSDI21_ARTIFACT_ONLY:
        DPS_RDP_alpha_positive_balance = 'DPS_RDP_alpha_positive_balance'  # max over all positive alphas
        DPS_RDP_dominant_deduction = 'DPS_RDP_dominant_deduct'  # max of consumed alphas this ordering should fix some issues of dpf-rdp
    else:
        DPS_RDP_alpha_positive_balance = DISABLED_FLAG
        DPS_RDP_dominant_deduction = DISABLED_FLAG

# not included in OSDI paper therefore disable that feature
# disabled_DominantRdpShareType = [DominantRdpShareType.DPS_RDP_alpha_positive_balance, DominantRdpShareType.DPS_RDP_dominant_deduction]

class RdpMechanism(str, Enum):
    GAUSSIAN = 'gaussian_mechanism'  # gaussian noise
    # for disable following feature for OSDI artifact release
    if not ENABLE_OSDI21_ARTIFACT_ONLY:
        LAPLACIAN = 'laplace_mechanism'  # laplacian noise
        GAUSSIAN_AND_LAPLACIAN = 'gaussian_laplace_mechanism'  # half and half
    else:
        GAUSSIAN_AND_LAPLACIAN = DISABLED_FLAG
        LAPLACIAN = DISABLED_FLAG

# not included in OSDI paper therefore disable that feature
# disabled_RdpMechanism = [RdpMechanism.LAPLACIAN, RdpMechanism.GAUSSIAN_AND_LAPLACIAN]

class RdpBoundMode(str, Enum):
    PARTIAL_BOUNDED_RDP = 'rdp_partial_bounded'
    if not ENABLE_OSDI21_ARTIFACT_ONLY:
        FULL_BOUNDED_RDP = 'rdp_full_bounded'
    else:
        FULL_BOUNDED_RDP = None

# not included in OSDI paper therefore disable that feature
# disabled_RdpBoundMode_list = [RdpBoundMode.FULL_BOUNDED_RDP]

class DpHandlerMessageType(str, Enum):
    ALLOCATION_SUCCESS = "V"
    ALLOCATION_FAIL = "F"
    ALLOCATION_REQUEST = "allocation_request"
    NEW_TASK = "new_task_created"
    DP_HANDLER_INTERRUPT_MSG = "interrupted_by_dp_hanlder"


class ResourceHandlerMessageType(str, Enum):
    RESRC_HANDLER_INTERRUPT_MSG = 'interrupted_by_resource_hanlder'
    RESRC_RELEASE = "released_resource"
    RESRC_PERMITED_FAIL_TO_ALLOC = "RESRC_PERMITED_FAIL_TO_ALLOC"
    RESRC_TASK_ARRIVAL = "RESRC_SCHED_TASK_ARRIVAL"

# alpha subsamples for budget curve of Renyi DP
ALPHAS = [1.000001, 1.0001, 1.5, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 20480]
TIMEOUT_VAL = "timeout_triggered"
DELTA = 1.0e-9

# @dataclass
# class TaskConfig:
#     num_blocks_mice_percentage: float = 75.0
#     epsilon_mice_percentage: float = 75.0
#     num_blocks_mice: int = 1
#     num_blocks_elephant: int = 10
#     num_blocks_mice_percentage: float = 75.0
#     epsilon_mean_tasks_per_block: float = 15
#     epsilon_mice: float = 1e-2  # N < 100
#     epsilon_elephant: float = 1e-1
#     epsilon_mice_percentage: float = 75.0
#     task_completion_time_constant: float = 0  # finish immediately
#     # max : half of capacity
#     task_completion_time_max: float = 100
#     task_completion_time_min: float = 10
#     num_cpu_constant: int = 1  # int [min max]
#     num_cpu_max: int = 80
#     num_cpu_min: int = 1
#     size_memory_max: float = 412
#     size_memory_min: float = 1
#     num_gpu_max: int = 3
#     num_gpu_min: int = 1
#     @classmethod
#     def from_config_dict(cls,config_dict:dict):
#         TaskConfig(
#             num_blocks_mice_percentage=config_dict['task.demand.num_blocks.mice_percentage'],
#             epsilon_mice_percentage=config_dict['task.demand.num_blocks.mice_percentage'],
#             num_blocks_mice=config_dict['task.demand.num_blocks.mice_percentage'],
#             num_blocks_elephant=config_dict['task.demand.num_blocks.mice_percentage'],
#             num_blocks_mice_percentage=config_dict['task.demand.num_blocks.mice_percentage'],
#             epsilon_mean_tasks_per_block=config_dict['task.demand.num_blocks.mice_percentage'],
#             epsilon_mice=config_dict[],
#             epsilon_elephant=config_dict[],
#             epsilon_mice_percentage=config_dict[],
#             task_completion_time_constant=config_dict[],
#             # max : half of capacity
#             task_completion_time_max=config_dict[],
#             task_completion_time_min=config_dict[],
#             num_cpu_constant=config_dict[],
#             num_cpu_max=config_dict[],
#             num_cpu_min=config_dict[],
#             size_memory_max=config_dict[],
#             size_memory_min=config_dict[],
#             num_gpu_max=config_dict[],
#             num_gpu_min=config_dict[],
#         )
#         cfg =
#         return
#
# @dataclass
# class PolicyConfig:
#     dp_policy: DpPolicyType
#     denominator: int
#     is_rdp: bool
#
#     # True: reject early before waiting
#     is_admission_control_enabled: bool = True
#
#     is_rdp_mechanism: RdpMechanism = RdpMechanism.GAUSSIAN
#     is_rdp_rdp_bound_mode: RdpBoundMode = RdpBoundMode.PARTIAL_BOUNDED_RDP
#     is_rdp_dominant_resource_share: DominantRdpShareType = DominantRdpShareType.DPS_RDP_alpha_all
#     dpf_family_dominant_resource_share: DominantDpShareType = DominantDpShareType.DPS_DP_L_Inf  # DRS_L2
#     # True: only grant top smallest tasks.
#     # False: do best effort allocation i.e. grant from top smallest as many as possible
#
#     dpf_family_grant_top_small: bool = False
#
#
# @dataclass
# class ResourceConfig:
#     block_lifetime: int   # policy level param
#     block_is_static: bool = False
#     block_init_amount: int = 11  # for block elephant demand
#     is_cpu_needed_only: bool = True
#     cpu_capacity: int = sys.maxsize  # number of cores
#     memory_capacity: int = 624  # in GB assume granularity is 1GB
#     gpu_capacity: int = 8  # in cards
#     clock_tick_seconds: int = 25  # if ticker is not adaptive
#     clock_dpf_adaptive_tick: bool = True
#
#     block_init_epsilon: int = 1.0  # normalized
#     block_init_delta: float = 1.0e-6  # only used for rdp should be small < 1
#
#     block_arrival_interval: int = 10


# @dataclass
# class SimConfig:
#     # runtime_timeout: int = 60
#
#
#     duration: str  # '%d s' % 300
#     workspace: str # "exp_result/workspace_%s" % datetime.now().strftime("%m-%d-%HH-%M-%S")
#     db_enable: bool = True
#     db_persist: bool = True
#     dot_enable: bool = False
#
#     # in min, abort simulation after timeout
#
#     gtkw_file: str = 'sim.gtkw'
#     gtkw_live: bool = False
#     enable_log: bool = True
#     log_level: str = "DEBUG"
#     progress_enable: bool = True
#     result_file: str  = 'result.json'
#     seed: int = 23338
#     timescale: str = 's'
#     vcd_dump_file: str = 'sim_dp.vcd'
#     vcd_enable: bool = True
#     vcd_persist: bool = True
#     enable_workspace_overwrite: bool = True
#     num_delta: float = 1e-8
#     instant_timeout: float = 0.01
#

# @dataclass
# class RootConfig:
#     task_config: TaskConfig
#     resource_config: ResourceConfig
#     sim_config: SimConfig
