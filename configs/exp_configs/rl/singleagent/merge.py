"""Benchmark for merge0.

Trains a small percentage of autonomous vehicles to dissipate shockwaves caused
by merges in an open network. The autonomous penetration rate in this example
is 10%.

- **Action Dimension**: (5, )
- **Observation Dimension**: (25, )
- **Horizon**: 750 steps
"""
from flow.envs import MergePOEnv
from flow.networks import MergeNetwork
from copy import deepcopy
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.networks.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, RLController

# time horizon of a single rollout
HORIZON = 750
# HORIZON = 700
# inflow rate at the highway
FLOW_RATE = 1500
# inflow rate at the merge
MERGE_RATE = 200
# percent of autonomous vehicles
RL_PENETRATION = 0.1

# number of rollouts per training iteration
N_ROLLOUTS = 10
# number of parallel workers
N_CPUS = 10

OBS_TYPE = "image" # Options: ["precise", "image"]
EVALUATE = False
CIRCLE_MASK = True
PERTURB = False

# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = deepcopy(ADDITIONAL_NET_PARAMS)
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 1
additional_net_params["pre_merge_length"] = 350

# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    num_vehicles=5)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode=9,
    ),
    num_vehicles=0)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
    departLane="random",
    departSpeed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE,
    departLane="random",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="inflow_merge_bottom",
    vehs_per_hour=MERGE_RATE,
    departLane="free",
    departSpeed=7.5)
inflow.add(
    veh_type="human",
    edge="inflow_merge_top",
    vehs_per_hour=MERGE_RATE,
    departLane="free",
    departSpeed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag="double_merge_precise_HW1500M200",

    # name of the flow environment the experiment is running on
    env_name=MergePOEnv,

    # name of the network class the experiment is running on
    network=MergeNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=0.5,
        render=True,
        sight_radius=42,
        show_radius=False,
        additional_params={
            "network": "merge",
            "obs_type": OBS_TYPE
        }
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=1, 
        # warmup_steps=200,
        additional_params={
            "max_accel": 1.5,
            "max_decel": 1.5,
            "target_velocity": 20,
            "obs_type": OBS_TYPE,
            "evaluate": EVALUATE,
            "num_rl": 5,
            "img_dim": 84,
            "circle_mask": CIRCLE_MASK,
            "perturb": PERTURB
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)
