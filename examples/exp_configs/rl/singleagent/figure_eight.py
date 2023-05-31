"""Figure eight example."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
from flow.envs import AccelEnv
from flow.networks import FigureEightNetwork

# time horizon of a single rollout
HORIZON = 1500
# number of rollouts per training iteration
N_ROLLOUTS = 10
# number of parallel workers
N_CPUS = 10

OBS_TYPE = "precise" # Options: ["precise", "image", "partial", "blank"]
EVALUTE = False
REWARD_FUNC = "accel" # Options: ["accel", "wave"]
MEMORY = False
CIRCLE_MASK = True

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = VehicleParams()
vehicles.add(
    veh_id='human',
    acceleration_controller=(IDMController, {
        'noise': 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=1.5,
    ),
    num_vehicles=13)
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        # decel=1.5,
    ),
    num_vehicles=1)

flow_params = dict(
    # name of the experiment
    exp_tag='singleagent_figure_eight',

    # name of the flow environment the experiment is running on
    env_name=AccelEnv,

    # name of the network class the experiment is running on
    network=FigureEightNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=True,
        save_render=False,
        restart_instance=False,
        sight_radius=30,
        show_radius=False,
        additional_params={
            "network": "figure_8",
            "obs_type": OBS_TYPE, 
        }
        # emission_path="./michael_files/emission_collection/"
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        warmup_steps=750,
        additional_params={
            "target_velocity": 20,
            "max_accel": 3,
            "max_decel": 3,
            "sort_vehicles": False,
            "radius_ring": [20,30],
            "obs_type": OBS_TYPE, 
            "evaluate": EVALUTE,
            "img_dim": 84,
            "reward": REWARD_FUNC,
            "memory": MEMORY,
            "circle_mask": CIRCLE_MASK
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params={
            "radius_ring": 32,
            "lanes": 1,
            "speed_limit": 30,
            "resolution": 40
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)
