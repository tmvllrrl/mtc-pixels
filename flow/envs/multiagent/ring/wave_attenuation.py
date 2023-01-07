"""
Environment used to train a stop-and-go dissipating controller.

This is the environment that was used in:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and
Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol.
abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465
"""

import numpy as np
from gym.spaces.box import Box
import random
from scipy.optimize import fsolve
from copy import deepcopy

import cv2
from PIL import Image

from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.multiagent.base import MultiEnv
from flow.envs.ring.wave_attenuation import v_eq_max_function


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}


class MultiWaveAttenuationPOEnv(MultiEnv):
    """Multiagent shared model version of WaveAttenuationPOEnv.

    Intended to work with Lord Of The Rings Network.
    Note that this environment current
    only works when there is one autonomous vehicle
    on each ring.

    Required from env_params: See parent class

    States
        See parent class

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class
    """

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-1, 
                high=1,
                shape=(84,84,), 
                # shape=(3,), 
                dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        add_params = self.net_params.additional_params
        num_rings = add_params['num_rings']
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(int(self.initial_vehicles.num_rl_vehicles / num_rings), ),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            max_length = self.env_params.additional_params['ring_length'][1]

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,
                (self.k.vehicle.get_speed(lead_id) -
                 self.k.vehicle.get_speed(rl_id))
                / max_speed,
                self.k.vehicle.get_headway(rl_id) / max_length
            ])
            obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring."""
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rew = {}
        for rl_id in rl_actions.keys():
            edge_id = rl_id.split('_')[1]
            edges = self.gen_edges(edge_id)
            vehs_on_edge = self.k.vehicle.get_ids_by_edge(edges)
            vel = np.array([
                self.k.vehicle.get_speed(veh_id)
                for veh_id in vehs_on_edge
            ])
            if any(vel < -100) or kwargs['fail']:
                return 0.

            target_vel = self.env_params.additional_params['target_velocity']
            max_cost = np.array([target_vel] * len(vehs_on_edge))
            max_cost = np.linalg.norm(max_cost)

            cost = vel - target_vel
            cost = np.linalg.norm(cost)

            rew[rl_id] = max(max_cost - cost, 0) / max_cost
        return rew

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)

    @staticmethod
    def gen_edges(i):
        """Return the edges corresponding to the rl id."""
        return ['top_{}'.format(i), 'left_{}'.format(i),
                'right_{}'.format(i), 'bottom_{}'.format(i)]


class MultiAgentWaveAttenuationPOEnv(MultiEnv):
    """Multi-agent variant of WaveAttenuationPOEnv.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on. If set to None, the environment sticks to the ring
      road specified in the original network definition.

    States
        The state of each agent (AV) consists of the speed and headway of the
        ego vehicle, as well as the difference in speed between the ego vehicle
        and its leader. There is no assumption on the number of vehicles in the
        network.

    Actions
        Actions are an acceleration for each rl vehicle, bounded by the maximum
        accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function rewards high average speeds from all vehicles in
        the network, and penalizes accelerations by the rl vehicle. This reward
        is shared by all agents.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=-5, 
                high=5, 
                # shape=(84,84,),
                shape=(3,), 
                dtype=np.float32)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),
            dtype=np.float32)

    def get_state(self):
        """See class definition."""
        '''
            Original Wu methodology
        '''
        obs = {}
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            max_length = self.env_params.additional_params['ring_length'][1]

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,
                (self.k.vehicle.get_speed(lead_id) -
                 self.k.vehicle.get_speed(rl_id))
                / max_speed,
                self.k.vehicle.get_headway(rl_id) / max_length
            ])
            obs.update({rl_id: observation})

        '''
            Image-based method
        '''
        # obs = {}
        # for rl_id in self.k.vehicle.get_rl_ids():
        #     sight_radius = self.sim_params.sight_radius
        #     rl_id = self.k.vehicle.get_rl_ids()[0]
        #     x, y = self.k.vehicle.get_2d_position(rl_id)
        #     x, y = self.map_coordinates(x, y)
        #     observation = Image.open(f"../../michael_files/sumo_obs/state_{self.k.simulation.id}.jpeg").convert("RGB")        
        #     left, upper, right, lower = x - sight_radius, y - sight_radius, x + sight_radius, y + sight_radius
        #     observation = observation.crop((left, upper, right, lower))
        #     observation = observation.convert("L")
        #     observation = observation.resize((84,84))
        #     # observation.save(f'./michael_files/sumo_obs/example{self.k.simulation.id}_{self.k.simulation.timestep}.png')
        #     observation = np.asarray(observation)
        #     observation = self.cv2_clipped_zoom(observation, 1.5)
        #     height, width = observation.shape[0:2]
        #     sight_radius = height / 2
        #     mask = np.zeros((height, width), np.uint8)
        #     cv2.circle(mask, (int(sight_radius), int(sight_radius)),
        #             int(sight_radius), (255, 255, 255), thickness=-1)
        #     observation = cv2.bitwise_and(observation, observation, mask=mask)
        #     observation = observation / 255.

        #     obs.update({rl_id: observation})

        return obs

    def _apply_rl_actions(self, rl_actions):
        """Split the accelerations by ring."""
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return 0

        vel = np.array([
            self.k.vehicle.get_speed(veh_id)
            for veh_id in self.k.vehicle.get_ids()
        ])

        if any(vel < -100) or kwargs['fail']:
            return 0.

        # reward average velocity
        eta_2 = 4.
        reward = eta_2 * np.mean(vel) / 20

        # punish accelerations (should lead to reduced stop-and-go waves)
        eta = 3  # 0.25
        mean_actions = np.mean(np.abs(list(rl_actions.values())))
        accel_threshold = 0

        if mean_actions > accel_threshold:
            reward += eta * (accel_threshold - mean_actions)

        return {key: reward for key in self.k.vehicle.get_rl_ids()}

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            self.k.vehicle.set_observed(lead_id)

    def reset(self, new_inflow_rate=None):
        """See parent class.

        The sumo instance is reset with a new ring length, and a number of
        steps are performed with the rl vehicle acting as a human vehicle.
        """
        # skip if ring length is None
        if self.env_params.additional_params['ring_length'] is None:
            return super().reset()

        # reset the step counter
        self.step_counter = 0

        # update the network
        initial_config = InitialConfig(bunching=50, min_gap=0)
        length = random.randint(
            self.env_params.additional_params['ring_length'][0],
            self.env_params.additional_params['ring_length'][1])
        additional_net_params = {
            'length':
                length,
            'lanes':
                self.net_params.additional_params['lanes'],
            'speed_limit':
                self.net_params.additional_params['speed_limit'],
            'resolution':
                self.net_params.additional_params['resolution']
        }
        net_params = NetParams(additional_params=additional_net_params)

        self.network = self.network.__class__(
            self.network.orig_name, self.network.vehicles,
            net_params, initial_config)
        self.k.vehicle = deepcopy(self.initial_vehicles)
        self.k.vehicle.kernel_api = self.k.kernel_api
        self.k.vehicle.master_kernel = self.k

        # solve for the velocity upper bound of the ring
        v_guess = 4
        v_eq_max = fsolve(v_eq_max_function, np.array(v_guess),
                          args=(len(self.initial_ids), length))[0]

        print('\n-----------------------')
        print('ring length:', net_params.additional_params['length'])
        print('v_max:', v_eq_max)
        print('-----------------------')

        # restart the sumo instance
        self.restart_simulation(
            sim_params=self.sim_params,
            render=self.sim_params.render)

        # perform the generic reset function
        return super().reset()

    def map_coordinates(self, x, y):
        offset, boundary_width = self.k.simulation.offset, self.k.simulation.boundary_width
        half_width = boundary_width / 2

        x, y = x - offset, y - offset
        x, y = x + half_width, y + half_width
        x, y = x / boundary_width, y / boundary_width
        x, y = x * 300, 300 - (y * 300)

        return x, y
    
    def cv2_clipped_zoom(self, img, zoom_factor=0):

        """
        Center zoom in/out of the given image and returning an enlarged/shrinked view of 
        the image without changing dimensions
        ------
        Args:
            img : ndarray
                Image array
            zoom_factor : float
                amount of zoom as a ratio [0 to Inf). Default 0.
        ------
        Returns:
            result: ndarray
            numpy ndarray of the same shape of the input img zoomed by the specified factor.          
        """
        if zoom_factor == 0:
            return img


        height, width = img.shape[:2] # It's also the final desired shape
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        
        ### Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
        y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1,x1,y2,x2])
        # Map back to original image coordinates
        bbox = (bbox / zoom_factor).astype(np.int)
        y1, x1, y2, x2 = bbox
        cropped_img = img[y1:y2, x1:x2]
        
        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(new_width, width)
        pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
        
        result = cv2.resize(cropped_img, (resize_width, resize_height))
        result = np.pad(result, pad_spec, mode='constant')
        assert result.shape[0] == height and result.shape[1] == width
        return result