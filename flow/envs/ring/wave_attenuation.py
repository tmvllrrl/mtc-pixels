"""
Environment used to train a stop-and-go dissipating controller.

This is the environment that was used in:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and
Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol.
abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465
"""

from email.errors import ObsoleteHeaderDefect
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.envs.base import Env

from space_time import plot_std

from gym.spaces.box import Box

from copy import deepcopy
import numpy as np
import random
from PIL import Image
from scipy.optimize import fsolve

import uuid
import time
import os

import cv2
import matplotlib.pyplot as plt

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration of autonomous vehicles
    'max_accel': 1,
    # maximum deceleration of autonomous vehicles
    'max_decel': 1,
    # bounds on the ranges of ring road lengths the autonomous vehicle is
    # trained on
    'ring_length': [220, 270],
}


def v_eq_max_function(v, *args):
    """Return the error between the desired and actual equivalent gap."""
    num_vehicles, length = args

    # maximum gap in the presence of one rl vehicle
    s_eq_max = (length - num_vehicles * 5) / (num_vehicles - 1)

    v0 = 30
    s0 = 2
    tau = 1
    gamma = 4

    error = s_eq_max - (s0 + v * tau) * (1 - (v / v0) ** gamma) ** -0.5

    return error


class WaveAttenuationEnv(Env):
    """Fully observable wave attenuation environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in a variable density ring road.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on. If set to None, the environment sticks to the ring
      road specified in the original network definition.

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function rewards high average speeds from all vehicles in
        the network, and penalizes accelerations by the rl vehicle.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        self.avg_velocity_collector = []
        self.min_velocity_collector = []
        self.rl_velocity_collector = []
        self.rl_accel_realized_collector = []

        self.memory = []
        self.img_dim = env_params.additional_params['img_dim']

        self.rl_action_collector = []

        super().__init__(env_params, sim_params, network, simulator)


    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ["Velocity", "Absolute_pos"]
        return Box(
            low=0,
            high=1,
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        self.rl_action_collector.append(float(rl_actions[0])) # Collecting rl_actions
        # print("rl_action", rl_actions)
        self.k.vehicle.apply_acceleration(
            self.k.vehicle.get_rl_ids(), rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.additional_params['reward'] == "wave":
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
            eta = 4  # 0.25
            mean_actions = np.mean(np.abs(np.array(rl_actions)))
            accel_threshold = 0

            if mean_actions > accel_threshold:
                reward += eta * (accel_threshold - mean_actions)

            return float(reward)

    def get_state(self):
        """See class definition."""
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.k.vehicle.get_ids()]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.k.vehicle.get_ids()]

        return np.array(speed + pos)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        The sumo instance is reset with a new ring length, and a number of
        steps are performed with the rl vehicle acting as a human vehicle.
        """
        self.avg_velocity_collector = []
        self.min_velocity_collector = []
        self.rl_velocity_collector = []
        self.rl_accel_realized_collector = []

        self.space_time_collector = []

        self.memory = []

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


class WaveAttenuationPOEnv(WaveAttenuationEnv):
    """POMDP version of WaveAttenuationEnv.

    Note that this environment only works when there is one autonomous vehicle
    on the network.

    Required from env_params:

    * max_accel: maximum acceleration of autonomous vehicles
    * max_decel: maximum deceleration of autonomous vehicles
    * ring_length: bounds on the ranges of ring road lengths the autonomous
      vehicle is trained on

    States
        The state consists of the speed and headway of the ego vehicle, as well
        as the difference in speed between the ego vehicle and its leader.
        There is no assumption on the number of vehicles in the network.

    Actions
        See parent class

    Rewards
        See parent class

    Termination
        See parent class

    """

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.additional_params['reward'] == "wave":
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
            eta = 4  # 0.25
            mean_actions = np.mean(np.abs(np.array(rl_actions)))
            accel_threshold = 0

            if mean_actions > accel_threshold:
                reward += eta * (accel_threshold - mean_actions)

            return float(reward)
        
        elif self.env_params.additional_params['reward'] == "chatgpt":
            
            if rl_actions is None:
                return 0
            
            rl_id = self.k.vehicle.get_rl_ids()[0]
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            w1 = 0.5
            w2 = 0.3
            w3 = 1.0

            avg_speed = np.mean([self.k.vehicle.get_speed(veh_id) for veh_id in self.k.vehicle.get_ids()])
            max_speed = self.net_params.additional_params['speed_limit']

            agent_acceleration = np.mean(np.abs(np.array(rl_actions)))

            mid_point = self.net_params.additional_params['length'] / 2.
            rl_x = self.k.vehicle.get_x_by_id(rl_id)
            lead_x = self.k.vehicle.get_x_by_id(lead_id)

            if rl_x > mid_point and lead_x < mid_point:
                # reset occurred
                lead_gap = abs((self.net_params.additional_params['length'] - rl_x) + lead_x)
            else:
                lead_gap = abs(lead_x - rl_x)
            
            collision_penalty = 1 if lead_gap < 12. else 0

            reward = w1 * (avg_speed/max_speed) - w2 * abs(agent_acceleration) - w3 * collision_penalty

            return float(reward)

    @property
    def observation_space(self):
        """See class definition."""
        obs_type = self.env_params.additional_params['obs_type']

        if obs_type == "precise":
            obs_shape = (3,)
        elif obs_type in ["image", "blank"]:
            obs_shape = (self.img_dim, self.img_dim, )
        elif obs_type == "only_pos":
            obs_shape = (1, )
        elif obs_type == "chatgpt":
            obs_shape = (8, )

        obs_space = Box(low=-float('inf'), 
                high=float('inf'),
                shape=obs_shape,
                dtype=np.float32)

        return obs_space


    def get_state(self):
        """See class definition."""

        
        if self.env_params.additional_params['evaluate']:
            results_name = "ring_results"
            '''
                Saves various information about the run to files to be used later on.
                Some of the files are used for graphs, others calculate some avg. 
                statistics on the data
            '''
            if self.step_counter == self.env_params.horizon + self.env_params.warmup_steps:
                
                if not os.path.exists(f"./michael_files/{results_name}/"):
                    os.mkdir(f"./michael_files/{results_name}/")
            
                with open(f"./michael_files/{results_name}/avg_velocity.txt", "a") as f:
                    np.savetxt(f, np.asarray(self.avg_velocity_collector), delimiter=",", newline=",")
                    f.write("\n")
                
                with open(f"./michael_files/{results_name}/min_velocity.txt", "a") as f:
                    np.savetxt(f, np.asarray(self.min_velocity_collector), delimiter=",", newline=",")
                    f.write("\n")
                
                with open(f"./michael_files/{results_name}/rl_velocity.txt", "a") as f:
                    np.savetxt(f, np.asarray(self.rl_velocity_collector), delimiter=",", newline=",")
                    f.write("\n")
            
                with open(f"./michael_files/{results_name}/rl_accel_realized.txt", "a") as f:
                    np.savetxt(f, np.asarray(self.rl_accel_realized_collector), delimiter=",", newline=",")
                    f.write("\n")

                self.rl_action_collector = np.asarray(self.rl_action_collector)
                np.savez(f"./michael_files/{results_name}/rl_action_collector.npz", rl_actions=self.rl_action_collector)

                self.space_time_collector = np.asarray(self.space_time_collector)
                np.savez(f"./michael_files/{results_name}/space_time_collector.npz", space_time_collector=self.space_time_collector)
                plot_std(self.space_time_collector, horizon=3000, warmup=3000, results_name=results_name)

            speed = np.asarray([self.k.vehicle.get_speed(veh_id) for veh_id in self.k.vehicle.get_ids()])
            self.avg_velocity_collector.append(np.mean(speed))
            self.min_velocity_collector.append(np.min(speed))

            rl_id = self.k.vehicle.get_rl_ids()[0]
            self.rl_velocity_collector.append(self.k.vehicle.get_speed(rl_id))
            self.rl_accel_realized_collector.append(self.k.vehicle.get_realized_accel(rl_id))

            if self.step_counter % 10 == 0 and self.step_counter != self.env_params.horizon + self.env_params.warmup_steps:
                st_state = []
                for veh_id in self.k.vehicle.get_ids():
                    pos = self.k.vehicle.get_x_by_id(veh_id)
                    vel = self.k.vehicle.get_speed(veh_id)

                    pos_vel = (pos, vel)
                    st_state.append(pos_vel)

                self.space_time_collector.append(st_state)

        obs_type = self.env_params.additional_params['obs_type']

        if obs_type == "precise":
            '''
                Following code is the original code from Cathy Wu 
            '''
            rl_id = self.k.vehicle.get_rl_ids()[0]
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            if self.env_params.additional_params['ring_length'] is not None:
                max_length = self.env_params.additional_params['ring_length'][1]
            else:
                max_length = self.k.network.length()

            observation = np.array([
                self.k.vehicle.get_speed(rl_id) / max_speed,
                (self.k.vehicle.get_speed(lead_id) -
                self.k.vehicle.get_speed(rl_id)) / max_speed,
                (self.k.vehicle.get_x_by_id(lead_id) -
                self.k.vehicle.get_x_by_id(rl_id)) % self.k.network.length()
                / max_length
            ])

        elif obs_type == "only_pos":
            '''
                Following code is the original code from Cathy Wu 
                However, modified to only look at the position of the AV
                and leading vehicle
            '''
            rl_id = self.k.vehicle.get_rl_ids()[0]
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id

            # normalizers
            max_speed = 15.
            if self.env_params.additional_params['ring_length'] is not None:
                max_length = self.env_params.additional_params['ring_length'][1]
            else:
                max_length = self.k.network.length()

            observation = np.array([
                (self.k.vehicle.get_x_by_id(lead_id) -
                 self.k.vehicle.get_x_by_id(rl_id)) % self.k.network.length()
                / max_length
            ])

        elif obs_type == "chatgpt":

            rl_id = self.k.vehicle.get_rl_ids()[0]
            lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
            follow_id = self.k.vehicle.get_follower(rl_id) or rl_id

            speed_dif = self.k.vehicle.get_speed(rl_id) - self.k.vehicle.get_speed(lead_id)
            speed_dif += self.k.vehicle.get_speed(rl_id) - self.k.vehicle.get_speed(follow_id)

            dist_dif = self.k.vehicle.get_x_by_id(rl_id) - self.k.vehicle.get_x_by_id(lead_id)
            dist_dif += self.k.vehicle.get_x_by_id(rl_id) - self.k.vehicle.get_x_by_id(follow_id)

            observation = np.array([
                self.k.vehicle.get_speed(rl_id), # Agent's speed
                self.k.vehicle.get_x_by_id(rl_id), # Agent's position
                speed_dif, # speed difference between vehicles behind and ahead
                self.k.vehicle.get_x_by_id(lead_id), # Position of lead veh
                self.k.vehicle.get_x_by_id(follow_id), # Position of follow veh
                dist_dif, # Gap distance
                self.net_params.additional_params['speed_limit'], # Speed limit
                2, # Visibility range
            ])

        elif obs_type == "image": 
            '''
                SUMO GUI Partial Observations
                Following code is another method for getting partial observations from the sumo-gui screenshots
                Uses the 2D position of the RL vehicle for a more accurate screenshot
            '''
            sight_radius = self.sim_params.sight_radius
            rl_id = self.k.vehicle.get_rl_ids()[0]
            x, y = self.k.vehicle.get_2d_position(rl_id)
            x, y = self.map_coordinates(x, y)
            observation = Image.open(f"./michael_files/sumo_obs/state_{self.k.simulation.id}.jpeg").convert("RGB")        
            left, upper, right, lower = x - sight_radius, y - sight_radius, x + sight_radius, y + sight_radius
            observation = observation.crop((left, upper, right, lower))
            observation = observation.convert("L")
            observation = observation.resize((self.img_dim,self.img_dim))
            observation = np.asarray(observation)
            observation = self.cv2_clipped_zoom(observation, 1.5)

            if self.env_params.additional_params['circle_mask']:
                height, width = observation.shape[0:2]
                sight_radius = height / 2
                mask = np.zeros((height, width), np.uint8)
                cv2.circle(mask, (int(sight_radius), int(sight_radius)),
                        int(sight_radius), (255, 255, 255), thickness=-1)
                observation = cv2.bitwise_and(observation, observation, mask=mask)

            observation = Image.fromarray(observation)
            observation.save(f'./michael_files/sumo_obs/example{self.k.simulation.id}_{self.k.simulation.timestep}.png')
            observation = np.asarray(observation)

            observation = observation / 255.

        elif obs_type == "blank":
            '''
                All white observations to make sure that learning on images is working and that the policy
                is not just randomly learning to do the correct behavior.
            '''
            observation = np.zeros((84,84)) / 255.
            observation = np.asarray(observation)

        
        # time_taken = time.time() - start_time
        # with open(f"./time_taken_SPO{self.k.simulation.id}.txt", "a") as logfile:
        #     logfile.write(f"{time_taken}\n")

        return observation

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        # rl_id = self.k.vehicle.get_rl_ids()[0]
        # lead_id = self.k.vehicle.get_leader(rl_id) or rl_id
        # self.k.vehicle.set_observed(lead_id)
    
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

    def gaussian_noise(self, img, sigma):
        noise = np.random.randn(img.shape[0], img.shape[1])
        img = img.astype('int16')
        img_noise = img + noise * sigma
        img_noise = np.clip(img_noise, 0, 255)
        img_noise = img_noise.astype('uint8')
        return img_noise

# Command to clear memory cache.
# sudo sh -c 'echo 3 >/proc/sys/vm/drop_caches'

