"""
Environments for training vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.envs.base import Env
from flow.core import rewards

from gym.spaces.box import Box

from PIL import Image
import numpy as np
import os
import cv2
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # maximum number of controllable vehicles in the network
    "num_rl": 5,
}


class MergePOEnv(Env):
    """Partially observable merge environment.

    This environment is used to train autonomous vehicles to attenuate the
    formation and propagation of waves in an open merge network.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * num_rl: maximum number of controllable vehicles in the network

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, as
        well as the ego speed of the autonomous vehicles.

        In order to maintain a fixed observation size, when the number of AVs
        in the network is less than "num_rl", the extra entries are filled in
        with zeros. Conversely, if the number of autonomous vehicles is greater
        than "num_rl", the observations from the additional vehicles are not
        included in the state space.

    Actions
        The action space consists of a vector of bounded accelerations for each
        autonomous vehicle $i$. In order to ensure safety, these actions are
        bounded by failsafes provided by the simulator at every time step.

        In order to account for variability in the number of autonomous
        vehicles, if n_AV < "num_rl" the additional actions provided by the
        agent are not assigned to any vehicle. Moreover, if n_AV > "num_rl",
        the additional vehicles are not provided with actions from the learning
        agent, and instead act as human-driven vehicles as well.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity, while slightly penalizing small time headways
        among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # maximum number of controlled vehicles
        self.num_rl = env_params.additional_params['num_rl']
        
        # image dimensions, only 1 value b/c the image is square
        self.img_dim = env_params.additional_params['img_dim']

        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()

        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []

        self.avg_velocity_collector = []
        self.min_velocity_collector = []    
        self.rl_velocity_collector = []
        self.rl_accel_realized_collector = []

        self.results_dir_name = "trial_results"

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.num_rl, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""

        obs_type = self.env_params.additional_params['obs_type']

        if obs_type == "precise":
            obs_shape = (5 * self.num_rl, )
        elif obs_type == "image":
            obs_shape = (self.img_dim, self.img_dim, self.num_rl, )

        obs_space = Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=obs_shape,
            dtype=np.float32)

        return obs_space

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.k.vehicle.get_rl_ids():
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""

        if self.env_params.additional_params['evaluate']:
            if self.step_counter == self.env_params.horizon + self.env_params.warmup_steps:
                
                if not os.path.exists(f"../../michael_files/{self.results_dir_name}/"):
                    os.mkdir(f"./michael_files/{self.results_dir_name}/")
            
                with open(f"../../michael_files/{self.results_dir_name}/avg_velocity.txt", "a") as f:
                    np.savetxt(f, np.asarray(self.avg_velocity_collector), delimiter=",", newline=",")
                    f.write("\n")
                
                with open(f"../../michael_files/{self.results_dir_name}/min_velocity.txt", "a") as f:
                    np.savetxt(f, np.asarray(self.min_velocity_collector), delimiter=",", newline=",")
                    f.write("\n")
                
                with open(f"../../michael_files/{self.results_dir_name}/rl_velocity.txt", "a") as f:
                    np.savetxt(f, np.asarray(self.rl_velocity_collector), delimiter=",", newline=",")
                    f.write("\n")
            
                with open(f"../../michael_files/{self.results_dir_name}/rl_accel_realized.txt", "a") as f:
                    np.savetxt(f, np.asarray(self.rl_accel_realized_collector), delimiter=",", newline=",")
                    f.write("\n")

            speed = np.asarray([self.k.vehicle.get_speed(veh_id) for veh_id in self.k.vehicle.get_ids()])
            self.avg_velocity_collector.append(np.mean(speed))
            self.min_velocity_collector.append(np.min(speed))

            # Only looking at one RL AV for Merge networks
            if len(self.rl_veh) != 0: rl_id = self.rl_veh[0]
            self.rl_velocity_collector.append(self.k.vehicle.get_speed(rl_id))
            self.rl_accel_realized_collector.append(self.k.vehicle.get_realized_accel(rl_id))

        obs_type = self.env_params.additional_params['obs_type']

        if obs_type == "precise":
            '''
                Original FLOW method for training with Merge
            '''
            self.leader = []
            self.follower = []

            # normalizing constants
            max_speed = self.k.network.max_speed()
            max_length = self.k.network.length()

            observation = [0 for _ in range(5 * self.num_rl)]
            for i, rl_id in enumerate(self.rl_veh):
                this_speed = self.k.vehicle.get_speed(rl_id)
                lead_id = self.k.vehicle.get_leader(rl_id)
                follower = self.k.vehicle.get_follower(rl_id)

                if lead_id in ["", None]:
                    # in case leader is not visible
                    lead_speed = max_speed
                    lead_head = max_length
                else:
                    self.leader.append(lead_id)
                    lead_speed = self.k.vehicle.get_speed(lead_id)
                    lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                        - self.k.vehicle.get_x_by_id(rl_id) \
                        - self.k.vehicle.get_length(rl_id)

                if follower in ["", None]:
                    # in case follower is not visible
                    follow_speed = 0
                    follow_head = max_length
                else:
                    self.follower.append(follower)
                    follow_speed = self.k.vehicle.get_speed(follower)
                    follow_head = self.k.vehicle.get_headway(follower)

                observation[5 * i + 0] = this_speed / max_speed
                observation[5 * i + 1] = (lead_speed - this_speed) / max_speed
                observation[5 * i + 2] = lead_head / max_length
                observation[5 * i + 3] = (this_speed - follow_speed) / max_speed
                observation[5 * i + 4] = follow_head / max_length

        elif obs_type == "image":  
            '''
                Image based method for training with Merge
            '''
            observation = np.zeros((self.num_rl,self.img_dim,self.img_dim))
            
            for i, rl_id in enumerate(self.rl_veh):
                sight_radius = self.sim_params.sight_radius

                if self.k.vehicle.get_2d_position(rl_id) != -1001:
                    x, y = self.k.vehicle.get_2d_position(rl_id)
                else:
                    continue
                x, y = self.map_coordinates(x, y)
                
                bev = Image.open(f"../../michael_files/sumo_obs/state_{self.k.simulation.id}.jpeg").convert("RGB")        
                left, upper, right, lower = x - sight_radius, y - sight_radius, x + sight_radius, y + sight_radius
                bev = bev.crop((left, upper, right, lower))
                bev = bev.convert("L").resize((self.img_dim,self.img_dim))
                bev = np.asarray(bev)
                bev = self.cv2_clipped_zoom(bev, 1.5)

                if self.env_params.additional_params['circle_mask']:
                    height, width = bev.shape[0:2]
                    sight_radius = height / 2
                    mask = np.zeros((height, width), np.uint8)
                    cv2.circle(mask, (int(sight_radius), int(sight_radius)),
                            int(sight_radius), (255, 255, 255), thickness=-1)
                    bev = cv2.bitwise_and(bev, bev, mask=mask)

                # bev = Image.fromarray(bev)
                # bev.save(f'../../michael_files/sumo_obs/example{self.k.simulation.id}_{self.k.simulation.timestep}_{i}.png')
                # bev = bev.resize((42,42))
                # bev = np.asarray(bev)
                
                bev = bev / 255.

                observation[i] = bev

            observation = np.moveaxis(observation, 0, -1)

        return observation

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # return a reward of 0 if a collision occurred
            if kwargs["fail"]:
                return 0

            # reward high system-level velocities
            cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            for rl_id in self.rl_veh:
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 1.00, 0.10

            return max(eta1 * cost1 + eta2 * cost2, 0)

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        # for veh_id in self.leader + self.follower:
        #     self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []

        self.avg_velocity_collector = []
        self.min_velocity_collector = []    
        self.rl_velocity_collector = []
        self.rl_accel_realized_collector = []

        return super().reset()

    def map_coordinates(self, x, y):
        offset, boundary_width = self.k.simulation.offset, self.k.simulation.boundary_width
        half_width = boundary_width / 2

        x = (((x - offset) + half_width) / boundary_width) * 1600

        return x, 207
    
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