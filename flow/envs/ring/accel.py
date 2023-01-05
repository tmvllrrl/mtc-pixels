"""Environment for training the acceleration behavior of vehicles in a ring."""

from flow.core import rewards
from flow.envs.base import Env

from gym.spaces.box import Box

import numpy as np
import cv2
import os
from PIL import Image

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    'max_accel': 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    'max_decel': 3,
    # desired velocity for all vehicles in the network, in m/s
    'target_velocity': 10,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False
}


class AccelEnv(Env):
    """Fully observed acceleration environment.

    This environment used to train autonomous vehicles to improve traffic flows
    when acceleration actions are permitted by the rl agent.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * sort_vehicles: specifies whether vehicles are to be sorted by position
      during a simulation step. If set to True, the environment parameter
      self.sorted_ids will return a list of all vehicles sorted in accordance
      with the environment

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function is the two-norm of the distance of the speed of the
        vehicles in the network from the "target_velocity" term. For a
        description of the reward, see: flow.core.rewards.desired_speed

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.

    Attributes
    ----------
    prev_pos : dict
        dictionary keeping track of each veh_id's previous position
    absolute_position : dict
        dictionary keeping track of each veh_id's absolute position
    obs_var_labels : list of str
        referenced in the visualizer. Tells the visualizer which
        metrics to track
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()

        self.avg_velocity_collector = []
        self.min_velocity_collector = []
        self.rl_velocity_collector = []
        self.rl_accel_collector = []
        self.rl_accel_realized_collector = []

        self.memory = []
        self.memory.append(np.zeros((84,84)))
        self.memory.append(np.zeros((84,84)))
        self.memory.append(np.zeros((84,84)))

        self.results_dir_name = "trial_results"

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
        return Box(
            low=0,
            high=1,
            shape=(84,84, ),
            # shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        # print(rl_actions)
        """See class definition."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]
        self.k.vehicle.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""

        '''
            Original reward function within Accel class
        '''
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            return rewards.desired_velocity(self, fail=kwargs['fail'])

    def get_state(self):
        """See class definition."""
        
        # Save the avg and min velocity collectors to a file
        # if self.step_counter == self.env_params.horizon + self.env_params.warmup_steps:
             
        #     if not os.path.exists(f"../../michael_files/{self.results_dir_name}/"):
        #        os.mkdir(f"./michael_files/{self.results_dir_name}/")
         
        #     with open(f"../../michael_files/{self.results_dir_name}/avg_velocity.txt", "a") as f:
        #         np.savetxt(f, np.asarray(self.avg_velocity_collector), delimiter=",", newline=",")
        #         f.write("\n")
            
        #     with open(f"../../michael_files/{self.results_dir_name}/min_velocity.txt", "a") as f:
        #         np.savetxt(f, np.asarray(self.min_velocity_collector), delimiter=",", newline=",")
        #         f.write("\n")
            
        #     with open(f"../../michael_files/{self.results_dir_name}/rl_velocity.txt", "a") as f:
        #         np.savetxt(f, np.asarray(self.rl_velocity_collector), delimiter=",", newline=",")
        #         f.write("\n")
        
        #     with open(f"../../michael_files/{self.results_dir_name}/rl_accel_realized.txt", "a") as f:
        #         np.savetxt(f, np.asarray(self.rl_accel_realized_collector), delimiter=",", newline=",")
        #         f.write("\n")

        # speed = np.asarray([self.k.vehicle.get_speed(veh_id) for veh_id in self.k.vehicle.get_ids()])
        # self.avg_velocity_collector.append(np.mean(speed))
        # self.min_velocity_collector.append(np.min(speed))

        # rl_id = self.k.vehicle.get_rl_ids()[0]
        # self.rl_velocity_collector.append(self.k.vehicle.get_speed(rl_id))
        # self.rl_accel_realized_collector.append(self.k.vehicle.get_realized_accel(rl_id))


        '''
            Following code is the original code from Cathy Wu 
        '''
        # speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
        #          for veh_id in self.sorted_ids]
        # pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
        #        for veh_id in self.sorted_ids]

        # observation = np.array(speed + pos)
        

        '''
            SUMO GUI Full Observations
            Following code uses screenshot from sumo-gui to train the model
        '''
        # observation = Image.open(f"./sumo_obs/state_{self.k.simulation.id}.jpeg")
        # observation = observation.convert("L")
        # observation = observation.resize((84,84)) # Resizing the image to be smaller
        # observation = np.asarray(observation) / 255.

        '''
            SUMO GUI Partial Observations
            Following code is another method for getting partial observations from the sumo-gui screenshots
            Uses the 2D position of the RL vehicle for a more accurate screenshot
        '''
        sight_radius = self.sim_params.sight_radius
        rl_id = self.k.vehicle.get_rl_ids()[0]
        x, y = self.k.vehicle.get_2d_position(rl_id)
        x, y = self.map_coordinates(x, y)
        observation = Image.open(f"../../michael_files/sumo_obs/state_{self.k.simulation.id}.jpeg").convert("RGB")        
        left, upper, right, lower = x - sight_radius, y - sight_radius, x + sight_radius, y + sight_radius
        observation = observation.crop((left, upper, right, lower))
        observation = observation.convert("L") # Grayscale the image
        observation = observation.resize((84,84)) # Resize to fit the convolution layers
        observation = np.asarray(observation)
        observation = self.cv2_clipped_zoom(observation, 1.5) # Zoom in on the image
        height, width = observation.shape[0:2]
        sight_radius = height / 2
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (int(sight_radius), int(sight_radius)),
                   int(sight_radius), (255, 255, 255), thickness=-1)
        observation = cv2.bitwise_and(observation, observation, mask=mask)
        # observation = Image.fromarray(observation)
        # observation.save(f'../../michael_files/sumo_obs/example{self.k.simulation.id}_{self.k.simulation.timestep}.png')
        # observation = np.asarray(observation)
        observation = observation / 255.

        '''
            All white observations to make sure that learning on images is working and that the policy
            is not just randomly learning to do the correct behavior.
        '''
        # observation = np.zeros((84,84)) / 255.

        # if self.step_counter == 0:
        #     for i in range(2):
        #         self.memory.append(observation)
        # else:
        # self.memory.pop(0)
        # self.memory.insert(len(self.memory), observation)

        # return np.moveaxis(np.asarray(self.memory),0,-1)

        return observation

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        # if self.k.vehicle.num_rl_vehicles > 0:
        #     for veh_id in self.k.vehicle.get_human_ids():
        #         self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            return sorted(self.k.vehicle.get_ids(), key=self._get_abs_position)
        else:
            return self.k.vehicle.get_ids()

    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        return self.absolute_position.get(veh_id, -1001)

    def reset(self):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        self.avg_velocity_collector = []
        self.min_velocity_collector = []
        self.rl_velocity_collector = []
        self.rl_accel_collector = []
        self.rl_accel_realized_collector = []

        self.memory = []
        self.memory.append(np.zeros((84,84)))
        self.memory.append(np.zeros((84,84)))
        self.memory.append(np.zeros((84,84)))

        obs = super().reset()

        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        return obs

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