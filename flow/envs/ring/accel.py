"""Environment for training the acceleration behavior of vehicles in a ring."""

from flow.core import rewards
from flow.envs.base import Env

from gym.spaces.box import Box

import numpy as np
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
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]
        self.k.vehicle.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            return rewards.desired_velocity(self, fail=kwargs['fail'])

    def get_state(self):
        """See class definition."""

        # speed = np.asarray([self.k.vehicle.get_speed(veh_id) for veh_id in self.k.vehicle.get_ids()])
        # self.avg_velocity_collector.append(np.mean(speed))
        # self.min_velocity_collector.append(np.min(speed))

        # rl_id = self.k.vehicle.get_rl_ids()[0]
        # self.rl_velocity_collector.append(self.k.vehicle.get_speed(rl_id))
        # self.rl_accel_collector.append(self.k.vehicle.get_accel(rl_id))
        # self.rl_accel_realized_collector.append(self.k.vehicle.get_realized_accel(rl_id))

        # # Save the avg and min velocity collectors to a file
        # if self.step_counter == self.env_params.horizon + self.env_params.warmup_steps:
        #     print(self.step_counter)
        #     print(len(self.avg_velocity_collector))
        #     print(len(self.min_velocity_collector))

        #     with open(f"/home/michael/Desktop/flow/michael_files/avg_velocity.txt", "a") as f:
        #         np.savetxt(f, np.asarray(self.avg_velocity_collector), delimiter=",", newline="")
        #         f.write("\n")
            
        #     with open(f"/home/michael/Desktop/flow/michael_files/min_velocity.txt", "a") as f:
        #         np.savetxt(f, np.asarray(self.min_velocity_collector), delimiter=",", newline="")
        #         f.write("\n")
            
        #     with open(f"/home/michael/Desktop/flow/michael_files/rl_velocity.txt", "a") as f:
        #         np.savetxt(f, np.asarray(self.rl_velocity_collector), delimiter=",", newline="")
        #         f.write("\n")

        #     with open(f"/home/michael/Desktop/flow/michael_files/rl_accel.txt", "a") as f:
        #         np.savetxt(f, np.asarray(self.rl_accel_collector), delimiter=",", newline="")
        #         f.write("\n")
        
        #     with open(f"/home/michael/Desktop/flow/michael_files/rl_accel_realized.txt", "a") as f:
        #         np.savetxt(f, np.asarray(self.rl_accel_realized_collector), delimiter=",", newline="")
        #         f.write("\n")

        
        '''
            Following code is the original code from Cathy Wu 
        '''
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.sorted_ids]

        '''
            SUMO GUI Full Observations
            Following code uses screenshot from sumo-gui to train the model
        '''
        # observation = Image.open(f"/home/michael/Desktop/flow_screenshots/state_{self.k.simulation.id}.jpeg")
        # # observation = observation.crop((191, 0, 852, 661)) Keeping this line to save the numbers
        # observation = observation.resize((84,84)) # Resizing the image to be smaller
        # observation = np.asarray(observation) / 255.

        '''
            SUMO GUI Partial Observations
            Following code uses partial observations from screenshots from sumo-gui to train the model
            Uses numpy to find red pixels within the screenshot
        '''
        # sight_radius = self.sim_params.sight_radius
        # observation = Image.open(f"/home/michael/Desktop/flow/sumo_full_obs/state_{self.k.simulation.id}.jpeg").convert("RGB")
        # observation = np.moveaxis(np.asarray(observation), -1, 0)
        # redpix, greenpix = observation[0], observation[1] 
        # redpix_indices = np.where(np.logical_and(redpix > 180, redpix < 220, greenpix < 50))
        # y, x = int(np.mean(redpix_indices[0])), int(np.mean(redpix_indices[1]))
        # observation = Image.fromarray(np.moveaxis(observation, 0, -1))
        # left, upper, right, lower = x - sight_radius, y - sight_radius, x + sight_radius, y + sight_radius
        # observation = observation.crop((left, upper, right, lower))
        # # observation.save(f'./sumo_partial_obs/example{self.k.simulation.id}_{self.k.simulation.timestep}.png')
        # observation = observation.resize((84,84)) # Resizing the image to be smaller
        # observation = np.asarray(observation) / 255.

        '''
            SUMO GUI Partial Observations
            Following code is another method for getting partial observations from the sumo-gui screenshots
            Uses the 2D position of the RL vehicle for a more accurate screenshot
        '''
        # sight_radius = self.sim_params.sight_radius
        # rl_id = self.k.vehicle.get_rl_ids()[0]
        # x, y = self.k.vehicle.get_2d_position(rl_id)
        # x, y = self.map_coordinates(x, y)
        # observation = Image.open(f"/home/michael/Desktop/flow/sumo_obs/state_{self.k.simulation.id}.jpeg").convert("RGB")        
        # left, upper, right, lower = x - sight_radius, y - sight_radius, x + sight_radius, y + sight_radius
        # observation = observation.crop((left, upper, right, lower))
        # # observation.save(f'./sumo_obs/example{self.k.simulation.id}_{self.k.simulation.timestep}.png')
        # observation = np.asarray(observation) / 255.

        '''
            Pyglet Renderer Full Observations
            Following code uses the Pyglet renderer with frames of the full observation space
        '''
        # print(type(self.frame))
        # print(self.frame.shape)
        # observation = Image.fromarray(np.asarray(self.frame))
        # observation = observation.resize((84,84))
        # observation = np.asarray(observation) / 255.

        '''
            Pyglet Renderer Partial Observations
            Following code uses the Pyglet renderer with sights around the RL vehicles for local observation
        '''
        # if np.asarray(self.sights).shape[0] == 0: # When the rendering is initialized, the shape is (0,)
        #     observation = np.uint8(np.full((100,100,3), 100)) # Create a blank gray square image
        #     observation = Image.fromarray(observation)
        # else: 
        #     observation = Image.fromarray(np.asarray(self.sights[0]))
        # observation.save("./sight_example.png")
        # observation = observation.resize((84,84))
        # observation = np.asarray(observation) / 255.

        '''
            Matplotlib Full Observations
            Following code uses Matplotlib to render frames based on the positions of the vehicle, which
            the RL controller learns on. 
        '''
        # car_pos = [int(self.k.vehicle.get_x_by_id(item)) for item in self.k.vehicle.get_ids()]
        # observation = self.plt_frame(car_pos)


        '''
            Matplotlib Partial Observations
            Following code uses Matplotlib to render frames based on the positions of the vehicle, which
            the RL controller learns on. 
        '''
        # sight_radius = self.sim_params.sight_radius
        # car_pos = [int(self.k.vehicle.get_x_by_id(item)) for item in self.k.vehicle.get_ids()]
        # observation = self.plt_frame(car_pos)
        # red_pixel = np.array([255, 0, 0])
        # red_idx = np.where(np.all(observation == red_pixel, axis=-1))
        # y, x = np.mean(red_idx[0]), np.mean(red_idx[1])
        # x_min = int(x - sight_radius)
        # y_min = int(y - sight_radius)
        # x_max = int(x + sight_radius)
        # y_max = int(y + sight_radius)
        # observation = observation[y_min:y_max, x_min:x_max]
        # observation = observation / 255.

        '''
            All white observations to make sure that learning on images is working and that the policy
            is not just randomly learning to do the correct behavior.
        '''
        # observation = np.zeros((84,84)) / 255.


        return np.array(speed + pos)

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

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
        obs = super().reset()

        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        return obs
