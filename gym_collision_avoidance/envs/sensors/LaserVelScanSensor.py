import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs.config import ProDMPConfig
import matplotlib.pyplot as plt

import time

class LaserVelScanSensor(Sensor):
    """ 2D LaserScan based on map of the environment (containing static objects and other agents)

    :param num_beams: (int) how many beams/rays should be in the laserscan
    :param num_to_store: (int) how many past laserscans to stack into one measurement
    :param range_resolution: (float) radians between each beam
    :param max_range: (float) largest value per beam (meters)
    :param min_range: (float) smallest value per beam (meters)
    :param min_angle: (float) relative to agent's current heading, angle of the first beam (radians)
    :param max_angle: (float) relative to agent's current heading, angle of the last beam (radians)
    :param angles: (np array) linearly spaced array of angles, ranging from min_angle to max_angle, containing num_beams
    :param ranges: (np array) linearly spaced array of ranges, ranging from min_range to max_range, spaced by range_resolution

    """
    def __init__(self):
        self.config = ProDMPConfig()
        Sensor.__init__(self)
        self.name = 'laservelscan'
        self.num_beams = self.config.LASERSCAN_LENGTH
        self.num_to_store = self.config.LASERSCAN_NUM_PAST
        self.range_resolution = 2 * np.pi / self.num_beams
        self.max_range = 10 # meters
        self.min_range = 0 # meters
        self.min_angle = 0
        self.max_angle = 2 * np.pi - self.range_resolution

        self.angles = np.linspace(self.min_angle, self.max_angle, self.num_beams)
        self.ranges = np.arange(self.min_range, self.max_range, self.range_resolution)

        self.debug = False

        self.measurement_history = np.zeros((self.num_to_store, self.num_beams))
        self.num_measurements_made = 0

        self.ray_cos = np.cos(self.angles)
        self.ray_sin = np.sin(self.angles)

        if self.debug:
            plt.figure('lidar')

    def sense(self, agents, agent_index, top_down_map=None):
        """
        Args:
            agents (list): all agents in the environment
            agent_index (int): index of this agent (the one with this sensor)
            top_down_map (2D np array): unneessary parameter

        Returns:
            measurement_history (np array): (:code:`num_to_store` x :code:`num_beams`) stacked history of laserscans, where each entry is a range in meters of the nearest obstacle at that angle

        """
        agent = agents[agent_index]
        agent_pos = agent.pos_global_frame
        other_agents = agents[:agent_index] + agents[agent_index + 1:]
        crowd_poss = np.array([a.pos_global_frame for a in other_agents])
        crowd_vels = np.array([a.vel_global_frame for a in other_agents])


        x_crowd_rel, y_crowd_rel = crowd_poss[:, 0] - agent_pos[0], \
            crowd_poss[:, 1] - agent_pos[1]
        orthog_dist = np.abs(
            np.outer(x_crowd_rel, self.ray_sin) - np.outer(y_crowd_rel, self.ray_cos)
        )
        intersections_mask = orthog_dist <= other_agents[0].radius
        along_dist = np.outer(x_crowd_rel, self.ray_cos) +\
            np.outer(y_crowd_rel, self.ray_sin)
        orthog_to_intersect_dist = np.sqrt(np.maximum(
            other_agents[0].radius ** 2 - orthog_dist ** 2, 0
        ))
        intersect_distances = np.where(
            intersections_mask, along_dist - orthog_to_intersect_dist, np.inf
        )
        min_intersect_distances = np.min(np.where(
            intersect_distances > 0, intersect_distances, np.inf), axis=0
        )
        ray_distances = np.minimum(min_intersect_distances, self.max_range)

        ray_velocities = np.zeros(ray_distances.shape)
        vel_along_all_dir_all_crowd = np.einsum(
            "ij,ij->i",
            np.concatenate(
                [np.array(list(zip(self.ray_cos, self.ray_sin)))] * len(crowd_poss)
            ),
            np.repeat(crowd_vels, self.num_beams, axis=0)
        )
        vel_along_all_dir_all_crowd *= intersections_mask.flatten()
        viable_distances = np.where(
            intersect_distances > 0, intersect_distances, np.inf
        )
        crowd_min_dist_idx = np.argmin(  # which one is closer
            viable_distances, axis=0
        )
        vel_along_dir = vel_along_all_dir_all_crowd[
            crowd_min_dist_idx * self.num_beams + np.arange(self.num_beams)
        ]
        intersection_mask_dir = min_intersect_distances != np.inf
        ray_velocities = vel_along_dir * intersection_mask_dir
        return np.concatenate([ray_distances, ray_velocities])
