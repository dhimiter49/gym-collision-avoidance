import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs import Config


class AgentsAbsStatesSensor(Sensor):
    """ A matrix of absolute positions and velocities.

    :param max_num_other_agents_observed: (int) only can observe up to this many agents

    """
    def __init__(
            self, max_num_other_agents_observed=Config.MAX_NUM_OTHER_AGENTS_OBSERVED,
    ):
        Sensor.__init__(self)
        self.name = 'agents_abs_states'
        self.max_num_other_agents_observed = max_num_other_agents_observed


    def sense(self, agents, agent_index, top_down_map=None):
        """ Return absolute positions for the current agent and the crowd.

        Args:
            agents (list): all :class:`~gym_collision_avoidance.envs.agent.Agent`
            agent_index (int): index of this agent (the one with this sensor)
            top_down_map (2D np array): binary image with 0 if that pixel is free space,
                1 if occupied (not used!)

        Returns:
            obs (np array): (number of crowd plus agent) x 6, 2-pos, 2-vel, 2-goal_pos

        """
        agent = agents[agent_index]
        agent_pos = agent.pos_global_frame
        agent_vel = agent.vel_global_frame
        agent_goal = agent.goal_global_frame
        other_agents = agents[:agent_index] + agents[agent_index + 1:]
        crowd_poss = np.array([a.pos_global_frame for a in other_agents])
        crowd_vels = np.array([a.vel_global_frame for a in other_agents])
        crowd_goals = np.array([a.goal_global_frame for a in other_agents])

        all_agents_obs = np.zeros(
            (self.max_num_other_agents_observed * 6 // 2, 2)
        )
        obs = np.concatenate([
            [agent_pos],
            [agent_vel],
            [agent_goal],
            crowd_poss,
            crowd_vels,
            crowd_goals,
        ])
        all_agents_obs[:len(obs)] = obs

        return all_agents_obs
