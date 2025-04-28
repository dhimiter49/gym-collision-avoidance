import numpy as np
import os
import operator
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from mpc.factory import get_mpc
from mpc.plan import Plan



MPC_DICT = {
    "-d": "simple",
    "-lp": "linear_plan",
    "-v": "velocity_control",
    "-cs": "cascading",
    "-vcs": "velocity_control_cascading"
}


PLAN_DICT = {
    "-lp": "Position",
    "-lpv": "PositionVelocity",
    "-vp": "Velocity",
}


class MPCPolicy(InternalPolicy):
    """
    Pre-trained policy using TRPL and ProDMP with MPC as controller.
    """
    def __init__(self, initial_heading):
        InternalPolicy.__init__(self, str="MPC")
        self.agent_dir = initial_heading


    def initialize_network(self, **kwargs):
        """
        Load the model parameters of either a default file, or if provided through
        kwargs, a specific path and/or tensorflow checkpoint.

        Args:
            kwargs['checkpt_name'] (str): name of checkpoint file to load (without file extension)
            kwargs['checkpt_dir'] (str): path to checkpoint

        """
        mpc_type = MPC_DICT["-v"]  # velocity control
        N = 21
        DT = 0.1

        self.planner = Plan(N, DT, 3)
        self.mpc = get_mpc(
            mpc_type,
            horizon=N,
            dt=DT,
            physical_space=0.4,
            const_dist_crowd=0.800001,
            agent_max_vel=3,
            agent_max_acc=1.5,
            n_crowd=0,
        )
        self.agent_vel = np.zeros(2)


    def find_next_action(self, obs, agents, i):
        """ Using only the dictionary obs, convert this to the vector needed for the GA3C-CADRL network, query the network, adjust the actions for this env.

        Args:
            obs (dict): this :class:`~gym_collision_avoidance.envs.agent.Agent` 's observation vector
            agents (list): [unused] of :class:`~gym_collision_avoidance.envs.agent.Agent` objects
            i (int): [unused] index of agents list corresponding to this agent

        Returns:
            [spd, heading change] command

        """
        pref_speed = obs['pref_speed']

        # prepare observation for MPC
        goal_dist = obs["dist_to_goal"]
        heading_to_goal = -obs["heading_ego_frame"]

        abs_heading_to_goal = heading_to_goal + self.agent_dir
        goal_rel_xy = goal_dist * np.array([
            np.cos(abs_heading_to_goal), np.sin(abs_heading_to_goal)
        ])
        crowd_poss = None
        crowd_vels = None
        if self.mpc.n_crowd > 0:
            crowd_poss = obs["other_agents_states"][:self.mpc.n_crowd, :2]
            crowd_vels = obs["other_agents_states"][:self.mpc.n_crowd, 2:4]

        walls = np.array([20, 20, 20, 20])
        obs = (goal_rel_xy, crowd_poss, self.agent_vel, crowd_vels, walls)

        # plan
        plan = self.planner.plan(obs)

        # predict next step
        pred_traj = self.mpc.get_action(plan, obs)
        next_vel = pred_traj[0]
        speed = np.linalg.norm(next_vel)
        heading = np.sign(next_vel[1]) * np.arccos(next_vel[0] / speed) -\
            self.agent_dir
        self.agent_vel = next_vel
        self.agent_dir += heading

        # adapt action to environment
        action = np.array([speed, heading])
        return action


if __name__ == '__main__':
    policy = MPCPolicy()
    policy.initialize_network()