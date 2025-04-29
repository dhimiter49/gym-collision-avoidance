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
        self.agent_vel = np.zeros(2)
        self.agent_dir = initial_heading


    def initialize_network(self, **kwargs):
        """
        Load the model parameters of either a default file, or if provided through
        kwargs, a specific path and/or tensorflow checkpoint.

        Args:
            kwargs['checkpt_name'] (str): name of checkpoint file to load
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
            n_crowd=1,
        )


    def find_next_action(self, obs, agents, i):
        """

        Args:
            obs (dict): this :class:`~gym_collision_avoidance.envs.agent.Agent`
            agents (list): [unused]
            i (int): [unused] index of agents list corresponding to this agent

        Returns:
            [spd, heading change] command

        """
        # prepare observation for MPC
        abs_state = obs["agents_abs_states"]
        n_crowd = int((len(abs_state) - 3) // 3)  # 3 relates to agent states
        agent_pos = abs_state[0]
        agent_vel = abs_state[1]
        goal_rel = abs_state[2] - agent_pos
        crowd_poss = abs_state[3:3 + n_crowd].reshape(n_crowd, 2) - agent_pos
        crowd_vels = abs_state[3 + n_crowd:3 + 2 * n_crowd].reshape(n_crowd, 2)

        walls = np.array([20, 20, 20, 20])
        obs = (goal_rel, crowd_poss, agent_vel, crowd_vels, walls)

        # plan
        plan = self.planner.plan(obs)

        # predict next step
        pred_traj = self.mpc.get_action(plan, obs)
        next_vel = pred_traj[0]

        # adapt action to environment
        speed = np.linalg.norm(next_vel)
        heading = np.sign(next_vel[1]) * np.arccos(next_vel[0] / speed) -\
            self.agent_dir
        self.agent_vel = next_vel
        self.agent_dir += heading
        action = np.array([speed, heading])
        return action


if __name__ == '__main__':
    policy = MPCPolicy()
    policy.initialize_network()