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
        self.n_crowd = kwargs["n_crowd"]
        radius = kwargs["radius"]
        radius_crowd = kwargs.get("radius_crowd", None)
        self.agent_dir = kwargs.get("initial_heading", 0)
        mpc_type = MPC_DICT["-v"]  # velocity control
        N = 10
        DT = 0.1
        max_vel = 1.0
        max_acc = 10.0
        stability_coeff = 0.15

        self.planner = Plan(N, DT, max_vel)
        self.mpc = get_mpc(
            mpc_type,
            horizon=N,
            dt=DT,
            physical_space=radius,
            radius_crowd=radius_crowd,
            const_dist_crowd=0.815001,
            agent_max_vel=max_vel,
            agent_max_acc=max_acc,
            n_crowd=self.n_crowd,
            uncertainty="dist",
            stability_coeff=stability_coeff,
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
        # see policies/sensors/AgentAbsStatesSensor.py
        (
            agent_pos,
            agent_vel,
            goal_pos,
            crowd_poss,
            crowd_vels,
        ) = abs_state[0], \
            abs_state[1], \
            abs_state[2], \
            abs_state[3:3 + self.n_crowd], \
            abs_state[3 + self.n_crowd: 3 + 2 * self.n_crowd]
        goal_rel = goal_pos - agent_pos
        crowd_poss_rel = crowd_poss - agent_pos

        walls = np.array([20, 20, 20, 20])
        obs = (goal_rel, crowd_poss_rel, agent_vel, crowd_vels, walls)

        # plan
        plan = self.planner.plan(obs)

        # predict next step
        pred_traj, _ = self.mpc.get_action(plan, obs)
        next_vel = pred_traj[0]

        # adapt action to environment
        speed = np.linalg.norm(next_vel)
        if speed == 0:
            heading = -self.agent_dir
        else:
            heading = np.sign(next_vel[1]) * np.arccos(next_vel[0] / speed) -\
                self.agent_dir
        self.agent_dir += heading
        action = np.array([speed, heading])
        return action


if __name__ == '__main__':
    policy = MPCPolicy(0)
    policy.initialize_network()