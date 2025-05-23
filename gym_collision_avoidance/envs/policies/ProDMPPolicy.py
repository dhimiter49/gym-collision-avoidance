import numpy as np
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from trust_region_projections.utils.custom_store import CustomStore
from trust_region_projections.algorithms.pg.pg import PolicyGradient
from trust_region_projections.utils.torch_utils import tensorize, get_numpy


class ProDMPPolicy(InternalPolicy):
    """
    Pre-trained policy using TRPL and ProDMP with MPC as controller.
    """

    def __init__(self, initial_heading):
        InternalPolicy.__init__(self, str="ProDMP")

        self.agent_vel = np.zeros(2)
        self.agent_dir = initial_heading
        self.lidar_angle = 2 * np.pi / 40


    def initialize_network(self, **kwargs):
        """
        Load the model parameters of either a default file, or if provided through
        kwargs, a specific path and/or tensorflow checkpoint.

        Args:
            kwargs['checkpt_name'] (str): name of checkpoint file to load
            kwargs['checkpt_dir'] (str): path to checkpoint

        """
        if "checkpt_name" in kwargs:
            checkpt_name = kwargs["checkpt_name"]

        store = CustomStore(
            storage_folder="", note=None, exp_id=checkpt_name, new=False, mode="a"
        )
        self.agent, _ = PolicyGradient.agent_from_data(
            store, train_steps=None, checkpoint_iteration=-1, testing=True
        )
        self.prodmp_env = self.agent.sampler.envs_test


    def find_next_action(self, obs, agents, i):
        """
        Args:
            obs (dict): this :class:`~gym_collision_avoidance.envs.agent.Agent`
            agents (list): [unused]
            i (int): [unused] index of agents list corresponding to this agent

        Returns:
            [spd, heading change] command

        """
        # prodmp observation
        lidar_abs = obs["laserscan"]
        abs_state = obs["agents_abs_states"]
        n_crowd = int(len(abs_state) - 3) // 3
        (agent_pos, agent_vel, goal_pos, crowd_poss, crowd_vels, crowd_goal_poss) = (
            abs_state[0],
            abs_state[1],
            abs_state[2],
            abs_state[3:3 + n_crowd],
            abs_state[3 + n_crowd:3 + 2 * n_crowd],
            abs_state[-n_crowd:],
        )
        goal_rel = goal_pos - agent_pos

        # prodmp environment setup
        prodmp_obs = np.concatenate([goal_rel, agent_vel, lidar_abs]).flatten()
        prodmp_obs = np.concatenate([prodmp_obs, [0]]).flatten()  # time input
        prodmp_obs = tensorize(prodmp_obs, self.agent.cpu, self.agent.dtype)


        # predict
        prodmp_weights = self.agent.policy(prodmp_obs, train=False)[0]
        prodmp_weights = [get_numpy(prodmp_weights)]

        # hard set environment in order to run ProDMP
        self.prodmp_env.reset()
        self.prodmp_env.venv.envs[0].hard_set_vars(
            {
                "_agent_pos": agent_pos,
                "_agent_vel": agent_vel,
                "_goal_pos": goal_pos,
                "_crowd_poss": crowd_poss,
                "_crowd_vels": crowd_vels,
                "_crowd_goal_poss": crowd_goal_poss,
            }
        )
        _, _, _, infos = self.prodmp_env.step(prodmp_weights)
        next_vel = infos[0]["step_actions"][0]

        # adapt action to environment
        speed = np.linalg.norm(next_vel)
        heading = np.sign(next_vel[1]) * np.arccos(next_vel[0] / speed) - self.agent_dir
        self.agent_vel = next_vel
        self.agent_dir += heading
        action = np.array([speed, heading])
        return action


if __name__ == "__main__":
    policy = ProDMPPolicy()
    policy.initialize_network(
        checkpt_name=(
            "/home/dhimiter/Documents/RAM/TrustRegionProjections/archive/"
            "mp_const_newest/32k_65k_131k-mpc/90463468-d862-4f10-be14-7854a4760ce2__/"
        )
    )
