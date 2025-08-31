import numpy as np
import gymnasium as gym
import fancy_gym
import stable_baselines3 as sbl
from stable_baselines3.common.monitor import Monitor
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from cnn1d_feature_extractor import CNN1dFE


def make_env(env_id: str, **kwargs) -> callable:
    """
    returns callable to create gym environment or monitor

    Args:
        env_id: gym env ID

    Returns: callable for env constructor
    """
    def _get_env():
        env = gym.make(env_id, **kwargs)

        return Monitor(env)

    return _get_env


class SACPolicy(InternalPolicy):
    """
    Pre-trained policy using SAC from sbl3.
    """
    def __init__(self, initial_heading):
        InternalPolicy.__init__(self, str="SAC")
        self.agent_vel = np.zeros(2)
        self.agent_dir = initial_heading
        self.AGENT_MAX_VEL = 1.


    def initialize_network(self, **kwargs):
        """
        Load the model parameters of either a default file, or if provided through
        kwargs, a specific path and/or tensorflow checkpoint.

        Args:
            kwargs['checkpt_name'] (str): name of checkpoint file to load
            kwargs['checkpt_dir'] (str): path to checkpoint

        """
        self.n_crowd = kwargs["n_crowd"]
        checkpt_name = kwargs["checkpt_name"]
        level = checkpt_name.count("/")
        steps = checkpt_name.split("/")[-1].split("_")[2]
        env_path = "/".join(checkpt_name.split("/")[:level]) +\
            "/rl_model_vecnormalize_" + steps + "_steps.pkl"
        path_dirs = checkpt_name.split("/")
        env_name = path_dirs[
            np.where(list("Navigation" in dir for dir in path_dirs))[0][0]
        ]
        env_id = "fancy/" + env_name
        env_fns = [make_env(env_id) for _ in range(1)]
        self.env = VecNormalize.load(env_path, DummyVecEnv(env_fns))
        self.env.env_method("set_num_crowd", self.n_crowd)
        self.env.env_method("set_wxh", 50, 50)
        self.env.training = False
        self.env.norm_reward = False
        self.model = sbl.SAC.load(checkpt_name, env=self.env)
        self.n_crowd = kwargs["n_crowd"]


    def find_next_action(self, obs, agents, i):
        """
        Args:
            obs (dict): this :class:`~gym_collision_avoidance.envs.agent.Agent`
            agents (list): [unused]
            i (int): [unused] index of agents list corresponding to this agent

        Returns:
            [spd, heading change] command

        """
        abs_state = obs["agents_abs_states"]
        (agent_pos, agent_vel, goal_pos, crowd_poss, crowd_vels, crowd_goal_poss) = (
            abs_state[0],
            abs_state[1],
            abs_state[2],
            abs_state[3:3 + self.n_crowd],
            abs_state[3 + self.n_crowd:3 + 2 * self.n_crowd],
            abs_state[3 + 2 * self.n_crowd:3 + 3 * self.n_crowd],
        )
        # hard set environment in order to run ProDMP
        self.env.reset()
        self.env.env_method(
            "hard_set_vars",
            {
                "_agent_pos": agent_pos,
                "_agent_vel": agent_vel,
                "_goal_pos": goal_pos,
                "_crowd_poss": crowd_poss,
                "_crowd_vels": crowd_vels,
                "_crowd_goal_poss": crowd_goal_poss,
            }
        )
        obs = self.env.normalize_obs(self.env.env_method("get_obs")[0])
        next_vel, _ = self.model.predict(obs)

        vel_norm = np.linalg.norm(next_vel)
        if vel_norm > self.AGENT_MAX_VEL:
            next_vel *= self.AGENT_MAX_VEL / vel_norm
        return next_vel


if __name__ == '__main__':
    policy = SACPolicy(0)
    policy.initialize_network()