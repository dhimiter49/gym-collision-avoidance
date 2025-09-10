import os
import sys
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["GYM_CONFIG_CLASS"] = "FullTestSuite"
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.experiments.src.env_utils import (
    create_env,
    policies,
    run_episode,
    store_stats,
)


def reset_env(
    env,
    test_case_fn,
    test_case_args,
    test_case,
    num_agents,
    policies_dict,
    policy,
    prev_agents,
    exp_name
):
    test_case_args["num_agents"] = num_agents
    test_case_args["test_case_index"] = test_case
    test_case_args["prev_agents"] = prev_agents
    if isinstance(policy, list):
        env.unwrapped.plot_policy_name = "mix" + exp_name
        test_case_args["policies"] = [policies_dict[p]["policy"] for p in policy]
        test_case_args["agents_sensors"] = [policies_dict[p]["sensors"] for p in policy]

        test_case_args["agents_dynamics"] = []
        for p in policy:
            if "MPC" == p or "ProDMP" == p or "SAC" == p:
                test_case_args["agents_dynamics"].append("myunicycle")
            else:
                test_case_args["agents_dynamics"].append("unicycle")
    else:
        env.unwrapped.plot_policy_name = policy + exp_name
        test_case_args["policies"] = policies_dict[policy]["policy"]
        if "sensors" in policies[policy]:
            test_case_args["agents_sensors"] = policies_dict[policy]["sensors"]
        else:
            test_case_args["agents_sensors"] = []
        if "MPC" == policy or "ProDMP" == policy or "SAC" == policy:
            test_case_args["agents_dynamics"] = "myunicycle"
        else:
            test_case_args["agents_dynamics"] = "unicycle"

    agents = test_case_fn(**test_case_args)
    radius_agents = [agent.radius for agent in agents]
    if prev_agents is None:
        for i, (p, agent) in enumerate(zip(policy, agents)):
            if "MPC" == p:
                agent.policy.initialize_network(
                    n_crowd=num_agents - 1,
                    radius_crowd=np.delete(radius_agents, i),
                    radius=radius_agents[i],
                    initial_heading=agent.heading_global_frame
                )
            if "ProDMP" == p:
                checkpt_name = (
                    # "/home/dhimiter/Documents/RAM/TrustRegionProjections/archive/"
                    # "const_lidar_vel/prodmp_mpc_cnn/scale8/"
                    # "34686aa4-7042-4853-bf2b-bd148d3b3731_"
                    "/home/dhimiter/Documents/RAM/TrustRegionProjections/results/"
                    "mp_config/CrowdNavigationORCALiDARVel-v0/"
                    "dcd8c1df-434c-4761-b304-5c3a6e470f7d"
                    # "75fef92e-725d-46f2-9c8e-cfcfd62a6efa"
                    # "/home/dhimiter/Documents/RAM/TrustRegionProjections/results/"
                    # "mp_config/CrowdNavigationConstLiDARVel-v0/"
                    # "0e61cde4-e62b-4549-8a96-09d84e5ac280/"
                )
                agent.policy.initialize_network(
                    checkpt_name=checkpt_name,
                    n_crowd=num_agents - 1,
                )
            if "SAC" == p:
                checkpt_name = (
                    # "/home/dhimiter/Documents/RAM/sbl/exp/CrowdNavigationConstLiDARVel-v0/"
                    # "/sac/12_08_2025-10:55:14-4869/model_sac/rl_model_50000000_steps.zip"
                    "/home/dhimiter/Documents/RAM/sbl/exp/CrowdNavigationORCALiDARVel-v0/"
                    "/sac/19_08_2025-10:49:18-7205/model_sac/rl_model_50000000_steps.zip"
                )
                agent.policy.initialize_network(
                    checkpt_name=checkpt_name,
                    n_crowd=num_agents - 1,
                )
            if "checkpt_name" in policies_dict[p]:
                agent.policy.env = env
                agent.policy.initialize_network(**policies_dict[p])
            if "sensor_args" in policies_dict[p]:
                for sensor in agent.sensors:
                    sensor.set_args(policies_dict[p]["sensor_args"])
    env.set_agents(agents)
    init_obs = env.reset()
    env.unwrapped.test_case_index = test_case
    return init_obs


def main():
    exp_name = "" if "-n" not in sys.argv else sys.argv[sys.argv.index("-n") + 1]
    if exp_name != "":
        exp_name = "_" + exp_name
    np.random.seed(0)

    test_case_fn = tc.full_test_suite
    test_case_args = {}

    if Config.FIXED_RADIUS_AND_VPREF:
        radius_bounds = [0.5, 0.5]
        test_case_args["vpref_constraint"] = True
        test_case_args["radius_bounds"] = radius_bounds
        vpref1_str = "vpref1.0_r{}-{}/".format(
            radius_bounds[0], radius_bounds[1]
        )
    else:
        vpref1_str = ""

    env = create_env()

    print(
        "Running {test_cases} test cases for {num_agents} for policies:"
        " {policies}".format(
            test_cases=Config.NUM_TEST_CASES,
            num_agents=Config.NUM_AGENTS_TO_TEST,
            policies=Config.POLICIES_TO_TEST,
        )
    )
    with tqdm(
        total=len(Config.NUM_AGENTS_TO_TEST)
        * len(Config.POLICIES_TO_TEST)
        * Config.NUM_TEST_CASES
    ) as pbar:
        for num_agents in Config.NUM_AGENTS_TO_TEST:
            env.set_plot_save_dir(
                os.path.dirname(os.path.realpath(__file__))
                + "/../results/full_test_suites/{vpref1_str}{num_agents}_agents/figs/"
                .format(vpref1_str=vpref1_str, num_agents=num_agents)
            )
            # for policy in Config.POLICIES_TO_TEST:
            np.random.seed(0)
            prev_agents = None
            df = pd.DataFrame()
            policy_name = "MIX" if len(Config.POLICIES_TO_TEST) else Config.POLICIES_TO_TEST[0]
            for test_case in range(Config.NUM_TEST_CASES):
                ##### Actually run the episode ##########
                _ = reset_env(
                    env,
                    test_case_fn,
                    test_case_args,
                    test_case,
                    num_agents,
                    policies,
                    Config.POLICIES_TO_TEST,
                    prev_agents,
                    exp_name
                )
                episode_stats, prev_agents = run_episode(env)
                df = store_stats(
                    df,
                    {"test_case": test_case, "policy_id": policy_name + exp_name},
                    episode_stats,
                )
                ########################################
                pbar.update(1)

            if Config.RECORD_PICKLE_FILES:
                file_dir = os.path.dirname(
                    os.path.realpath(__file__)
                ) + "/../results/full_test_suites/{vpref1_str}".format(
                    vpref1_str=vpref1_str
                )
                file_dir += "{num_agents}_agents/stats/".format(
                    num_agents=num_agents
                )
                os.makedirs(file_dir, exist_ok=True)
                log_filename = file_dir + "/stats_{}.p".format(
                    policy_name + exp_name
                )
                # log_filename = file_dir+'/stats_{}_{}.p'.format(policy, now.strftime("%m_%d_%Y__%H_%M_%S"))
                df.to_pickle(log_filename)

    return True


if __name__ == "__main__":
    main()
    print("Experiment over.")
