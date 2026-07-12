import glob
import os
import yaml
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from agent.agent import Agent
from environment.env import Env

# Determine paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # code/experiment
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..'))  # code
ENV_DIR = os.path.join(PROJECT_ROOT, 'environment')           # code/environment
AGENT_DIR = os.path.join(PROJECT_ROOT, 'agent')           # code/environment
XHS_DIR = os.path.join(PROJECT_ROOT, 'xh_search_results')           # code/xh_search_results
EXP_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'experiment.yml')  # code/experiment/experiment.yml


def get_env_config_from_experiment(exp_config_path=EXP_CONFIG_PATH):
    """
    Read the active `env_config` from experiment.yml and return
    (env_config_filename, env_name), e.g. ("env-a1.yml", "env-a1").
    """
    with open(exp_config_path) as f:
        cfg = yaml.safe_load(f)
    env_config_filename = cfg.get('env_config')
    if not env_config_filename:
        raise ValueError(f"No 'env_config' key found in {exp_config_path}")
    env_name = os.path.splitext(os.path.basename(env_config_filename))[0]
    return env_config_filename, env_name



def enumerate_deterministic_policies(env):
    """
    Generate every deterministic policy as a tuple of actions per state.
    """
    actions_list = [list(np.where(env.valid_action_mask[s])[0])
                    for s in range(env.num_states)]
    return itertools.product(*actions_list)


def policy_to_pi(env, policy_actions):
    """
    Convert a deterministic policy tuple to a stochastic policy matrix.
    """
    pi = np.zeros((env.num_states, env.max_num_actions))
    for s, a in enumerate(policy_actions):
        pi[s, a] = 1.0
    return pi


def run_exhaustive_search(env_config_filename=None, prefix=None):
    """
    Run an exhaustive search over deterministic policies.

    If `env_config_filename` is given (e.g. "env-a1.yml") only that single
    environment is processed; otherwise the active env from experiment.yml
    is used. `prefix` defaults to the environment name (e.g. "env-a1"), so the
    output CSV is `{env_name}_exhaustive_search_results.csv`, matching what
    compute_method_effectiveness.py expects.
    """
    # Resolve which environment to process and the output prefix
    if env_config_filename is None:
        env_config_filename, env_name = get_env_config_from_experiment()
    else:
        env_name = os.path.splitext(os.path.basename(env_config_filename))[0]
    if prefix is None:
        prefix = env_name

    yaml_path = os.path.join(ENV_DIR, env_config_filename)

    results = []

    with open(yaml_path) as yf:
        raw_cfg = yaml.safe_load(yf)
    if 'transitions_converted' not in raw_cfg:
        raise ValueError(f"{os.path.basename(yaml_path)} has no 'transitions_converted' section")

    env = Env(config_path=yaml_path)
    agent = Agent(
        config_path=os.path.join(AGENT_DIR, 'agent.yml'),
        gamma=None,
        env=env
    )

    for policy in enumerate_deterministic_policies(env):
        # Set deterministic policy
        agent.pi = policy_to_pi(env, policy)
        # Compute gain and bias at initial state
        gain = agent.compute_gain()
        bias_vec = agent.compute_bias()
        bias_initial = bias_vec[env.initial_state]

        results.append({
            'environment': env_name,
            'policy': policy,
            'gain': gain,
            'bias': bias_initial
        })

    df = pd.DataFrame(results)

    # Save results to experiment folder
    out_csv = os.path.join(XHS_DIR, f'{prefix}_exhaustive_search_results.csv')
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")

    # # Plot: Gain vs Bias for all policies
    # plt.figure()
    # plt.scatter(df['gain'], df['bias'], alpha=0.7)
    # plt.xlabel('Gain')
    # plt.ylabel('Bias at Initial State')
    # plt.title('Gain vs Bias for Deterministic Policies')
    # plt.grid(True)
    # scatter_fp = os.path.join(XHS_DIR, f'{prefix}_gain_bias_scatter.png')
    # plt.tight_layout()
    # plt.savefig(scatter_fp)
    # print(f"Scatter plot saved to {scatter_fp}")

    return df


if __name__ == '__main__':
    # Uses the env_config from experiment.yml; output prefix = that env's name.
    run_exhaustive_search()
