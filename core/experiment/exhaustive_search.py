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


def run_exhaustive_search(prefix='phase1'):
    
    results = []

    # Find all environment YAML files
    yml_paths = glob.glob(os.path.join(ENV_DIR, '*.yml'))
    for yaml_path in yml_paths:
        with open(yaml_path) as yf:
            raw_cfg = yaml.safe_load(yf)
        if 'transitions_converted' not in raw_cfg:
            print(f"Skipping {os.path.basename(yaml_path)}: no 'transitions_converted' section")
            continue
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
                'environment': os.path.splitext(os.path.basename(yaml_path))[0],
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
    run_exhaustive_search(prefix='test')
