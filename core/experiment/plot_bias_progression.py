import os
import glob
import numpy as np
import pandas as pd

from agent.agent import Agent
from environment.env import Env

# Paths setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # code/experiment
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..')) # code
FIGS_DIR = os.path.join(PROJECT_ROOT, 'figs')                   # code/figs
AGENT_DIR = os.path.join(PROJECT_ROOT, 'agent')                   # code/figs
EXPERIMENT_OUTPUT_DIR = os.path.join(FIGS_DIR, 'experiment_outputs_final')
ENV_OUTPUT_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, 'env-b3')  # adjust per env name
FILE_DIR = os.path.join(ENV_OUTPUT_DIR, 'visualizations')       # where CSVs are stored

# Environment and agent config
ENV_CONFIG = os.path.join(PROJECT_ROOT, 'environment', 'env-b3.yml')
AGENT_CONFIG = os.path.join(AGENT_DIR, 'agent.yml')

# Patterns for gain CSVs (discounted and discounting-free)
PATTERNS = [
    os.path.join(FILE_DIR, 'gain_progression_discounted_*_gamma_*.csv'),
    os.path.join(FILE_DIR, 'gain_progression_discounting_free_*_gamma_None.csv')
]


def compute_bias_progressions():
    # Gather all matching CSV files
    csv_paths = []
    for pat in PATTERNS:
        csv_paths.extend(glob.glob(pat))
    csv_paths = sorted(set(csv_paths))
    if not csv_paths:
        print(f"No files matching patterns: {PATTERNS}")
        return

    # Initialize environment and agent once
    env = Env(config_path=ENV_CONFIG)
    agent = Agent(config_path=AGENT_CONFIG, gamma=None, env=env)

    for gain_csv in csv_paths:
        df = pd.read_csv(gain_csv)
        biases = []
        for _, row in df.iterrows():
            # Set theta parameters
            agent.theta = np.array([row['theta0'], row['theta1']])
            agent.update_policy()
            bias_vec = agent.compute_bias()
            biases.append(bias_vec[env.initial_state])

        # Add bias column
        df_out = df.copy()
        df_out['bias'] = biases

        # Save as bias_progression...
        out_csv = gain_csv.replace('gain_progression', 'bias_progression')
        df_out.to_csv(out_csv, index=False)
        print(f"Saved bias progression to {out_csv}")


if __name__ == '__main__':
    compute_bias_progressions()
