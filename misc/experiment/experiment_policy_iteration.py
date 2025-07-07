import os
import yaml
import matplotlib.pyplot as plt

from util import evaluate_all_policies, print_policy_evaluations
from mdp_util import parse_mdp_from_config
from agent.policy_iteration import DiscountedPolicyIterationAgent, AverageRewardPolicyIterationAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXP_CONFIG_PATH = os.path.join(BASE_DIR, "experiment.yml")
AGENT_CONFIG_PATH = os.path.join(BASE_DIR, "..", "agent", "agent.yml")

with open(EXP_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    
with open(AGENT_CONFIG_PATH, "r") as f:
    agent_config = yaml.safe_load(f)

env_config_filename = config['env_config'] 
ENV_CONFIG_PATH = os.path.join(BASE_DIR, "..", "environment", env_config_filename)

mdp = parse_mdp_from_config(ENV_CONFIG_PATH)

gamma = agent_config.get("gamma", 0.99) 
dpi_agent = DiscountedPolicyIterationAgent(mdp, gamma=gamma)
policy_d, V = dpi_agent.run_policy_iteration()

print(f"gamma: {gamma}")
print("Discounted Policy Iteration Results:")
print("Optimal Policy:", policy_d)
print("State Value Function:", V)

api_agent = AverageRewardPolicyIterationAgent(mdp)
policy_a, h, g = api_agent.run_policy_iteration()
print("\nAverage-Reward Policy Iteration Results:")
print("Optimal Policy:", policy_a)
print("Bias Function (h):", h)
print("Average Reward (g):", g)
print()

results_discounted = evaluate_all_policies(DiscountedPolicyIterationAgent, mdp, gamma=gamma)
print_policy_evaluations(results_discounted, scheme="discounted")

results_average = evaluate_all_policies(AverageRewardPolicyIterationAgent, mdp)
print_policy_evaluations(results_average, scheme="average")