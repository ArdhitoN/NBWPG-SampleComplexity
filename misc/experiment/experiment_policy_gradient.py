import os
import yaml
import numpy as np

import matplotlib.pyplot as plt

from agent.agent_policy_gradient import PolicyGradientAgent

from environment.env_symbolic import Env

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_CONFIG_PATH = os.path.join(BASE_DIR, "experiment.yml")
AGENT_CONFIG_PATH = os.path.join(BASE_DIR, "..", "agent", "agent.yml")

def run_experiment():
    with open(EXP_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    episodes = config.get("episodes", 5000)
    max_steps = config.get("max_steps", 100)
    log_interval = config.get("log_interval", 100)
    evaluate = config.get("evaluation", False)
    
    env_config_filename = config.get("env_config", "env-a1.yml")
    ENV_CONFIG_PATH = os.path.join(BASE_DIR, "..", "environment", env_config_filename)
    
    print(ENV_CONFIG_PATH)
    env = Env(ENV_CONFIG_PATH)
    agent = PolicyGradientAgent(env=env, config_path=AGENT_CONFIG_PATH)
    
    # agent.actions = env.actions

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        trajectory = [] 
        total_reward = 0

        for step in range(max_steps):
            valid_actions = list(env.transitions[state].keys())
            # action = agent.get_action(state, valid_actions)
            action = agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            
            trajectory.append((state, action, reward, done))
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        valid_actions = list(env.transitions[state].keys())
        agent.update(trajectory, valid_actions)
        rewards_per_episode.append(total_reward)

        if episode % log_interval == 0:
            avg_reward = total_reward / max_steps
            print(f"Episode {episode}, Total Reward: {total_reward}, Average Reward: {avg_reward}")
            
    # Evaluation
    if evaluate:
        eval_rewards = []
        for _ in range(10):  
            reward = agent.evaluate(max_steps)
            eval_rewards.append(reward)
        print(f"Evaluation - Avg Reward: {np.mean(eval_rewards):.2f}, "
            f"Std Dev: {np.std(eval_rewards):.2f}")

    plt.plot(range(episodes), rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Policy Gradient Agent Performance")
    plt.show()

if __name__ == "__main__":
    run_experiment()
