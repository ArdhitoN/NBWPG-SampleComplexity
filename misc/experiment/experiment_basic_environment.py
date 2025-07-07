import yaml
import matplotlib.pyplot as plt
import os

from agent.agent import RLAgent
from environment.env_symbolic import Env

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXP_CONFIG_PATH = os.path.join(BASE_DIR, "experiment.yml")
AGENT_CONFIG_PATH = os.path.join(BASE_DIR, "..", "agent", "agent.yml")

def run_experiment():
    with open(EXP_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
        
    episodes = config["episodes"]
    max_steps = config["max_steps"]
    log_interval = config["log_interval"]
    
    env_config_filename = config['env_config']  
    ENV_CONFIG_PATH = os.path.join(BASE_DIR, "..", "environment", env_config_filename)
    
    env = Env(ENV_CONFIG_PATH)
    agent = RLAgent(AGENT_CONFIG_PATH)
    
    agent.actions = env.actions

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            valid_actions = list(env.transitions[state].keys())
            action = agent.get_action(state, valid_actions)  
            
            next_state, reward, done = env.step(action)
            
            valid_actions_next = list(env.transitions[next_state].keys())
            agent.update(state, action, reward, next_state, valid_actions_next)            
            
            state = next_state
                
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)
        
        if episode % log_interval == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    plt.plot(range(episodes), rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("RL Agent Performance")
    plt.show()
     
if __name__ == "__main__":
    run_experiment()
