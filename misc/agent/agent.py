import yaml
import numpy as np

class RLAgent:
    def __init__(self, config_path="agent.yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.learning_rate = config.get("learning_rate", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.1)
        self.policy_type = config.get("policy", "epsilon-greedy")
        self.seed = config.get("seed", None)
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.actions = config.get("actions", ["a00", "a01", "a10", "a11"])
        self.q_table = {}

    def get_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = self.actions

        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in valid_actions}

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)

        q_values = {action: self.q_table[state].get(action, 0.0) for action in valid_actions}
        return max(q_values, key=q_values.get)
        
    def update(self, state, action, reward, next_state, valid_actions_next=None):
        if valid_actions_next is None:
            valid_actions_next = self.actions

        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in valid_actions_next}

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])