import numpy as np
import random
import yaml


class PolicyGradientAgent:
    def __init__(self, env, config_path="agent.yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.env = env
        self.lr_policy = config["learning_rate"]
        self.lr_value = config["learning_rate_v"]
        self.gamma = config["gamma"]
        self.seed = config["seed"]
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.theta_disc = {state: np.zeros(len(self.env.actions)) for state in self.env.states}
        self.theta_avg = {state: np.zeros(len(self.env.actions)) for state in self.env.states}

        self.value_w_disc = {state: 0 for state in self.env.states}
        self.value_w_avg = {state: 0 for state in self.env.states}  
        
    
    def get_policy(self, state, valid_actions, use_discounted=True):
        """Compute with softmax"""
        if use_discounted:
            preferences = self.theta_disc[state]
        else:
            preferences = self.theta_avg[state]
        
        valid_action_indices = [self.env.actions.index(action) for action in valid_actions]
        
        valid_preferences = preferences[valid_action_indices]
        exp_preferences = np.exp(valid_preferences - np.max(valid_preferences))
        policy = exp_preferences / np.sum(exp_preferences)
        
        full_policy = np.zeros(len(self.env.actions))
        for i, idx in enumerate(valid_action_indices):
            full_policy[idx] = policy[i]
        
        return full_policy
    

    def select_action(self, state, valid_actions=None, use_discounted=True):
        if valid_actions is None:
            raise ValueError(f"No valid action for agent.")

            
        policy = self.get_policy(state, valid_actions, use_discounted)
        # action = np.random.choice(valid_actions, p=policy)

        action_idx = np.random.choice(len(self.env.actions), p=policy)

        # print(f'policy: {policy}, valid_actions: {valid_actions}, action_idx: {action_idx}')        
        return self.env.actions[action_idx]
    
    
    def get_value(self, state, use_discounted=True):
        """Return value estimate for a state"""
        return self.value_w_disc[state] if use_discounted else self.value_w_avg[state]
    
    def compute_returns(self, rewards, dones):
        discounted_return = 0 
        discounted_returns = []
        for t in range(len(rewards) - 1, -1, -1):
            discounted_return = rewards[t] + self.gamma * discounted_return * (1 - dones[t])
            discounted_returns.insert(0, discounted_return)
            
        avg_return = np.mean(rewards)
        return discounted_returns, avg_return
    
    def update(self, trajectory, valid_actions):
        states, actions, rewards, dones = zip(*trajectory)
        discounted_returns, avg_return = self.compute_returns(rewards, dones)
        
        # For discounted reward
        for t, (state, action) in enumerate(zip(states, actions)):
            action_idx  = self.env.actions.index(action)
            policy_disc, policy_avg = self.get_policy(state, valid_actions=valid_actions, use_discounted=True), self.get_policy(state, valid_actions=valid_actions, use_discounted=False)
            grad_log_pi_disc, grad_log_pi_avg = np.zeros(len(self.env.actions)), np.zeros(len(self.env.actions))
            grad_log_pi_disc[action_idx], grad_log_pi_avg[action_idx] = 1 - policy_disc[action_idx], 1 - policy_avg[action_idx]
            for i in range(len(self.env.actions)):
                if i != action_idx:
                    grad_log_pi_disc[i], grad_log_pi_avg[i]  = -policy_disc[i], -policy_avg[i]
                
                    
            discounted_return = discounted_returns[t]
            
            # Calculate advantage (return - value estimate)
            value_estimate_disc, value_estimate_avg = self.get_value(state, use_discounted=True), self.get_value(state, use_discounted=False)
            discounted_advantage_disc = discounted_return - value_estimate_disc 
            discounted_advantage_avg = avg_return - value_estimate_avg
            
            # Update theta
            self.theta_disc[state] += self.lr_policy * discounted_advantage_disc * grad_log_pi_disc
            self.theta_avg[state] += self.lr_policy * discounted_advantage_avg * grad_log_pi_avg
            
            # Update v(s, w)
            # grad v(s, w) = 1, as value_w is a table-based func, where ∇_w v(s, w) = 1
            self.value_w_disc[state] += self.lr_value * discounted_advantage_disc
            self.value_w_avg[state] += self.lr_value * discounted_advantage_avg
             
             
            
        
    def evaluate(self, max_steps, use_discounted=True):
        """Evaluate policy performance (using greedy actions for simplicity)."""
        state = self.env.reset()
        total_reward = 0
        for _ in range(max_steps):
            valid_actions = list(self.env.transitions[state].keys())
            policy = self.get_policy(state, valid_actions, use_discounted)
            action_idx = np.argmax(policy)  # Greedy for evaluation
            action = self.env.actions[action_idx]
            state, reward, done = self.env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward
    
