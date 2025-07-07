import yaml
import random
import os
import numpy as np

class Env:
    def __init__(self, config_path="env-a1.yml", seed=None):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.rng = np.random.default_rng(seed)
        
        # S, A, P, r, p0
        self.config = config
        self.name = os.path.splitext(os.path.basename(config_path))[0]
        
        self.transitions = self.config["transitions_converted"]
        
        # Construct S
        self.states = sorted(self.transitions.keys())
        self.num_states = len(self.states)
        
        # Construct A and find maximum number of actions in one state (needed for the dimension of P and r)
        self.actions_per_state = {}
        
        max_num_actions = 0
        for state in self.states:
            actions = sorted(self.transitions[state].keys())
            self.actions_per_state[state] = actions
            max_num_actions = max(max_num_actions, len(actions))
            
        self.max_num_actions = max_num_actions

        # Construct P[s, a, s'], r[s, a, s'], and action mask for determining validity for actions
        self.P = np.zeros((self.num_states, self.max_num_actions, self.num_states)) 
        self.r = np.zeros((self.num_states, self.max_num_actions, self.num_states))
        self.valid_action_mask = np.zeros((self.num_states, self.max_num_actions), dtype=bool)
                
        for current_state, actions in self.transitions.items():
            for action, next_states in actions.items():
                for next_state, spec in next_states.items():
                    self.P[current_state, action, next_state] = spec["prob"]
                    self.r[current_state, action, next_state] = spec["reward"]
                
                self.valid_action_mask[current_state, action] = True
                    
        # Build initial state distribution (p0)
        self.initial_state_dist = np.zeros(self.num_states)
        for state, prob in self.config["initial_state_dist"].items():
            self.initial_state_dist[state] = prob            
        
        self.initial_state = self.rng.choice(self.states, p=self.initial_state_dist)
        self.current_state = self.initial_state
        
        assert np.isclose(np.sum(self.initial_state_dist), 1.0), "Initial state distribution does not sum to 1"
        
        prob_sums = np.sum(self.P, axis=2)
        assert np.all(np.where(self.valid_action_mask, np.isclose(prob_sums, 1.0), prob_sums == 0)), \
            "Some transition probabilities don't sum to 1 (valid actions) or 0 (invalid actions)"        
        
        # print("States:", self.states)
        # print("Actions per state:", self.actions_per_state)
        # print("Action mask:\n", self.valid_action_mask)
        # print("P:\n", self.P)
        # print("r:\n", self.r)
        # print("Initial state dist:", self.initial_state_dist)
        # print("Initial state:", self.initial_state)
        # print(f"max num actions: {self.max_num_actions}")
        
    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def step(self, action_idx):
        done = False  
        info = {}
        
        if not self.valid_action_mask[self.current_state][action_idx]:
            raise ValueError(f"Action '{action_idx}' is not valid in state '{self.current_state}'")
        
        next_state = self.rng.choice(self.states, p=self.P[self.current_state][action_idx])
        reward = self.r[self.current_state][action_idx][next_state]
        self.current_state = next_state
        
        return self.current_state, reward, done, info
