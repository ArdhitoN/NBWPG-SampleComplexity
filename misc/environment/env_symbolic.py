import yaml
import random
import os

class Env:
    def __init__(self, config_path="env-a1.yml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.config = config
        self.name = os.path.splitext(os.path.basename(config_path))[0]
        
        self.transitions = self.config["transitions"]
        self.initial_state = self.config["initial_state"] 
        self.current_state = self.initial_state
        
        states_set = []
        actions_set = []
        for s, actions in self.transitions.items():
            states_set.append(s)
            for a, spec in actions.items():
                actions_set.append(a)
        self.states = list(states_set)
        self.actions = list(actions_set)
        


    def reset(self):
        self.current_state = self.initial_state
        return self.current_state

    def step(self, action):
        if action not in self.transitions[self.current_state]:
            raise ValueError(f"Action '{action}' is not valid in state '{self.current_state}'")
        
        transition_spec = self.transitions[self.current_state][action]
        if isinstance(transition_spec, list):
            outcome = random.choices(transition_spec,
                                     weights=[t["prob"] for t in transition_spec],
                                     k=1)[0]
        else:
            outcome = transition_spec
        
        self.current_state = outcome["next_state"]
        reward = outcome["reward"]
        done = False  
        return self.current_state, reward, done
