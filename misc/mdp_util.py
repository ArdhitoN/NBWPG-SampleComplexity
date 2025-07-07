import yaml

def parse_mdp_from_config(config_path):
    """
    Given a YAML config file, 
    this will parse the environment specification into an MDP dictionary.
    
    The returned dictionary mdp has the form:
    
      mdp[s][a] = {
          "r": expected_reward,
          "p": { s_prime: probability, ... }
      }
      
    It computes the expected reward as a weighted sum and aggregates
    probabilities for transitions that lead to the same state.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    transitions = config["transitions"]
    
    states_set = set()
    for s, actions in transitions.items():
        states_set.add(s)
        for a, spec in actions.items():
            if isinstance(spec, list):
                for item in spec:
                    states_set.add(item["next_state"])
            else:
                states_set.add(spec["next_state"])
    states = list(states_set)
    
    mdp = {}
    
    for s in states:
        mdp[s] = {}
        if s in transitions:
            for a, spec in transitions[s].items():
                if isinstance(spec, list):
                    expected_reward = sum(item["reward"] * item["prob"] for item in spec)
                    p_dict = {}
                    for item in spec:
                        next_s = item["next_state"]
                        p_dict[next_s] = p_dict.get(next_s, 0) + item["prob"]
                    mdp[s][a] = {"r": expected_reward, "p": p_dict}
                else:
                    mdp[s][a] = {"r": spec["reward"], "p": {spec["next_state"]: spec["prob"]}}
    return mdp