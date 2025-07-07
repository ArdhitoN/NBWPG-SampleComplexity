from itertools import product

def enumerate_policies(mdp):
    state_actions = {s: list(mdp[s].keys()) for s in mdp}
    states = list(state_actions.keys())
    
    policies = []
    for actions in product(*(state_actions[s] for s in states)):
        
        policy = {s: a for s, a in zip(states, actions)}
        policies.append(policy)
    return policies

def evaluate_policy(agent_class, mdp, policy, tol=1e-6, max_eval_iter=1000, gamma=None):
    if gamma is not None:
        agent = agent_class(mdp, gamma=gamma, tol=tol)
    else:
        agent = agent_class(mdp, tol=tol, max_eval_iter=max_eval_iter)
    
    agent.policy = policy.copy()
    
    if hasattr(agent, "gamma"):
        agent.policy_evaluation()
        return policy, agent.V
    else:
        agent.policy_evaluation()
        return policy, agent.h, agent.g

def evaluate_all_policies(agent_class, mdp, tol=1e-6, max_eval_iter=1000, gamma=None):
    policies = enumerate_policies(mdp)
    results = []
    for policy in policies:
        result = evaluate_policy(agent_class, mdp, policy, tol=tol, max_eval_iter=max_eval_iter, gamma=gamma)
        results.append(result)
    return results


def print_policy_evaluations(results, scheme="discounted"):
    print(f"{scheme} reward policy evaluations:")
    for idx, res in enumerate(results, start=1):
        print(f"Policy pi{idx}: {res[0]}")
        if scheme == "discounted":
            policy, V = res
            print("  Discounted Value Function V:")
            for s, v in V.items():
                print(f"    {s}: {v:.4f}")
        else:
            policy, h, g = res
            print("  Average Reward g: {:.4f}".format(g))
            print("  Bias Function h:")
            for s, val in h.items():
                print(f"    {s}: {val:.4f}")
        print()
        