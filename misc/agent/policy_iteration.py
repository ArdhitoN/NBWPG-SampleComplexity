import numpy as np

class DiscountedPolicyIterationAgent:
    r"""
    Discounted Policy Iteration Agent for solving an MDP with a discount factor \(\gamma\).

    This agent finds a policy \(\pi\) and a value function \(V(s)\) satisfying the Bellman equation:
    
    .. math::
       V(s) = r(s, \pi(s)) + \gamma \sum_{s'} p(s' \mid s, \pi(s)) \, V(s'),
       
    where:
    
      - \(r(s, \pi(s))\) is the immediate expected reward,
      - \(\gamma \in (0,1)\) is the discount factor,
      - \(p(s' \mid s, \pi(s))\) is the probability of transitioning from state \(s\) to \(s'\) under action \(\pi(s)\).
    
    The MDP is provided as a dictionary:
    
        mdp[s][a] = {"r": expected_reward, "p": {s_prime: probability, ...}}
    """
    def __init__(self, mdp, gamma=0.99, tol=1e-6):
        self.mdp = mdp
        self.states = list(mdp.keys())
        self.gamma = gamma
        self.tol = tol
        
        # Initialize an arbitrary policy: choose action randomly for each state.
        self.policy = {}
        self.rng = np.random.default_rng(seed=42)
        for s in self.states:
            self.policy[s] = self.rng.choice(list(mdp[s].keys()))
        
        # Value function V(s) - Initialization
        self.V = {s: 0.0 for s in self.states}
    
    def policy_evaluation(self):
        r"""
        Evaluate the current policy \(\pi\) by solving the linear system
        
        .. math::
           V(s) = r(s, \pi(s)) + \gamma \sum_{s'} p(s' \mid s, \pi(s)) \, V(s')
           
        for all states \(s\). This is done by constructing the system in the form
        \(A V = b\) and solving it using `np.linalg.solve`.
        """
        n = len(self.states)
        A = np.zeros((n, n))
        b = np.zeros(n)
        state_index = {s: i for i, s in enumerate(self.states)}
        
        for s in self.states:
            i = state_index[s]
            a = self.policy[s]
            r_sa = self.mdp[s][a]["r"]
            b[i] = r_sa
            A[i, i] = 1.0
            for s_next, prob in self.mdp[s][a]["p"].items():
                j = state_index[s_next]
                A[i, j] -= self.gamma * prob
        sol = np.linalg.solve(A, b)
        for s, i in state_index.items():
            self.V[s] = sol[i]
    
    def policy_improvement(self):
        r"""
        Improve the policy \(\pi\) by updating it to
        
        .. math::
           \pi(s) = \arg\max_{a \in A(s)} \left\{ r(s,a) + \gamma \sum_{s'} p(s' \mid s,a) \, V(s') \right\}.
        
        Returns:
            True if the policy does not change (i.e. it is stable); otherwise, False.
        """
        policy_stable = True
        for s in self.states:
            old_action = self.policy[s]
            best_action = None
            best_value = -np.inf
            for a in self.mdp[s].keys():
                r_sa = self.mdp[s][a]["r"]
                value = r_sa
                for s_next, prob in self.mdp[s][a]["p"].items():
                    value += self.gamma * prob * self.V[s_next]
                if value > best_value:
                    best_value = value
                    best_action = a
            self.policy[s] = best_action
            if best_action != old_action:
                policy_stable = False
        return policy_stable
    
    def run_policy_iteration(self, max_iterations=1000):
        for _ in range(max_iterations):
            self.policy_evaluation()
            if self.policy_improvement():
                break

        return self.policy, self.V


class AverageRewardPolicyIterationAgent:
    r"""
    Average-Reward Policy Iteration Agent for solving an MDP under the long-run average reward criterion.

    This agent finds a policy \(\pi\), an average reward \(g\), and a differential value (bias) function \(h(s)\)
    that satisfy:
    
    .. math::
       h(s) = r(s,\pi(s)) - g + \sum_{s'} p(s' \mid s,\pi(s)) \, h(s'),
       
    for all states \(s\). Note that \(h(s)\) is only defined up to an additive constant.
   
    **Linear Algebra Perspective:**
    The above equation can be rearranged into a system of linear equations:
    
    .. math::
       (I - P_d) h = r_d - g\,e,
    
    where:
    
      - \(I\) is the identity matrix,
      - \(P_d\) is the state transition matrix under the policy \(\pi\) (with \(P_d(s,s') = p(s'|s,\pi(s))\)),
      - \(r_d\) is the vector of expected immediate rewards,
      - \(e\) is a vector of ones.
    
    Since \(P_d\) is stochastic (each row sums to 1), we have:
    
    .. math::
       (I - P_d) e = 0,
    
    meaning that \(e\) lies in the nullspace of \((I - P_d)\). Thus, if \(h\) is a solution, then so is
    
    .. math::
       h + c\,e \quad \text{for any scalar } c.
    
    This is why the solution for \(h\) is unique only up to an additive constant.
    
    To remove this ambiguity, we fix a reference state \(s_{\text{ref}}\) (typically the first state)
    and enforce:
    
    .. math::
       h(s_{\text{ref}}) = 0.
       
    Then, the average reward \(g\) can be computed from the equation for the reference state:
    
    .. math::
       g = r(s_{\text{ref}},\pi(s_{\text{ref}})) + \sum_{s'} p(s' \mid s_{\text{ref}},\pi(s_{\text{ref}})) \, h(s').
    """
    def __init__(self, mdp, tol=1e-6, max_eval_iter=1000):
        self.mdp = mdp
        self.states = list(mdp.keys())
        self.tol = tol
        self.max_eval_iter = max_eval_iter
        
        # Initialize an arbitrary policy.
        self.policy = {}
        self.rng = np.random.default_rng(seed=42)
        for s in self.states:
            self.policy[s] = self.rng.choice(list(mdp[s].keys()))
            
        # Initialize bias (differential value) and average reward.
        self.h = {s: 0.0 for s in self.states}
        self.g = 0.0
    
    def policy_evaluation(self):
        r"""
        Evaluate the current policy using an iterative relative value method.
        
        For each state \(s\), compute:
        
        .. math::
           h_{\text{new}}(s) = r(s,\pi(s)) - g + \sum_{s'} p(s' \mid s,\pi(s)) \, h(s').
        
        Then normalize the differential values by setting \(h(s_{\text{ref}})=0\), where \(s_{\text{ref}}\) is chosen as
        the first state in the list. This normalization fixes the arbitrary additive constant in \(h\).
        
        Finally, update the average reward \(g\) using the reference state:
        
        .. math::
           g = r(s_{\text{ref}},\pi(s_{\text{ref}})) + \sum_{s'} p(s' \mid s_{\text{ref}},\pi(s_{\text{ref}})) \, h(s').
        """
        h = {s: 0.0 for s in self.states}
        g = 0.0
        ref = self.states[0]
        
        for _ in range(self.max_eval_iter):
            h_new = {}
            for s in self.states:
                a = self.policy[s]
                r_sa = self.mdp[s][a]["r"]
                expected_h = sum(prob * h[s_next] for s_next, prob in self.mdp[s][a]["p"].items())
                h_new[s] = r_sa - g + expected_h
            
            # Normalize by forcing h(ref)=0
            offset = h_new[ref]
            for s in self.states:
                h_new[s] = h_new[s] - offset
            
            # Update g using the reference state
            a_ref = self.policy[ref]
            r_ref = self.mdp[ref][a_ref]["r"]
            expected_h_ref = sum(prob * h_new[s_next] for s_next, prob in self.mdp[ref][a_ref]["p"].items())
            g_new = r_ref + expected_h_ref
            
            diff = max(abs(h_new[s] - h[s]) for s in self.states) + abs(g_new - g)
            h = h_new
            g = g_new
            if diff < self.tol:
                break
        self.h = h
        self.g = g
    
    def policy_improvement(self):
        """
        Improves the policy based on the current bias function h and average reward g.
        The improvement step selects:
        
        .. math::
           \pi(s) = \arg\max_{a \in A(s)} \left\{ r(s,a) - g + \sum_{s'} p(s' \mid s,a) \, h(s') \right\}.
        """
        policy_stable = True
        for s in self.states:
            old_action = self.policy[s]
            best_action = None
            best_value = -np.inf
            for a in self.mdp[s].keys():
                r_sa = self.mdp[s][a]["r"]
                expected_h = sum(prob * self.h[s_next] for s_next, prob in self.mdp[s][a]["p"].items())
                value = r_sa - self.g + expected_h
                if value > best_value:
                    best_value = value
                    best_action = a
            self.policy[s] = best_action
            if best_action != old_action:
                policy_stable = False
        return policy_stable
    
    def run_policy_iteration(self, max_iterations=100):
        for _ in range(max_iterations):
            self.policy_evaluation()
            if self.policy_improvement():
                break
            
        return self.policy, self.h, self.g
    