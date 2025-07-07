
import numpy as np
import matplotlib.pyplot as plt
import yaml
from math import exp
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, 
                 config_path="agent_exact.yml", 
                 gamma=None, 
                 seed=None, 
                 env=None
                 ):
        
        self.config_path = config_path     
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
        self.seed = seed    
        self.rng = np.random.default_rng(seed)        

        self.gamma = gamma
        # self.lr = config.get("learning_rate", 0.1)
        
        self.env = env
        self.P = self.env.P
        self.r = self.env.r
        
        self.theta = np.zeros(self.env.max_num_actions) 
        self.pi = np.zeros((self.env.num_states, self.env.max_num_actions))
        self.theta_trajectory = [self.theta.copy()]        
        self.theta_trajectory_gain = [self.theta.copy()]
        self.theta_trajectory_bias = []
        self.update_policy()
        
        self.max_gain = -np.inf
        self.is_gain_converged = False
        self.delta = 1
        # self.beta = 1e-3
        self.beta = 1e-1
        # self.beta = 100
        self.beta_tilda = 0
        self.beta_div = 10
        self.optimize_gain = True
        self.gain_convergence_treshold = 1e-2
        self.gain_grad_norm_history = []
        
        
        self.average_reward = 0
        
        self.convergence_treshold = 1e-2
        self.total_samples_for_convergence = None
        self.gain_samples_for_convergence = None
        
        

    def update_policy(self):
        """Compute policy π(a|s; θ) using sigmoid parameterization."""
        # State indices: f(s) = s, where s = 0, 1, ..., num_states-1
        s = np.arange(self.env.num_states)
        # Compute z = f(s) * θ₀ + θ₁ for all states
        z = s * self.theta[0] + self.theta[1]
        # Probability of action 0: σ(z)
        
        # logger.info(f"z: {z}")
        pi_0 = 1 / (1 + np.exp(-z))
        # Probability of action 1: 1 - σ(z)
        pi_1 = 1 - pi_0
        # Assign probabilities to policy matrix
        self.pi[:, 0] = pi_0
        self.pi[:, 1] = pi_1
        
    def get_policy(self, state):
        return self.pi[state]
    
    def select_action(self, state):
        return self.rng.choice(self.env.max_num_actions, p=self.get_policy(state))

    def compute_transition_probabilities_and_rewards(self):
        """Compute P_pi (num_states x num_states) and r_pi (num_states) under current policy."""
        # P_pi[s, s'] = Σ_a π(a|s) P(s'|s, a)
        P_pi = np.sum(self.P * self.pi[:, :, None], axis=1)
        # r_pi[s] = Σ_a π(a|s) Σ_s' P(s'|s, a) r(s, a, s')
        r_pi = np.sum(self.P * self.r * self.pi[:, :, None], axis=(1, 2))
    
        return P_pi, r_pi
    
    def get_pstar_matrix_for_unichain(self, P_pi: np.ndarray) -> np.ndarray:
        """
        Compute the P* matrix for a unichain by solving:
           (P_pi - I)^T * mu = 0
        with the trick of replacing the last row with ones and the corresponding right-hand side entry to 1.
        Then, tile mu to form the matrix with each row equal to the stationary distribution.
        
        """
        nS = self.pi.shape[0]
        
        # Create A = (P_pi - I).T
        A = (P_pi - np.eye(nS)).T 
        # Replace the last row with ones (trick to enforce sum(mu)=1)
        A[-1, :] = np.ones(nS) # 
        # Build right-hand side vector b (last entry is 1)
        b = np.zeros(nS)
        b[-1] = 1
        
        # use pseudoinverse if matrix singular
        try:
            mu = np.linalg.solve(A, b)
            
        except np.linalg.LinAlgError:
            mu = np.linalg.pinv(A) @ b
            logger.warning(f"mu matrix singular, using pseudo-inverse")
    
        P_pi_star = np.tile(mu, (nS, 1))
        
        return P_pi_star

    
    def compute_stationary_distribution(self):
        """
        Compute the stationary distribution P_pi_star for the Markov chain induced by π.
        Since P_pi_star has every row equal to the stationary distribution,
        we extract it from the first row.
        """
        P_pi, _   = self.compute_transition_probabilities_and_rewards()
        P_pi_star = self.get_pstar_matrix_for_unichain(P_pi)
        return P_pi_star[0, :]  # Return the first row as the stationary distribution
    
    
    def compute_discounted_visitation(self, P_pi, initial_state_dist):
        N = self.pi.shape[0]
        I = np.eye(N)
        return np.linalg.solve(I - self.gamma * P_pi.T, initial_state_dist)
    
    
    def compute_Q_values(self):
        """Compute action-values for all state-action pairs (with optimality criteria based on the existence of the disc. factor gamma)."""        
        if self.gamma is None:  # Average reward
            V_g = self.compute_gain()
            V_b = self.compute_bias()
            # Q(s,a) = r(s,a) - V_g + Σ_s' P(s'|s,a) V_b(s')
            Q = np.sum(self.P * self.r, axis=2) - V_g + np.sum(self.P * V_b[None, None, :], axis=2)
 
        else:  # Discounted reward
            V = self.compute_full_discounted_value()
            # Q(s,a) = r(s,a) + γ Σ_s' P(s'|s,a) V(s')
            Q = np.sum(self.P * self.r, axis=2) + self.gamma * np.sum(self.P * V[None, None, :], axis=2)
        
        # Mask invalid actions
        Q = np.where(self.env.valid_action_mask, Q, 0.0)
        return Q
        
    def evaluate_a_policy_wrt_average_reward(self, P_pi: np.ndarray, r_pi: np.ndarray, unused=None) -> np.ndarray:
        """
        Given the transition matrix P_pi and the reward vector r_pi,
        compute the average reward (gain) for each state as:
             v_gain = P_pi_star @ r_pi
        In unichain MDPs, average reward (gain) is constant, thus every entry in v_gain is the same. 
        """
        P_pi_star = self.get_pstar_matrix_for_unichain(P_pi)
        v_gain = np.matmul(P_pi_star, r_pi)
        return v_gain
    
    def compute_gain(self):
        """
        Actually same as the method at evaluate_a_policy_wrt_average_reward, except this will return the concrete real value (sample) from one state
        """
        P_pi, r_bar = self.compute_transition_probabilities_and_rewards()
        v_gain = self.evaluate_a_policy_wrt_average_reward(P_pi, r_bar)
        return v_gain[0]
    
    def compute_bias(self):
        """ 
        Bias is equal to advantage function, which measures the advantage value of the policy from the average reward in the long run
        """
        P_pi, r_pi = self.compute_transition_probabilities_and_rewards()
        P_pi_star = self.get_pstar_matrix_for_unichain(P_pi)
        N = self.pi.shape[0]        
        I = np.eye(N)

        try:
            H_pi = -P_pi_star + np.linalg.inv(I - P_pi + P_pi_star)
        except np.linalg.LinAlgError:
            logger.info("Matrix inversion for I - P_pi + P_pi_star failed, using pseudo-inverse")
            H_pi = -P_pi_star + np.linalg.pinv(I - P_pi + P_pi_star)
        V_b = np.matmul(H_pi, r_pi)
        
        # logger.info(f"V_b: {V_b}")
        return V_b


    def compute_gain_and_bias(self):
        """
        Compute the gain V_g and bias vector V_b (relative values) using a reference state.
        We set V_b(s_ref)=0 and solve:
              (I - P_pi) v = r_bar,   with v[s_ref]=0.
        Then, the gain is given by:
              V_g = r_bar[s_ref] + sum(P_pi[s_ref, :] * v).
              
        """
        P_pi, r_bar = self.compute_transition_probabilities_and_rewards()
        
        N = self.nS
        # Get the reference state's index
        s_ref_index = self.state_indices[self.ref_state]
        I = np.eye(N)
        M = I - P_pi

        # Replace the row corresponding to the reference state
        M[s_ref_index, :] = 0
        M[s_ref_index, s_ref_index] = 1
        b = r_bar.copy()
        b[s_ref_index] = 0
        
        # v is our bias vector V_b with v(s_ref)=0
        v = np.linalg.solve(M, b)  
        V_g = r_bar[s_ref_index] + np.dot(P_pi[s_ref_index, :], v)
    
        return V_g, v
    

    def compute_full_discounted_value(self):
        if self.gamma is None:
            raise ValueError("Gamma must be set for discounted reward computation")
        P_pi, r_pi = self.compute_transition_probabilities_and_rewards()
        
        nS = self.pi.shape[0]
        I = np.eye(nS)
        
        # logger.info(f"r_bar: {r_bar}", f"P_pi: {P_pi}")
        return np.linalg.solve(I - self.gamma * P_pi, r_pi)
    
    
    # def compute_discounted_value(self):
    #     """
    #     Compute the discounted value function V_pi at the initial state.
    #     Requires gamma to be set.
    #     Complexity: O(nS^3 + nS * nA * nT)
    #     """
    #     if self.gamma is None:
    #         raise ValueError("Gamma must be set for discounted reward computation")
    #     P_pi, r_bar = self.compute_transition_probabilities_and_rewards()
    #     N = self.env.num_states
    #     I = np.eye(N)
    #     try:
    #         # Solve the linear system for V
    #         # V = (I - γP_pi)^(-1) r_bar
            
    #         # Observation:
    #         # Bigger gamma --> lower r_bar
    #         # Bigger gamma --> More far-sighted --> Bigger chance to choose a00 at s0 
    #         # --> First row of P_pi approaching [0.5 0.5] from [0 1] --> Weight for r_bar[0] is bigger
    #         # But the r_bar[0] itself is getting smaller  
    #         # logger.info("inv I - γP_pi")
    #         # logger.info(np.linalg.inv(I - self.gamma * P_pi))
    #         # logger.info((I - self.gamma * P_pi))
    #         logger.info(f"P_pi: {P_pi}")
    #         logger.info(f"r_pi: {r_bar}")
            
            
    #         V = np.linalg.solve(I - self.gamma * P_pi, r_bar)
    #     except np.linalg.LinAlgError:
    #         logger.info("Singular matrix: using pseudo-inverse to compute V_pi")
    #         V = np.linalg.pinv(I - self.gamma * P_pi) @ r_bar
            
    #     logger.info(f"V: {V}")
        
    #     # logger.info(f"np.sum(initial_state_dist * V) : {V[self.state_indices[self.ref_state]]}")
    #     return V[self.env.initial_state]
    
    
    def compute_discounted_value(self):
        """Compute discounted value at initial state distribution."""
        V = self.compute_full_discounted_value()
        return np.sum(self.env.initial_state_dist * V)

    
    def compute_grad_log_pi(self):
        """Compute ∇_θ log π(a|s; θ)."""
        s_indices = np.arange(self.env.num_states)
        z = s_indices * self.theta[0] + self.theta[1]
        # logger.info(f"z: {z}")
        sigma_z = 1 / (1 + np.exp(-z))
        grad_log_pi = np.zeros((self.env.num_states, 2, 2))
        grad_log_pi[:, 0, 0] = (1 - sigma_z) * s_indices
        grad_log_pi[:, 0, 1] = (1 - sigma_z)
        grad_log_pi[:, 1, 0] = -sigma_z * s_indices
        grad_log_pi[:, 1, 1] = -sigma_z
        return grad_log_pi
    
    def compute_exact_gradient(self, initial_state_dist=None):
        """Compute the exact policy gradient with respect to θ."""
        # Compute Q-values and state visitation frequencies
        Q = self.compute_Q_values()
        if self.gamma is None:
            d = self.compute_stationary_distribution()
        else:
            P_pi, _ = self.compute_transition_probabilities_and_rewards()
            d = self.compute_discounted_visitation(P_pi, initial_state_dist)

        grad_log_pi = self.compute_grad_log_pi()
        
        # Compute gradient: ∑_s d(s) ∑_a π(a|s) ∇_θ log π(a|s) Q(s,a)
        grad = np.sum(d[:, None, None] * self.pi[:, :, None] * grad_log_pi * Q[:, :, None], axis=(0, 1))
        return grad
    
    
    def compute_exact_gradient_gain(self):
        """Compute the exact gain gradient ∇v_g(θ)."""
        # Without Baseline
        Q = self.compute_Q_values()
        
        # With baseline
        V_b = self.compute_bias()
        A = Q - V_b[self.env.initial_state]
        
        d = self.compute_stationary_distribution()
        grad_log_pi = self.compute_grad_log_pi()
        
        # Without baseline
        # grad = np.sum(d[:, None, None] * self.pi[:, :, None] * grad_log_pi * Q[:, :, None], axis=(0, 1))
        
        # With baseline
        grad = np.sum(d[:, None, None] * self.pi[:, :, None] * grad_log_pi * A[:, :, None], axis=(0, 1))
        return grad
        
    def compute_exact_gradient_bias(self, s0):
        """Compute the exact bias gradient ∇v_b(θ, s0) as per Theorem 3.1."""
        P_pi, _ = self.compute_transition_probabilities_and_rewards()
        P_pi_star = self.get_pstar_matrix_for_unichain(P_pi)
        t_mix = self.compute_mixing_time(P_pi, P_pi_star)
        Q_b = self.compute_Q_values()
        
        # With baseline
        V_b = self.compute_bias()
        A = Q_b - V_b[s0]
        
        grad_log_pi = self.compute_grad_log_pi()
        grad_vg = self.compute_exact_gradient_gain()
        d_star = P_pi_star[0, :]  # Stationary distribution

        # Pre-mixing part
        pre_mixing_grad = np.zeros_like(self.theta)
        for t in range(t_mix):
            # Compute P_pi^t
            P_pi_t = np.linalg.matrix_power(P_pi, t)
            # State distribution at time t starting from s0
            p_t = P_pi_t[s0, :]
            
            # Expectation: Σ_s p_π^t(s|s0) Σ_a π(a|s) [ q_b(s,a) ∇log π(a|s) ]
            
            # Without baseline
            # term1 = np.sum(p_t[:, None, None] * self.pi[:, :, None] * grad_log_pi * Q_b[:, :, None], axis=(0, 1))
            
            # With baseline
            term1 = np.sum(p_t[:, None, None] * self.pi[:, :, None] * grad_log_pi * A[:, :, None], axis=(0, 1))
            
            
            pre_mixing_grad += (term1 - grad_vg)

        # Post-mixing part
        # post_mixing_grad = np.sum(d_star[:, None, None] * self.pi[:, :, None] * grad_log_pi * Q_b[:, :, None], axis=(0, 1)) 
        
        # Without baseline
        # post_mixing_grad = np.sum(d_star[:, None, None] * self.pi[:, :, None] * grad_log_pi * Q_b[:, :, None], axis=(0, 1)) + grad_vg # temp

        # With baseline
        post_mixing_grad = np.sum(d_star[:, None, None] * self.pi[:, :, None] * grad_log_pi * A[:, :, None], axis=(0, 1)) + grad_vg
        
        grad_vb = pre_mixing_grad + post_mixing_grad
        return grad_vb
    
    def compute_exact_action_fisher(self, s0):
        """Compute the action Fisher matrix Φ_a(θ, s0)"""
        grad_log_pi = self.compute_grad_log_pi()
        nS = self.env.num_states
        dim_theta = len(self.theta)
        Phi_a = np.zeros((nS, dim_theta, dim_theta))
        for s in range(nS):
            for a in range(self.env.max_num_actions):
                if self.env.valid_action_mask[s, a]:
                    grad = grad_log_pi[s, a, :]
                    Phi_a[s] += self.pi[s, a] * np.outer(grad, grad)
        
        return Phi_a
                
    
    def compute_exact_gain_action_fisher(self):
        """Compute the exact gain-action Fisher matrix Φ_ga"""
        P_pi_star = self.compute_stationary_distribution()
        Phi_a = self.compute_exact_action_fisher(self.env.initial_state)
        return np.sum(P_pi_star[:, None, None] * Phi_a, axis=0)
    
    
    def compute_exact_bias_action_fisher(self, s0=0):
        """Compute the exact bias-action Fisher matrix Φ_ba"""
        P_pi, _ = self.compute_transition_probabilities_and_rewards()
        P_pi_star = self.get_pstar_matrix_for_unichain(P_pi)
        N = self.pi.shape[0]
        I = np.eye(N)
        H_pi = -P_pi_star + np.linalg.inv(I - P_pi + P_pi_star)
        h_pi = H_pi[s0]
        weights = np.abs(h_pi)
        Phi_a = self.compute_exact_action_fisher(self.env.initial_state)
        return np.sum(weights[:, None, None] * Phi_a, axis=0)
    
    def compute_exact_gradient_v_gamma(self):
        P_pi, _ = self.compute_transition_probabilities_and_rewards()
        P_pi_gamma = self.compute_discounted_visitation(P_pi=P_pi, initial_state_dist=self.env.initial_state_dist)
        Q_gamma = self.compute_Q_values()
        
        # With baseline
        V_gamma = self.compute_full_discounted_value()
        # logger.info(f"V_gamma: {V_gamma}")
        A = Q_gamma - V_gamma[self.env.initial_state]
        
        grad_log_pi = self.compute_grad_log_pi()
        grad_pi = self.pi[:, :, None] * grad_log_pi
        
        # Without baseline
        # grad_v_gamma = np.sum(P_pi_gamma[:, None, None] * Q_gamma[:, :, None] * grad_pi, axis=(0,1))
        # With baseline
        grad_v_gamma = np.sum(P_pi_gamma[:, None, None] * A[:, :, None] * grad_pi, axis=(0, 1))
        return grad_v_gamma
     
    def compute_exact_gamma_action_fisher(self):
        P_pi, _ = self.compute_transition_probabilities_and_rewards()
        P_pi_gamma = self.compute_discounted_visitation(P_pi=P_pi, initial_state_dist=self.env.initial_state_dist)
        Phi_a = self.compute_exact_action_fisher(self.env.initial_state)
        Phi_gamma_action = np.sum(P_pi_gamma[:, None, None] * Phi_a, axis=0)
        return Phi_gamma_action
    
    
    def compute_preconditioning_matrix(self, variant="vanilla", beta=1.0, beta_div=10):
        """Compute preconditioning matrix C."""
        if variant == "vanilla":
            return np.eye(len(self.theta))
        elif variant == "natural":
            Phi_ba = self.compute_exact_bias_action_fisher(s0=self.env.initial_state)
            Phi_ga = self.compute_exact_gain_action_fisher()
            grad_g = self.compute_exact_gradient_gain()
            C = Phi_ba + beta * Phi_ga + (beta**2 / beta_div) * np.outer(grad_g, grad_g)
            # Return with regularization for stability
            return C + 1e-6 * np.eye(len(self.theta))  
        else:
            raise ValueError("Variant must be 'vanilla' or 'natural'")
        
    def backtracking_linesearch_ascent(self, grad, fn, niter=100):
        """Backtracking line search with Armijo condition."""
        alpha = 1.0
        kappa = 1e-4
        rho = 0.5
        fval = fn(self.theta)
        # Use gradient as step direction
        dirder = np.dot(grad, grad)  

        # logger.info(f"backtracking_linesearch_ascent")
        # logger.info(f"dirder: {dirder}")

        if dirder <= 0:
            return 0.0
        for _ in range(niter):
            theta_new = self.theta + alpha * grad
            fval_new = fn(theta_new)
            
            # logger.info(f"fn: {fn}")
            # logger.info(f"p: {self.theta}, alpha: {alpha}, stepdir: {grad}")
            # logger.info(f"fval_next: {fval_new}")
            # logger.info(f"alpha: {alpha}")
            # logger.info(not np.isfinite(fval_new) or fval_new < (fval + kappa * alpha * dirder))
            # logger.info(not np.isfinite(fval_new))
            # logger.info(fval_new < (fval + kappa * alpha * dirder))
            # logger.info(f" (fval + kappa * alpha * dirder): {(fval + kappa * alpha * dirder)}")
            if not np.isfinite(fval_new) or fval_new < (fval + kappa * alpha * dirder):
                alpha *= rho
            else:
                return alpha
        return 0.0
    
    def _objective_at(self, p):
        theta_old = self.theta.copy()
        self.theta = p
        self.update_policy()
        if self.gamma is None:                        
            v_g = self.compute_gain()
            
            if not self.is_gain_converged:
                val = v_g
            else:    
                v_b = self.compute_bias()[self.env.initial_state]
                c = self.max_gain - self.delta
                if v_g > c:
                    val = v_b + self.beta * np.log(v_g - c)
                    # logger.info(f"Gain {v_g} is greater than c={c}, returning {val}")
                else:
                    val = -np.inf
                    logger.warning(f"Gain {v_g} is not greater than c={c}, returning -inf")
        else:
            val = self.compute_discounted_value()
            
        self.theta = theta_old
        self.update_policy()
        return val

    # def compute_gain_at(self, theta):
    #     """Compute gain at a given theta."""
    #     original_theta = self.theta.copy()
    #     self.theta = theta
    #     self.update_policy()
    #     gain = self.compute_gain()
    #     self.theta = original_theta
    #     self.update_policy()
    #     return gain
    
    
    # def compute_bias_at(self, theta, s0):
    #     """Compute bias at a given theta for initial state s0."""
    #     original_theta = self.theta.copy()
    #     self.theta = theta
    #     self.update_policy()
    #     bias = self.compute_bias()[s0]
    #     self.theta = original_theta
    #     self.update_policy()
    #     return bias
    
    
    # def numerical_gradient(self, func, theta, h=1e-5):
    #     """Compute numerical gradient of a function with respect to theta."""
    #     grad = np.zeros_like(theta)
    #     for i in range(len(theta)):
    #         theta_plus = theta.copy()
    #         theta_plus[i] += h
    #         theta_minus = theta.copy()
    #         theta_minus[i] -= h
    #         grad[i] = (func(theta_plus) - func(theta_minus)) / (2 * h)
    #     return grad
    
    
    def compute_mixing_time(self, P_pi, P_pi_star, epsilon=1e-6, tol_abs=1e-6, tol_rel=1e-5):
        t = 0
        nS = self.env.num_states
        while True:
            P_pi_t = np.linalg.matrix_power(P_pi, t)
            max_tv_distance = 0
            for s0 in range(nS):
                tv_distance = 0.5 * np.sum(np.abs(P_pi_t[s0, :] - P_pi_star[s0, :]))
                max_tv_distance = max(max_tv_distance, tv_distance)
            if max_tv_distance <= epsilon:
                return t
            t += 1
            
            # Avoid infinite loops
            if t > 1000:  
                logger.info("Mixing time computation exceeded 1000 iterations; using default t_mix = 1000")
                logger.warning("Mixing time computation did not converge; using default t_mix = 1000")
                return 1000
    
    def estimate_gradients_and_fishers_avg(self, q_b, n_exp_episodes=16, t_max_episode=100, t_abs_min=2):
        """Sampling-based estimation of gradients and Fisher matrices"""
        dim_theta = len(self.theta)
        grad_g_list = []
        grad_b_list = []
        fisher_g_list = []
        fisher_b_list = []
        
        # t_max_episode = max(t_max_episode, 1000) 
        # logger.info(f"t_max_episode: {t_max_episode}")
        v_b = self.compute_bias()
        

        for _ in range(n_exp_episodes):
            state = self.env.reset()
            grad_g = np.zeros(dim_theta)
            grad_b = np.zeros(dim_theta)
            Phi_ga = np.zeros((dim_theta, dim_theta))
            Phi_ba = np.zeros((dim_theta, dim_theta))
            trajectory = []

            # Collect trajectory
            for t in range(t_max_episode):
                action = self.select_action(state)
                next_state, reward, _, _ = self.env.step(action)
                trajectory.append((state, action))
                state = next_state

            # Compute gradients and Fishers
            grad_log_pi = self.compute_grad_log_pi()
            for t, (s, a) in enumerate(trajectory):
                advantage = q_b[s, a] - v_b[s]
                z = advantage * grad_log_pi[s, a]
                 
                # z = q_b[s, a] * grad_log_pi[s, a]
                X = np.outer(grad_log_pi[s, a], grad_log_pi[s, a])
                if t == t_max_episode - 1:
                    grad_g = z
                    grad_b += (z - (t_max_episode - 1) * grad_g)
                    Phi_ga =  X
                    Phi_ba += (t_abs_min * X)
                else:
                    grad_b += z
                    if t < t_abs_min:
                        Phi_ba += X

            grad_g_list.append(grad_g)
            grad_b_list.append(grad_b)
            fisher_g_list.append(Phi_ga)
            fisher_b_list.append(Phi_ba)

        return (np.mean(grad_g_list, axis=0), np.mean(grad_b_list, axis=0),
                np.mean(fisher_g_list, axis=0), np.mean(fisher_b_list, axis=0))
           
                
    def discounting_free_polgrad(self, variant='vanilla', use_sampling=False, outer_iterations=1, inner_iterations=100, n_exp_episodes=16):
        
        policy_value = []
        
        current_samples = 0
        self.gain_samples_for_convergence = None
        self.total_samples_for_convergence = None
        
        for i in range(outer_iterations):
            for j in range(inner_iterations):
                # logger.info(f"self tehta: {self.theta}")
                v_g = self.compute_gain()
                q_b = self.compute_Q_values()
                s0 = self.env.initial_state
                if use_sampling:
                    P_pi, _ = self.compute_transition_probabilities_and_rewards()
                    P_pi_star = self.get_pstar_matrix_for_unichain(P_pi)
                    t_max_episode = self.compute_mixing_time(P_pi=P_pi, P_pi_star=P_pi_star) + 5
                    # if variant == 'natural':
                    #     logger.info(f"Mixing time t_mix: {t_max_episode}")
                    t_absorption_min = 2
                    grad_g, grad_b, Phi_ga, Phi_ba = self.estimate_gradients_and_fishers_avg(q_b=q_b, n_exp_episodes=n_exp_episodes, t_max_episode=t_max_episode, t_abs_min=t_absorption_min)
                    

                    current_samples += n_exp_episodes * t_max_episode    
                    # logger.info(f"n_exp_eps: {n_exp_episodes}, t_max_eps: {t_max_episode}")
                    # logger.info(f"current_samples: {current_samples}")
                    
                else:
                    grad_g = self.compute_exact_gradient_gain()
                    grad_b = self.compute_exact_gradient_bias(s0)
                    Phi_ga = self.compute_exact_gain_action_fisher() if variant == 'natural' else None
                    Phi_ba = self.compute_exact_bias_action_fisher(s0) if variant == 'natural' else None
                
                    # For exact methods, 1 iteration is 1 "sample"
                    current_samples += 1 
                    
                    
                if self.is_gain_converged:
                    self.beta_tilda = self.beta / (v_g - self.max_gain + self.delta)
                    grad_v = grad_b + self.beta_tilda * grad_g
                    if variant == 'natural':
                        preconditioning_matrix = Phi_ba + self.beta_tilda * Phi_ga + (self.beta_tilda ** 2 / self.beta) * np.outer(grad_g, grad_g) 
                        preconditioning_matrix += 1e-6 * np.eye(len(self.theta))  # Regularization for stability
                    else:
                        preconditioning_matrix = np.eye(len(self.theta))
                else:
                    grad_v = grad_g
                    if variant == 'natural':
                        preconditioning_matrix = Phi_ga
                        preconditioning_matrix += 1e-6 * np.eye(len(self.theta))  # Regularization for stability
                    else:
                        preconditioning_matrix = np.eye(len(self.theta))
            
            
                grad_v_norm = np.linalg.norm(grad_v)
                
                if not self.is_gain_converged:
                    if grad_v_norm < self.gain_convergence_treshold:
                        self.max_gain = v_g
                        self.is_gain_converged = True
                        self.theta_trajectory_bias.append(self.theta.copy())
                        
                        self.gain_samples_for_convergence = current_samples
                        logger.info(f"Gain converged at iteration {i * inner_iterations + j} with norm {grad_v_norm} and {self.gain_samples_for_convergence} samples")

                        continue
                else:
                    if grad_v_norm < self.convergence_treshold:
                        self.total_samples_for_convergence = current_samples
                        logger.info(f"Bias converged at iteration {i * inner_iterations + j} with norm {grad_v_norm} and {self.total_samples_for_convergence} samples")
                        break
                
                try:
                    delta_theta = np.linalg.solve(preconditioning_matrix, grad_v)
                except np.linalg.LinAlgError:
                    logger.warning("Preconditioning matrix singular, using pseudo-inverse")
                    delta_theta = np.linalg.pinv(preconditioning_matrix) @ grad_v
                        
                alpha = self.backtracking_linesearch_ascent(grad_v, self._objective_at)
                self.theta += alpha * delta_theta.reshape(self.theta.shape)
                self.update_policy()
                
                if not self.is_gain_converged:
                    self.theta_trajectory_gain.append(self.theta.copy())
                    # logger.info(f"Gain trajectory updated at iteration {i * inner_iterations + j}")
                else:
                    self.theta_trajectory_bias.append(self.theta.copy())
                    # logger.info(f"theta: {self.theta.copy()}")
                self.theta_trajectory.append(self.theta.copy())
                # logger.info(f"Iter {i * inner_iterations + j}: v_g={v_g:.4f}, grad_v_norm={grad_v_norm:.4e}, alpha={alpha:.4e}")
                
                v_g = self.compute_gain()
                v_b = self.compute_bias()
                policy_value.append(v_g + v_b[self.env.initial_state])
                
                                
            if self.is_gain_converged:
                self.beta /= self.beta_div
                logger.info(f"Reduced beta to {self.beta}")
                
                
            if self.total_samples_for_convergence is not None:
                logger.info(f"Total samples for convergence: {self.total_samples_for_convergence}")
                logger.info(f"Gain samples for convergence: {self.gain_samples_for_convergence}")
                logger.info(f"Final theta: {self.theta}")
                logger.info(f"Final policy value: {policy_value[-1]}")
                break
            
        if self.total_samples_for_convergence is None:
            logger.warning(f"DISCOUNTING-FREE method did not converge within {inner_iterations} iterations.")

        return policy_value, self.gain_samples_for_convergence, self.total_samples_for_convergence
                
                
    def estimate_gradients_and_fishers_discounted(self, q_gamma, n_exp_episodes=10, t_max_episode=None, sampling_method='proper'):
        dim_theta = len(self.theta)
        grad_gamma_list = []
        fisher_gamma_list = []
        v_gamma = self.compute_full_discounted_value()

        total_samples_in_batch = 0

        for exp_episode in range(n_exp_episodes):
            state = self.env.reset()
            grad_gamma = np.zeros(dim_theta)
            Phi_gamma_action = np.zeros((dim_theta, dim_theta))
            trajectory = []

            # Determine episode length
            if sampling_method == 'proper':
                t_max = self.rng.geometric(1 - self.gamma)
                # Cap to avoid infinite episodes
                t_max = min(t_max, 1000)
                # logger.info(f"Episode {exp_episode + 1}: geom, t_max = {t_max}")
            else:  # 'popular'
                t_max = t_max_episode
                # logger.info(f"Episode {exp_episode + 1}: popular, t_max = {t_max}")
                
                
            # Run episode and collect trajectory
            for t in range(t_max):
                action = self.select_action(state)
                next_state, reward, _, _ = self.env.step(action)
                trajectory.append((state, action, t))
                state = next_state
            
            total_samples_in_batch += len(trajectory)
            
            grad_log_pi = self.compute_grad_log_pi()
            if sampling_method == 'proper':
                # Use only the last time step
                if trajectory:  
                    s, a, t = trajectory[-1]
                    advantage = q_gamma[s, a] - v_gamma[s]
                    z = advantage * grad_log_pi[s, a]

                    grad_gamma = z
                    
                    Phi_gamma_action = np.outer(grad_log_pi[s, a], grad_log_pi[s, a])
            else:  # 'popular'
                # Use all time steps
                for s, a, t in trajectory:
                    # Without baseline
                    # z = q_gamma[s, a] * grad_log_pi[s, a]
                    
                    # With baseline
                    advantage = q_gamma[s, a] - v_gamma[s]
                    
    
                    # Without scaling factor
                    z = advantage * grad_log_pi[s, a]
                    
                    # With scaling factor
                    # discount_factor_t = self.gamma**t
                    # scaling_factor = (1 - self.gamma) * discount_factor_t 
                    # z = scaling_factor * advantage * grad_log_pi[s, a]
                    
                    grad_gamma += z
                    
                    # Without scaling factor
                    Phi_gamma_action +=  np.outer(grad_log_pi[s, a], grad_log_pi[s, a])
                    
                    # With scaling factor
                    # Phi_gamma_action += scaling_factor * np.outer(grad_log_pi[s, a], grad_log_pi[s, a])

                # Average over time steps
                # grad_gamma /= t_max
                # Phi_gamma_action /= t_max

            grad_gamma_list.append(grad_gamma)
            fisher_gamma_list.append(Phi_gamma_action)

        
        # Average over all episodes (axis=0 coresponds to episode)
        sample_mean_grad_gamma = np.mean(grad_gamma_list, axis=0)
        sample_mean_fisher_gamma = np.mean(fisher_gamma_list, axis=0)
        
        return sample_mean_grad_gamma, sample_mean_fisher_gamma, total_samples_in_batch
    

     
    def discounted_polgrad(self, variant='vanilla', use_sampling=False, iterations=100, sampling_method='proper', num_samples_per_update=100, sampling_n_exp_episodes=10, sampling_t_max=10, eval_interval=10, eval_n_exp_episodes=10, eval_t_max=10):
        if self.gamma is None:
            raise ValueError("Gamma must be set for discounted policy gradient")
        
        evaluation_history = []
        cumulative_discounted_rewards_exact = [] 
        
        
        current_samples = 0
        self.total_samples_for_convergence = None
        
        
        for i in range(iterations):
            q_gamma = self.compute_Q_values()
            
            samples_this_iter = 0
            
            if use_sampling:
                # if sampling_method == 'proper':
                #     n_exp_episodes = num_samples_per_update  # 1 sample per episode
                # else:  # 'popular'
                #     n_exp_episodes = sampling_n_exp_episodes  # Integer division, at least 1
                # logger.info(f"Sampling method: {sampling_method}, n_exp_episodes: {n_exp_episodes}, t_max_episode: {sampling_t_max if sampling_method == 'popular' else None}")
                grad_v_gamma, Phi_gamma_action, samples_this_iter = self.estimate_gradients_and_fishers_discounted(q_gamma=q_gamma, n_exp_episodes=sampling_n_exp_episodes, t_max_episode=sampling_t_max if sampling_method == 'popular' else None, sampling_method=sampling_method)
              
            else:
                grad_v_gamma = self.compute_exact_gradient_v_gamma()
                Phi_gamma_action = self.compute_exact_gamma_action_fisher() if variant == 'natural' else None
                samples_this_iter = 1  # 1 iteration is 1 "sample"
                
            current_samples += samples_this_iter

            grad_norm = np.linalg.norm(grad_v_gamma)
            # logger.info(f"Iter {i}: v_gamma={v_gamma:.4f}, grad_norm={grad_norm:.4e}, samples_so_far={current_samples}")

            if grad_norm < self.convergence_treshold:
                logger.info(f"DISCOUNTED method converged at iteration {i} with {current_samples} samples.")
                self.total_samples_for_convergence = current_samples
                return evaluation_history, cumulative_discounted_rewards_exact, self.total_samples_for_convergence
                            
            if variant == 'natural':
                preconditioning_matrix = Phi_gamma_action + 1e-6 * np.eye(len(self.theta)) # Regularization
            else:
                preconditioning_matrix = np.eye(len(self.theta))
                
                
            try:
                delta_theta = np.linalg.solve(preconditioning_matrix, grad_v_gamma)
            except np.linalg.LinAlgError:
                logger.warning("Preconditioning matrix singular, using pseudo-inverse")
                delta_theta = np.linalg.pinv(preconditioning_matrix) @ grad_v_gamma
                    


            alpha = self.backtracking_linesearch_ascent(grad_v_gamma, self._objective_at)
            self.theta += alpha * delta_theta
            self.update_policy()
            self.theta_trajectory.append(self.theta.copy())
            
            
            if (i + 1) % eval_interval == 0 or (i + 1) == 1:
                avg_reward_eval = self.estimate_average_reward(
                    num_episodes=eval_n_exp_episodes, 
                    max_steps_per_episode=eval_t_max
                )
                # avg_reward_eval = self.compute_gain()
                evaluation_history.append(((i + 1), avg_reward_eval))
                logger.info(f"Iteration {i + 1}/{iterations}, Gamma: {self.gamma}, Method: {variant} {'Sampling ('+sampling_method+')' if use_sampling else 'Exact'}. Avg Eval Reward: {avg_reward_eval:.4f}")

            # For exact eval purposes
            v_gamma = self.compute_discounted_value()
            cumulative_discounted_rewards_exact.append(((i + 1), v_gamma))
            

          
        logger.warning(f"DISCOUNTED method did not converge within {iterations} iterations.")

        return evaluation_history, cumulative_discounted_rewards_exact, None
    
            
    def estimate_average_reward(self, num_episodes=10, max_steps_per_episode=100):
        """Estimate the average reward of the current policy."""
        total_reward = 0.0
        total_steps = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            total_reward += episode_reward
            total_steps += max_steps_per_episode
        return total_reward / total_steps