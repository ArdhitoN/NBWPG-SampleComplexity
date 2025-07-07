import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import pandas as pd

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

class Visualizer:
    def __init__(self, env_name, agent, gamma=None, seed=None, output_dir='./figs/exact_policy_gradient/'):
        self.env_name = env_name
        self.agent = agent
        self.gamma = gamma
        self.seed = seed
        self.output_dir = output_dir

    def visualize_gain_landscape(self, theta0_range, theta1_range):
        Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
        Gain = np.zeros_like(Theta0)
        theta_old = self.agent.theta.copy()
        for i in range(Theta0.shape[0]):
            for j in range(Theta0.shape[1]):
                new_theta = theta_old.copy()
                new_theta[0] = Theta0[i, j]
                new_theta[1] = Theta1[i, j]
                self.agent.theta = new_theta
                self.agent.update_policy()
                Gain[i, j] = np.round(self.agent.compute_gain(), 5)

        self.agent.theta = theta_old
        self.agent.update_policy()

        os.makedirs(self.output_dir, exist_ok=True)

        plt.contourf(Theta0, Theta1, Gain, levels=50, cmap="viridis")
        plt.xlabel(r"$\theta_0$")
        plt.ylabel(r"$\theta_1$")
        # plt.title(f"Gain Landscape")
        # plt.suptitle(f"Environment: {self.env_name}")
        
        plt.colorbar(label=r"$Gain\ (v_g)$")
        plt.savefig(os.path.join(self.output_dir, f"gain_landscape_gamma_{self.gamma}.png"))
        plt.close()

        # 4) Flatten grid → save CSV
        #    Create a DataFrame with columns: theta0, theta1, gain
        n_rows, n_cols = Theta0.shape
        flat_records = []
        for i in range(n_rows):
            for j in range(n_cols):
                flat_records.append({
                    'theta0': Theta0[i, j],
                    'theta1': Theta1[i, j],
                    'gain': Gain[i, j]
                })
        df_gain = pd.DataFrame(flat_records)

        filename_csv = f"gain_landscape_gamma_{self.gamma}.csv"
        csv_path = os.path.join(self.output_dir, filename_csv)
        df_gain.to_csv(csv_path, index=False)
        logger.info(f"Saved Gain‐landscape CSV: {csv_path}")
        
        
    def visualize_bias_landscape(self, theta0_range, theta1_range):
        bias_values = []
        theta0_vals = []
        theta1_vals = []
        theta_old = self.agent.theta.copy()
        
        flat_records = []
        for theta1 in theta1_range:
            for theta0 in theta0_range:
                new_theta = theta_old.copy()
                new_theta[0] = theta0
                new_theta[1] = theta1
                self.agent.theta = new_theta
                self.agent.update_policy()
                b_val = np.round(self.agent.compute_bias()[0], 5)
                bias_values.append(b_val)
                theta0_vals.append(theta0)
                theta1_vals.append(theta1)
                flat_records.append({
                    'theta0': theta0,
                    'theta1': theta1,
                    'bias': b_val
                })


        self.agent.theta = theta_old
        self.agent.update_policy()

        n0 = len(theta0_range)
        n1 = len(theta1_range)
        Theta0 = np.array(theta0_vals).reshape(n1, n0)
        Theta1 = np.array(theta1_vals).reshape(n1, n0)
        Bias = np.array(bias_values).reshape(n1, n0)

        os.makedirs(self.output_dir, exist_ok=True)

        plt.contourf(Theta0, Theta1, Bias, levels=50, cmap="viridis")
        plt.xlabel(r"$\theta_0$")
        plt.ylabel(r"$\theta_1$")
        # plt.title(f"Bias Landscape")
        # plt.suptitle(f"Environment: {self.env_name}")
        plt.colorbar(label=r"$Bias\ v_b(s_0)$")
        plt.savefig(os.path.join(self.output_dir, f"bias_landscape_{self.env_name}_gamma_{self.gamma}.png"))
        plt.close()
        
        df_bias = pd.DataFrame(flat_records)
        filename_csv = f"bias_landscape_gamma_{self.gamma}.csv"
        csv_path = os.path.join(self.output_dir, filename_csv)
        df_bias.to_csv(csv_path, index=False)
        logger.info(f"Saved Bias‐landscape CSV: {csv_path}")


    def visualize_discounted_value_landscape(self, theta0_range, theta1_range, initial_state_dist):
        if self.gamma is None:
            raise ValueError("This method is for discounted reward only (gamma must be set)")
        
        Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
        Value = np.zeros_like(Theta0)
        theta_old = self.agent.theta.copy()
        
        flat_records = []
        for i in range(Theta0.shape[0]):
            for j in range(Theta0.shape[1]):
                new_theta = theta_old.copy()
                new_theta[0] = Theta0[i, j]
                new_theta[1] = Theta1[i, j]
                self.agent.theta = new_theta
                self.agent.update_policy()
                v_val = self.agent.compute_discounted_value()
                Value[i, j] = v_val
                flat_records.append({
                    'theta0': Theta0[i, j],
                    'theta1': Theta1[i, j],
                    'value': v_val
                })

        self.agent.theta = theta_old
        self.agent.update_policy()

        os.makedirs(self.output_dir, exist_ok=True)

        gamma_str = f"{self.gamma:.2f}"
        plt.contourf(Theta0, Theta1, Value, levels=50, cmap="viridis")
        plt.xlabel(r"$\theta_0$")
        plt.ylabel(r"$\theta_1$")
        # plt.title(f"Discounted Value Landscape")
        # plt.suptitle(f"Environment: {self.env_name}, Discount Factor (γ): {gamma_str}")
        plt.colorbar(label=r"$Discounted Value\ (v_\gamma(s_0))$")
        plt.savefig(os.path.join(self.output_dir, f"discounted_value_landscape_gamma_{gamma_str}.png"))
        plt.close()
        
        df_val = pd.DataFrame(flat_records)
        filename_csv = f"discounted_value_landscape_gamma_{gamma_str}.csv"
        csv_path = os.path.join(self.output_dir, filename_csv)
        df_val.to_csv(csv_path, index=False)
        logger.info(f"Saved Discounted‐Value‐landscape CSV: {csv_path}")        
        
    def visualize_gain_progression(self, method_name):
        """Visualize gain trajectory over iterations."""
        theta_trajectory = np.array(self.agent.theta_trajectory)
        gain_history = []
        theta_old = self.agent.theta.copy()

        records = []
        for iteration, theta in enumerate(theta_trajectory):
            self.agent.theta = theta
            self.agent.update_policy()
            gain = self.agent.compute_gain()
            gain_history.append(gain)
            records.append({
                'iteration': iteration,
                'theta0': theta[0],
                'theta1': theta[1],
                'gain': gain
            })


        self.agent.theta = theta_old
        self.agent.update_policy()

        os.makedirs(self.output_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(gain_history, label='Gain', color='blue')
        plt.xlabel("Iteration")
        plt.ylabel(r"$Gain\ (v_g)$")
        # plt.title(f"Gain Trajectory ({method_name})")
        # plt.suptitle(f"Environment: {self.env_name}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"gain_progression_{method_name}_gamma_{self.gamma}.png"))
        plt.close()
        
        df_prog = pd.DataFrame(records)
        filename_csv = f"gain_progression_{method_name}_gamma_{self.gamma}.csv"
        csv_path = os.path.join(self.output_dir, filename_csv)
        df_prog.to_csv(csv_path, index=False)
        logger.info(f"Saved Gain‐progression CSV: {csv_path}")

        
        
    def visualize_gain_plus_bias_progression(self, method_name):
        """Visualize gain trajectory over iterations."""
        theta_trajectory = np.array(self.agent.theta_trajectory)
        policy_value_history = []
        theta_old = self.agent.theta.copy()
        
        records = []

        for iteration, theta in enumerate(theta_trajectory):
            self.agent.theta = theta
            self.agent.update_policy()
            v_g = self.agent.compute_gain()
            v_b = self.agent.compute_bias()
            policy_value = v_g + v_b[0]
            policy_value_history.append(policy_value)
            
            records.append({
                'iteration': iteration,
                'theta0': theta[0],
                'theta1': theta[1],
                'gain_plus_bias': policy_value
            })


        self.agent.theta = theta_old
        self.agent.update_policy()

        os.makedirs(self.output_dir, exist_ok=True)

        # 1) Plot
        tlist = [r['iteration'] for r in records]
        valuelist = [r['gain_plus_bias'] for r in records]
        plt.figure(figsize=(10, 6))
        plt.plot(tlist, valuelist, label=r"$v_g + v_b(s_0)$", color='blue')
        plt.xlabel("Iteration")
        plt.ylabel(r"$v_g + v_b(s_0)$")
        # plt.title(f"Policy Value Progression ({method_name})")
        # plt.suptitle(f"Environment: {self.env_name}")
        plt.grid(True)
        plt.legend()
        filename_png = f"gain_plus_bias_progression_{method_name}_gamma_{self.gamma}.png"
        plt.savefig(os.path.join(self.output_dir, filename_png))
        plt.close()

        # 2) Save CSV
        df_gb = pd.DataFrame(records)
        filename_csv = f"gain_plus_bias_progression_{method_name}_gamma_{self.gamma}.csv"
        csv_path = os.path.join(self.output_dir, filename_csv)
        df_gb.to_csv(csv_path, index=False)
        logger.info(f"Saved Gain+Bias‐progression CSV: {csv_path}")


        
        
        
    def visualize_discounted_value_progression(self, method_name):
        """Visualize gain trajectory over iterations."""
        theta_trajectory = np.array(self.agent.theta_trajectory)
        policy_value_history = []
        theta_old = self.agent.theta.copy()

        records  = []
        for iteration, theta in enumerate(theta_trajectory):
            self.agent.theta = theta
            self.agent.update_policy()
            v_gamma = self.agent.compute_discounted_value()
            policy_value_history.append(v_gamma)
            records.append({
                'iteration': iteration,
                'theta0': theta[0],
                'theta1': theta[1],
                'v_gamma': v_gamma
            })


        self.agent.theta = theta_old
        self.agent.update_policy()

        os.makedirs(self.output_dir, exist_ok=True)

         # 1) Plot
        tlist = [r['iteration'] for r in records]
        vallist = [r['v_gamma'] for r in records]
        plt.figure(figsize=(10, 6))
        plt.plot(tlist, vallist, label=r"$v_\gamma(s_0)$", color='blue')
        plt.xlabel("Iteration")
        plt.ylabel(r"$v_\gamma(s_0)$")
        # plt.title(f"Discounted Value Progression ({method_name})\nEnv: {self.env_name}, γ={self.gamma:.2f}")
        plt.grid(True)
        plt.legend()
        filename_png = f"v_gamma_progression_{method_name}_gamma_{self.gamma:.2f}.png"
        plt.savefig(os.path.join(self.output_dir, filename_png))
        plt.close()

        # 2) Save CSV
        df_v = pd.DataFrame(records)
        filename_csv = f"v_gamma_progression_{method_name}_gamma_{self.gamma:.2f}.csv"
        csv_path = os.path.join(self.output_dir, filename_csv)
        df_v.to_csv(csv_path, index=False)
        logger.info(f"Saved Discounted‐value‐progression CSV: {csv_path}")
        
    def visualize_gain_trajectory(self, theta0_range, theta1_range, method_name):
        """Visualize bias landscape with optimization trajectory, only if optimizing w.r.t. bias."""
        Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
        Gain = np.zeros_like(Theta0)
        theta_old = self.agent.theta.copy()
        
        grid_records = []
        for i in range(Theta0.shape[0]):
            for j in range(Theta0.shape[1]):
                new_theta = theta_old.copy()
                new_theta[0] = Theta0[i, j]
                new_theta[1] = Theta1[i, j]
                self.agent.theta = new_theta
                self.agent.update_policy()
                val = np.round(self.agent.compute_gain(), 5)
                Gain[i, j] = val
                grid_records.append({
                    'theta0': Theta0[i, j],
                    'theta1': Theta1[i, j],
                    'gain': val
                })
        
        # 2) Record the optimization trajectory (one row per step)
        theta_trajectory = np.array(self.agent.theta_trajectory_gain)  # shape=(T,2)
        traj_records = []
        for iteration, theta in enumerate(theta_trajectory):
            self.agent.theta = theta
            self.agent.update_policy()
            g_val = self.agent.compute_gain()
            traj_records.append({
                'iteration': iteration,
                'theta0': theta[0],
                'theta1': theta[1],
                'gain': g_val
            })

        # Restore original theta
        self.agent.theta = theta_old
        self.agent.update_policy()
        
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Determine axes limits based on theta ranges
        x_min, x_max = theta0_range[0], theta0_range[-1]
        y_min, y_max = theta1_range[0], theta1_range[-1]        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Define colors and styles
        color_map = 'viridis'
        color_trajectory_line = "r"
        color_trajectory_points_facecolors = "r"  
        color_trajectory_points_edgecolors = 'k'
        color_start_point = "b"
        color_end_point = "g"
        
        true_end = theta_trajectory[-1]
        end_x = np.clip(theta_trajectory[-1, 0], x_min, x_max)
        end_y = np.clip(theta_trajectory[-1, 1], y_min, y_max)
        if not np.allclose(true_end, [end_x, end_y]):
            end_label = "End (clamped)"
        else:
            end_label = "End"
            
            
        # Plotting
        if end_label.endswith("(clamped)"):
            # direction vector from clamped to true
            dx_true = true_end[0] - end_x
            dy_true = true_end[1] - end_y

            # decide primary direction: if it went out more horizontally, push horizontally, else vertically
            if abs(dx_true) > abs(dy_true):
                # pushed horizontally
                dx_off = np.sign(dx_true) * 0.10 * (x_max - x_min)
                dy_off = 0
            else:
                # pushed vertically
                dx_off = 0
                dy_off = np.sign(dy_true) * 0.10 * (y_max - y_min)

            # if it’s clamped on right edge, force the text to the left
            if end_x >= x_max:
                dx_off = -abs(dx_off or 0.05*(x_max-x_min))
            # if clamped on left edge, force to the right
            elif end_x <= x_min:
                dx_off = abs(dx_off or 0.05*(x_max-x_min))
            # similarly for top/bottom
            if end_y >= y_max:
                dy_off = -abs(dy_off or 0.05*(y_max-y_min))
            elif end_y <= y_min:
                dy_off = abs(dy_off or 0.05*(y_max-y_min))


            plt.annotate(
                "clamped",
                xy=(end_x, end_y),
                xytext=(end_x + dx_off, end_y + dy_off),
                arrowprops=dict(
                    arrowstyle="->",
                    shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0.2"  # smooth curved arrow
                ),
                fontsize="small",
                color=color_end_point,
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
            )

        plt.contourf(Theta0, Theta1, Gain, levels=50, cmap=color_map)
        plt.scatter(theta_trajectory[:, 0], theta_trajectory[:, 1], s=10, label='Trajectory Steps',
                    facecolors=color_trajectory_points_facecolors, edgecolors=color_trajectory_points_edgecolors, clip_on=True)
        plt.plot(theta_trajectory[:, 0], theta_trajectory[:, 1], linestyle='--', alpha=0.5, linewidth=1, color=color_trajectory_line, clip_on=True)
        plt.plot(theta_trajectory[0, 0], theta_trajectory[0, 1], marker='o', label='Start', markersize=10, color=color_start_point, clip_on=True)
        plt.plot(end_x, end_y, marker='o', label=end_label, markersize=10, color=color_end_point, clip_on=True)
        plt.xlabel(r"$\theta_0$")
        plt.ylabel(r"$\theta_1$")
        # plt.title(f"Gain Landscape with Optimization Trajectory ({method_name})")
        # plt.suptitle(f"Environment: {self.env_name}")
        plt.colorbar(label=r"$Gain\ (v_g)$")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"gain_trajectory_{method_name}.png"))
        plt.close()
        
        
        # 4) Save the grid CSV
        df_grid = pd.DataFrame(grid_records)
        filename_csv_grid = f"gain_landscape_for_trajectory_{method_name}.csv"
        csv_path_grid = os.path.join(self.output_dir, filename_csv_grid)
        df_grid.to_csv(csv_path_grid, index=False)
        logger.info(f"Saved Gain‐landscape CSV (for trajectory): {csv_path_grid}")

        # 5) Save the trajectory CSV
        df_traj = pd.DataFrame(traj_records)
        filename_csv_traj = f"gain_trajectory_data_{method_name}.csv"
        csv_path_traj = os.path.join(self.output_dir, filename_csv_traj)
        df_traj.to_csv(csv_path_traj, index=False)
        logger.info(f"Saved Gain‐trajectory CSV: {csv_path_traj}")


    def visualize_bias_trajectory(self, theta0_range, theta1_range, method_name):
        """Visualize bias landscape with optimization trajectory, only if optimizing w.r.t. bias."""
        Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
        Bias = np.zeros_like(Theta0)
        theta_old = self.agent.theta.copy()
        
        grid_records = []
        for i in range(Theta0.shape[0]):
            for j in range(Theta0.shape[1]):
                new_theta = theta_old.copy()
                new_theta[0] = Theta0[i, j]
                new_theta[1] = Theta1[i, j]
                self.agent.theta = new_theta
                self.agent.update_policy()
                
                b_val = np.round(self.agent.compute_bias()[0], 5)
                Bias[i, j] = b_val
                grid_records.append({
                    'theta0': Theta0[i, j],
                    'theta1': Theta1[i, j],
                    'bias': b_val
                })
                
        theta_trajectory = np.array(self.agent.theta_trajectory_bias)
        traj_records = []
        for iteration, theta in enumerate(theta_trajectory):
            self.agent.theta = theta
            self.agent.update_policy()
            b_val = self.agent.compute_bias()[0]
            traj_records.append({
                'iteration': iteration,
                'theta0': theta[0],
                'theta1': theta[1],
                'bias': b_val
            })

        self.agent.theta = theta_old
        self.agent.update_policy()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Determine axes limits based on theta ranges
        x_min, x_max = theta0_range[0], theta0_range[-1]
        y_min, y_max = theta1_range[0], theta1_range[-1]        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        
        # Define colors and styles
        color_map = 'viridis'
        color_trajectory_line = "r"
        color_trajectory_points_facecolors = "r"  
        color_trajectory_points_edgecolors = 'k'
        color_start_point = "b"
        color_end_point = "g"
        
        true_end = theta_trajectory[-1]
        end_x = np.clip(theta_trajectory[-1, 0], x_min, x_max)
        end_y = np.clip(theta_trajectory[-1, 1], y_min, y_max)
        if not np.allclose(true_end, [end_x, end_y]):
            end_label = "End (clamped)"
        else:
            end_label = "End"
            
            
        # Plotting
        if end_label.endswith("(clamped)"):
            # direction vector from clamped to true
            dx_true = true_end[0] - end_x
            dy_true = true_end[1] - end_y

            # decide primary direction: if it went out more horizontally, push horizontally, else vertically
            if abs(dx_true) > abs(dy_true):
                # pushed horizontally
                dx_off = np.sign(dx_true) * 0.10 * (x_max - x_min)
                dy_off = 0
            else:
                # pushed vertically
                dx_off = 0
                dy_off = np.sign(dy_true) * 0.10 * (y_max - y_min)

            # if it’s clamped on right edge, force the text to the left
            if end_x >= x_max:
                dx_off = -abs(dx_off or 0.05*(x_max-x_min))
            # if clamped on left edge, force to the right
            elif end_x <= x_min:
                dx_off = abs(dx_off or 0.05*(x_max-x_min))
            # similarly for top/bottom
            if end_y >= y_max:
                dy_off = -abs(dy_off or 0.05*(y_max-y_min))
            elif end_y <= y_min:
                dy_off = abs(dy_off or 0.05*(y_max-y_min))


            plt.annotate(
                "clamped",
                xy=(end_x, end_y),
                xytext=(end_x + dx_off, end_y + dy_off),
                arrowprops=dict(
                    arrowstyle="->",
                    shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0.2"  # smooth curved arrow
                ),
                fontsize="small",
                color=color_end_point,
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
            )

        plt.contourf(Theta0, Theta1, Bias, levels=50, cmap=color_map)
        plt.scatter(theta_trajectory[:, 0], theta_trajectory[:, 1], s=10, label='Trajectory Steps',
                    facecolors=color_trajectory_points_facecolors, edgecolors=color_trajectory_points_edgecolors, clip_on=True)
        plt.plot(theta_trajectory[:, 0], theta_trajectory[:, 1], linestyle='--', alpha=0.5, linewidth=1, color=color_trajectory_line, clip_on=True)
        plt.plot(theta_trajectory[0, 0], theta_trajectory[0, 1], marker='o', label='Start', markersize=10, color=color_start_point, clip_on=True)
        plt.plot(end_x, end_y, marker='o', label=end_label, markersize=10, color=color_end_point, clip_on=True)
        plt.xlabel(r"$\theta_0$")
        plt.ylabel(r"$\theta_1$")
        # plt.title(f"Bias Landscape with Optimization Trajectory ({method_name})")
        # plt.suptitle(f"Environment: {self.env_name}")
        plt.colorbar(label=r"$Bias\ (v_b(s_0))$")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"bias_trajectory_{method_name}.png"))
        plt.close()

    def visualize_discounted_value_trajectory(self, theta0_range, theta1_range, method_name, initial_state_dist):
        if self.gamma is None:
            raise ValueError("This method is for discounted reward only (gamma must be set)")
        Theta0, Theta1 = np.meshgrid(theta0_range, theta1_range)
        Value = np.zeros_like(Theta0)
        theta_old = self.agent.theta.copy()
        
        grid_records = []
        for i in range(Theta0.shape[0]):
            for j in range(Theta0.shape[1]):
                new_theta = theta_old.copy()
                new_theta[0] = Theta0[i, j]
                new_theta[1] = Theta1[i, j]
                self.agent.theta = new_theta
                self.agent.update_policy()
                v_val = self.agent.compute_discounted_value()
                Value[i, j] = v_val
                grid_records.append({
                    'theta0': Theta0[i, j],
                    'theta1': Theta1[i, j],
                    'v_gamma': v_val
                })

        # 2) Record trajectory
        theta_trajectory = np.array(self.agent.theta_trajectory)
        traj_records = []
        for iteration, theta in enumerate(theta_trajectory):
            self.agent.theta = theta
            self.agent.update_policy()
            v_val = self.agent.compute_discounted_value()
            traj_records.append({
                'iteration': iteration,
                'theta0': theta[0],
                'theta1': theta[1],
                'v_gamma': v_val
            })

        self.agent.theta = theta_old
        self.agent.update_policy()

        # logger.info(f"Discounted Value Trajectory Shape for {method_name}: {theta_trajectory.shape}")

        os.makedirs(self.output_dir, exist_ok=True)

        gamma_str = f"{self.gamma:.2f}"
        
        # Determine axes limits based on theta ranges
        x_min, x_max = theta0_range[0], theta0_range[-1]
        y_min, y_max = theta1_range[0], theta1_range[-1]        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Define colors and styles
        color_map = 'viridis'
        # color_map = 'plasma'
        color_trajectory_line = "r"
        color_trajectory_points_facecolors = "r"  
        color_trajectory_points_edgecolors = 'k'
        color_start_point = "b"
        color_end_point = "g"
        
        true_end = theta_trajectory[-1]
        end_x = np.clip(theta_trajectory[-1, 0], x_min, x_max)
        end_y = np.clip(theta_trajectory[-1, 1], y_min, y_max)
        if not np.allclose(true_end, [end_x, end_y]):
            end_label = "End (clamped)"
        else:
            end_label = "End"
            
            
        # Plotting
        if end_label.endswith("(clamped)"):
            # direction vector from clamped to true
            dx_true = true_end[0] - end_x
            dy_true = true_end[1] - end_y

            # decide primary direction: if it went out more horizontally, push horizontally, else vertically
            if abs(dx_true) > abs(dy_true):
                # pushed horizontally
                dx_off = np.sign(dx_true) * 0.10 * (x_max - x_min)
                dy_off = 0
            else:
                # pushed vertically
                dx_off = 0
                dy_off = np.sign(dy_true) * 0.10 * (y_max - y_min)

            # if it’s clamped on right edge, force the text to the left
            if end_x >= x_max:
                dx_off = -abs(dx_off or 0.05*(x_max-x_min))
            # if clamped on left edge, force to the right
            elif end_x <= x_min:
                dx_off = abs(dx_off or 0.05*(x_max-x_min))
            # similarly for top/bottom
            if end_y >= y_max:
                dy_off = -abs(dy_off or 0.05*(y_max-y_min))
            elif end_y <= y_min:
                dy_off = abs(dy_off or 0.05*(y_max-y_min))


            plt.annotate(
                "clamped",
                xy=(end_x, end_y),
                xytext=(end_x + dx_off, end_y + dy_off),
                arrowprops=dict(
                    arrowstyle="->",
                    shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0.2"  # smooth curved arrow
                ),
                fontsize="small",
                color=color_end_point,
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
            )

        
        plt.contourf(Theta0, Theta1, Value, levels=50, cmap=color_map)
        plt.scatter(theta_trajectory[:, 0], theta_trajectory[:, 1], s=10, label='Trajectory Steps',
                    facecolors=color_trajectory_points_facecolors, edgecolors=color_trajectory_points_edgecolors, clip_on=True)
        plt.plot(theta_trajectory[:, 0], theta_trajectory[:, 1], linestyle='--', alpha=0.5, linewidth=1, color=color_trajectory_line, clip_on=True)
        plt.plot(theta_trajectory[0, 0], theta_trajectory[0, 1], marker='o', label='Start', markersize=10, color=color_start_point, clip_on=True)
        plt.plot(end_x, end_y, marker='o', label=end_label, markersize=10, color=color_end_point, clip_on=True)
        plt.xlabel(r"$\theta_0$")
        plt.ylabel(r"$\theta_1$")
        # plt.title(f"Discounted Value Landscape with Optimization Trajectory ({method_name})")
        # plt.suptitle(f"Environment: {self.env_name}, Discount Factor (γ): {gamma_str}")
        attributes = method_name.split('_')
        logger.info(attributes)
        if len(attributes) == 3:
            pg_type, sampling_or_exact, sampling_type = attributes[0], attributes[1], attributes[2]
            title = f"{pg_type} | {sampling_or_exact} ({sampling_type}) | γ = {gamma_str}"
        elif len(attributes) == 2:
            pg_type, sampling_or_exact = attributes[0], attributes[1]
            title = f"{pg_type} | {sampling_or_exact} | γ = {gamma_str}"
        else:
            raise ValueError("Something went wrong!")
        
       
        # plt.title(title, pad=20)
        plt.colorbar(label=r"$Discounted Value\ (v_\gamma(s_0))$")
        plt.legend(loc='upper right')
        
        # make room for the lifted title
        plt.gcf().subplots_adjust(top=0.85)
        
        plt.savefig(os.path.join(self.output_dir, f"discounted_value_trajectory_{method_name}_gamma_{gamma_str}.png"), bbox_inches='tight')
        plt.close()
        
        
         # 4) Save grid CSV
        df_grid = pd.DataFrame(grid_records)
        filename_csv_grid = f"discounted_value_landscape_for_trajectory_{method_name}_gamma_{gamma_str}.csv"
        csv_path_grid = os.path.join(self.output_dir, filename_csv_grid)
        df_grid.to_csv(csv_path_grid, index=False)
        logger.info(f"Saved Discounted‐value‐landscape CSV: {csv_path_grid}")

        # 5) Save trajectory CSV
        df_traj = pd.DataFrame(traj_records)
        filename_csv_traj = f"discounted_value_trajectory_data_{method_name}_gamma_{gamma_str}.csv"
        csv_path_traj = os.path.join(self.output_dir, filename_csv_traj)
        df_traj.to_csv(csv_path_traj, index=False)
        logger.info(f"Saved Discounted‐value‐trajectory CSV: {csv_path_traj}")
    
    