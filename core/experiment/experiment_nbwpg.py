import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm 
import optuna
from visualizer import Visualizer

from core.environment.env import Env
from core.agent.agent import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_final_comparison.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- Global Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXP_CONFIG_PATH = os.path.join(BASE_DIR, "experiment.yml")

# OUTPUT_DIR_BASE will be set from config or defaulted
# To be defined after OUTPUT_DIR_BASE is known
BEST_GAMMA_CACHE_DIR = None 
GAMMA_TUNING_PLOTS_DIR = None 
VISUALIZATIONS_DIR = None 

# Global dict for Optuna objective arguments (simplified approach)
_optuna_objective_data = {}

def setup_paths(config, env_name_prefix=None):
    """Sets up global path variables based on config."""
    global OUTPUT_DIR_BASE, BEST_GAMMA_CACHE_DIR, GAMMA_TUNING_PLOTS_DIR, VISUALIZATIONS_DIR
    
    output_dir_name = config.get("output_directory_name", "experiment_outputs_unified")
    OUTPUT_DIR_BASE = os.path.join(BASE_DIR, "..", "figs", output_dir_name, env_name_prefix)
    
    BEST_GAMMA_CACHE_DIR = os.path.join(OUTPUT_DIR_BASE, "best_gamma_cache")
    GAMMA_TUNING_PLOTS_DIR = os.path.join(OUTPUT_DIR_BASE, "gamma_tuning_plots")
    VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR_BASE, "visualizations") # Main dir for Visualizer outputs

    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    os.makedirs(BEST_GAMMA_CACHE_DIR, exist_ok=True)
    os.makedirs(GAMMA_TUNING_PLOTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    logger.info(f"Base output directory: {OUTPUT_DIR_BASE}")

def get_env_name_prefix(env_config_filename_full):
    """Extracts a clean environment name prefix from the filename."""
    base_name = os.path.basename(env_config_filename_full)
    env_name_prefix, _ = os.path.splitext(base_name) # Remove .yml
    return env_name_prefix

def run_single_config_discounted(method_variant, use_sampling, sampling_type,
                      gamma_val, seed_val, env_path, agent_config_path,
                      num_iterations, sampling_n_exp_episodes, sampling_t_max,
                      eval_interval, eval_n_exp_episodes, eval_t_max):
    """
    Runs a single experimental configuration and returns its evaluation history.
    """
    current_env = Env(env_path, seed=seed_val)
    agent = Agent(
        config_path=agent_config_path,
        gamma=gamma_val,
        seed=seed_val,
        env=current_env,
    )

    full_method_name = f"discounted_{method_variant}{'_'+sampling_type if use_sampling else '_exact'}"
    
    evaluation_history, cumulative_discounted_rewards_exact, samples_to_converge = agent.discounted_polgrad(
        variant=method_variant,
        use_sampling=use_sampling,
        iterations=num_iterations,
        sampling_method=sampling_type if use_sampling else 'proper',
        sampling_n_exp_episodes=sampling_n_exp_episodes,
        sampling_t_max=sampling_t_max,
        eval_interval=eval_interval,
        eval_n_exp_episodes=eval_n_exp_episodes,
        eval_t_max=eval_t_max
    )
    
    # Calculate samples used in this run
    if use_sampling:
        samples_used = num_iterations * sampling_n_exp_episodes * sampling_t_max
    
    # Each iteration is one "sample" for exact methods
    else:
        samples_used = num_iterations  
    
    return evaluation_history, full_method_name, agent, samples_to_converge, samples_used


def run_single_config_discounting_free(method_variant, use_sampling,
                               seed_val, env_path, agent_config_path_main,
                               eval_interval=None, eval_n_exp_episodes=None, eval_t_max=None,
                               outer_iterations=1, inner_iterations=100):
    current_env = Env(env_path, seed=seed_val)
    agent = Agent(
        config_path=agent_config_path_main, gamma=None, seed=seed_val, env=current_env,
    )
    full_method_name = f"discounting_free_{method_variant}{'_sampling' if use_sampling else '_exact'}"
     
    eval_history, gain_samples_for_convergence, total_samples_for_convergence  = agent.discounting_free_polgrad(
        variant=method_variant, use_sampling=use_sampling, outer_iterations=outer_iterations, inner_iterations=inner_iterations, n_exp_episodes=16,
    )
    return eval_history, full_method_name, agent, gain_samples_for_convergence, total_samples_for_convergence


def objective_for_optuna(trial: optuna.Trial):
    global _optuna_objective_data
    method_config_tuple = _optuna_objective_data['method_config_tuple']
    tuning_seeds = _optuna_objective_data['tuning_seeds']
    env_path = _optuna_objective_data['env_path']
    agent_config_path = _optuna_objective_data['agent_config_path_main']
    num_iterations = _optuna_objective_data['num_iterations']
    sampling_n_exp_episodes = _optuna_objective_data['sampling_n_exp_episodes']
    sampling_t_max = _optuna_objective_data['sampling_t_max']
    eval_interval = _optuna_objective_data['eval_interval']
    eval_n_exp_episodes = _optuna_objective_data['eval_n_exp_episodes']
    eval_t_max = _optuna_objective_data['eval_t_max']
    eval_metric_window = _optuna_objective_data['eval_metric_window']
    gamma_search_space = _optuna_objective_data['gamma_search_space']

    method_variant, use_sampling, sampling_type = method_config_tuple
    gamma_val = trial.suggest_float("gamma", gamma_search_space[0], gamma_search_space[1])
    seed_final_performances = []
    current_gamma_histories = []
    total_samples_used = 0
    for seed_val in tuning_seeds:
        eval_history, _, _, samples_to_converge, _ = run_single_config_discounted(
            method_variant, use_sampling, sampling_type, gamma_val, seed_val, 
            env_path, agent_config_path, num_iterations, sampling_n_exp_episodes, sampling_t_max, 
            eval_interval, eval_n_exp_episodes, eval_t_max
        )
        logger.info(f"samples to converge: {samples_to_converge}")
        total_samples_used += samples_to_converge
        if eval_history:
            current_gamma_histories.append(eval_history)
            final_points = [r for _, r in eval_history[-eval_metric_window:]]
            if final_points:
                seed_final_performances.append(np.mean(final_points))
    trial.set_user_attr("histories", current_gamma_histories)
    trial.set_user_attr("gamma_tested", gamma_val)
    trial.set_user_attr("samples_used", total_samples_used)
    return np.mean(seed_final_performances) if seed_final_performances else -float('inf')

def run_tuning_for_method(tuning_mode, method_config_tuple, common_params, optuna_params, grid_params):
    method_variant, use_sampling, sampling_type = method_config_tuple
    full_method_name_for_cache = f"discounted_{method_variant}{'_' + sampling_type if use_sampling else '_exact'}"
    env_name_prefix_for_cache = get_env_name_prefix(os.path.basename(common_params['env_path']))
    cache_file_suffix = "optuna.yml" if tuning_mode == "optuna" else "grid.yml"
    cache_file_path = os.path.join(BEST_GAMMA_CACHE_DIR, f"{env_name_prefix_for_cache}_{full_method_name_for_cache}_best_gamma_{cache_file_suffix}")
    
    
    # if os.path.exists(cache_file_path):
    #     with open(cache_file_path, 'r') as f:
    #         cached_data = yaml.safe_load(f)
    #     if cached_data and 'best_gamma' in cached_data:
    #         logger.info(f"Loaded cached best gamma for {full_method_name_for_cache}: {cached_data['best_gamma']:.4f}")
    #         return cached_data['best_gamma'], {}, full_method_name_for_cache, 0  # No samples used if cached

    global _optuna_objective_data
    _optuna_objective_data = {**common_params, 'method_config_tuple': method_config_tuple}
    total_tuning_samples = 0

    if tuning_mode == "optuna":
        _optuna_objective_data['gamma_search_space'] = optuna_params['gamma_search_space']
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=_optuna_objective_data['tuning_seeds'][0]))
        study.optimize(objective_for_optuna, n_trials=optuna_params['optuna_n_trials'], show_progress_bar=True)
        best_gamma = study.best_trial.params['gamma']
        best_performance = study.best_trial.value
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs:
                logger.info("gamma tested")
                logger.info(trial.user_attrs["gamma_tested"])

                logger.info("histories")
                logger.info(trial.user_attrs["histories"])
                
                logger.info("samples used")
                logger.info(trial.user_attrs.get("samples_used"))
            
        all_histories = {trial.user_attrs["gamma_tested"]: trial.user_attrs["histories"] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs}
        total_tuning_samples = sum(trial.user_attrs.get("samples_used", 0) for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE)
    elif tuning_mode == "grid":
        best_performance = -float('inf')
        best_gamma = None
        all_histories = {}
        for gamma_val in grid_params['gamma_values_for_grid']:
            seed_final_performances = []
            current_gamma_histories = []
            for seed_val in common_params['tuning_seeds']:
                eval_history, _, _, _, samples_used = run_single_config_discounted(
                    method_variant, use_sampling, sampling_type, gamma_val, 
                    seed_val, common_params['env_path'], common_params['agent_config_path_main'], 
                    common_params['num_iterations'], common_params['sampling_n_exp_episodes'], 
                    common_params['sampling_t_max'], common_params['eval_interval'], 
                    common_params['eval_n_exp_episodes'], common_params['eval_t_max']
                )
                total_tuning_samples += samples_used
                if eval_history:
                    current_gamma_histories.append(eval_history)
                    final_points = [r for _, r in eval_history[-common_params['eval_metric_window']:]]
                    if final_points:
                        seed_final_performances.append(np.mean(final_points))
            all_histories[gamma_val] = current_gamma_histories
            if seed_final_performances and np.mean(seed_final_performances) > best_performance:
                best_performance = np.mean(seed_final_performances)
                best_gamma = gamma_val
        if best_gamma is None:
            best_gamma = grid_params['gamma_values_for_grid'][0]
    else:
        best_gamma = common_params['default_gamma_if_tuning_none']
        all_histories = {}
        current_gamma_histories = []
        for seed_val in common_params['tuning_seeds']:
            eval_history, _, _, _, samples_used = run_single_config_discounted(
                method_variant, use_sampling, sampling_type, best_gamma, seed_val, common_params['env_path'], common_params['agent_config_path_main'], common_params['num_iterations'], common_params['sampling_n_exp_episodes'], common_params['sampling_t_max'], common_params['eval_interval'], common_params['eval_n_exp_episodes'], common_params['eval_t_max']
            )
            total_tuning_samples += samples_used
            if eval_history:
                current_gamma_histories.append(eval_history)
        all_histories[best_gamma] = current_gamma_histories

    with open(cache_file_path, 'w') as f:
        yaml.dump({'best_gamma': best_gamma, 'performance': float(best_performance if tuning_mode != "none" else -float('inf'))}, f)
    return best_gamma, all_histories, full_method_name_for_cache, total_tuning_samples


def tune_gamma_with_optuna(method_config_tuple, tuning_seeds,
                           env_path, agent_config_path,
                           num_iterations, sampling_n_exp_episodes, sampling_t_max,
                           eval_interval, eval_n_exp_episodes, eval_t_max,
                           eval_metric_window, 
                           gamma_search_space, optuna_n_trials):
    
    method_variant, use_sampling, sampling_type = method_config_tuple
    full_method_name_for_cache = f"discounted_{method_variant}"
    if use_sampling: 
        full_method_name_for_cache += f"_sampling_{sampling_type}"
    else: 
        full_method_name_for_cache += "_exact"
    
    env_name_prefix_for_cache = get_env_name_prefix(os.path.basename(env_path))
    cache_file_path = os.path.join(BEST_GAMMA_CACHE_DIR, f"{env_name_prefix_for_cache}_{full_method_name_for_cache}_best_gamma_optuna.yml")

    # if os.path.exists(cache_file_path):
    #     try:
    #         with open(cache_file_path, 'r') as f:
    #             cached_data = yaml.safe_load(f)
    #         if cached_data and 'best_gamma' in cached_data:
    #             logger.info(f"Loaded cached best gamma for {full_method_name_for_cache} on {env_name_prefix_for_cache} (Optuna tuned): {cached_data['best_gamma']:.4f} (Perf: {cached_data.get('performance', 'N/A'):.4f})")
    #             # For Optuna, we'd ideally cache the study or at least the trials to reconstruct plots.
    #             # For now, if cache hits, we return the gamma but won't have histories for plotting gamma search.
    #             # To get plots, delete cache or modify to store/load trial data.
    #             # Returning empty histories if cache is hit for simplicity of this example.
    #             return cached_data['best_gamma'], {}, full_method_name_for_cache 
    #     except Exception as e:
    #         logger.warning(f"Error loading cache file {cache_file_path}: {e}. Retuning with Optuna.")

    logger.info(f"Tuning gamma with Optuna for method: {full_method_name_for_cache} on env: {env_name_prefix_for_cache} \
                using {len(tuning_seeds)} seeds. Trials: {optuna_n_trials}")

    # Pass necessary data to the objective function
    global _optuna_objective_data
    _optuna_objective_data = {
        'method_config_tuple': method_config_tuple,
        'tuning_seeds': tuning_seeds,
        'env_path': env_path,
        'agent_config_path': agent_config_path,
        'num_iterations': num_iterations,
        'sampling_n_exp_episodes': sampling_n_exp_episodes,
        'sampling_t_max': sampling_t_max,
        'eval_interval': eval_interval,
        'eval_n_exp_episodes': eval_n_exp_episodes,
        'eval_t_max': eval_t_max,
        'eval_metric_window': eval_metric_window,
        'gamma_search_space': gamma_search_space
    }

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_for_optuna, n_trials=optuna_n_trials, show_progress_bar=True)

    best_gamma_for_this_method = study.best_trial.params['gamma']
    best_avg_final_performance = study.best_trial.value
    
    logger.info(f"Optuna tuning complete for {full_method_name_for_cache} on {env_name_prefix_for_cache}.")
    logger.info(f"Best gamma found: {best_gamma_for_this_method:.4f} (Performance: {best_avg_final_performance:.4f})")

    try:
        with open(cache_file_path, 'w') as f:
            yaml.dump({'best_gamma': best_gamma_for_this_method, 
                       'performance': float(best_avg_final_performance),
                       'env_name': env_name_prefix_for_cache,
                       'optuna_trials_completed': len(study.trials)}, f)
    except Exception as e:
        logger.error(f"Error saving Optuna cache file {cache_file_path}: {e}")

    # Reconstruct all_tuning_eval_histories from study trials for plotting
    all_tuning_eval_histories = {}
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.user_attrs: 
            gamma_tested = trial.user_attrs.get("gamma_tested")
            histories = trial.user_attrs.get("histories")
            if gamma_tested is not None and histories:
                 # If multiple trials test the same gamma (can happen if search space is small or discrete suggestion)
                 # this will overwrite. For float gammas, less likely.
                 # For plotting, we might want to average if multiple trials hit near-identical gammas.
                 # Here, we just take the last one for a given gamma if there are collisions (unlikely for float).
                all_tuning_eval_histories[gamma_tested] = histories
            else:
                logger.debug(f"Trial {trial.number} missing user_attrs for gamma/histories.")


    return best_gamma_for_this_method, all_tuning_eval_histories, full_method_name_for_cache


def tune_best_gamma_for_method(method_config_tuple, gamma_values, tuning_seeds,
                               env_path, agent_config_path,
                               num_iterations, sampling_n_exp_episodes, sampling_t_max,
                               eval_interval, eval_n_exp_episodes, eval_t_max,
                               eval_metric_window):
    method_variant, use_sampling, sampling_type = method_config_tuple
    
    full_method_name_for_cache = f"discounted_{method_variant}"
    if use_sampling: 
        full_method_name_for_cache += f"_sampling_{sampling_type}"
    else: 
        full_method_name_for_cache += "_exact"
    
    # Include env name in cache file path to avoid conflicts if tuning for different envs
    env_name_prefix_for_cache = get_env_name_prefix(os.path.basename(env_path))
    cache_file_path = os.path.join(BEST_GAMMA_CACHE_DIR, f"{env_name_prefix_for_cache}_{full_method_name_for_cache}_best_gamma.yml")


    logger.info(f"Tuning gamma for method: {full_method_name_for_cache} on env: {env_name_prefix_for_cache} using {len(tuning_seeds)} seeds.")
    
    best_avg_final_performance = -float('inf')
    best_gamma_for_this_method = None
    all_tuning_eval_histories = {} 

    for gamma_val in tqdm(gamma_values, desc=f"Tuning Gamma for {full_method_name_for_cache} ({env_name_prefix_for_cache})", leave=False):
        seed_final_performances = []
        current_gamma_histories = [] 
        for seed_val in tuning_seeds:
            eval_history, _ = run_single_config_discounted(
                method_variant, use_sampling, sampling_type, gamma_val, seed_val,
                env_path, agent_config_path, num_iterations, sampling_n_exp_episodes,
                sampling_t_max, eval_interval, eval_n_exp_episodes, eval_t_max
            )
            if eval_history:
                current_gamma_histories.append(eval_history) 
                final_points = [r for _, r in eval_history[-eval_metric_window:]]
                if final_points:
                    seed_final_performances.append(np.mean(final_points))
        
        all_tuning_eval_histories[gamma_val] = current_gamma_histories

        if seed_final_performances:
            current_gamma_avg_performance = np.mean(seed_final_performances)
            if current_gamma_avg_performance > best_avg_final_performance:
                best_avg_final_performance = current_gamma_avg_performance
                best_gamma_for_this_method = gamma_val
    
    if best_gamma_for_this_method is not None:
        logger.info(f"Best gamma for {full_method_name_for_cache} on {env_name_prefix_for_cache}: {best_gamma_for_this_method} (Performance: {best_avg_final_performance:.4f})")
        try:
            with open(cache_file_path, 'w') as f:
                yaml.dump({'best_gamma': best_gamma_for_this_method, 
                           'performance': float(best_avg_final_performance), # Ensure float for YAML
                           'env_name': env_name_prefix_for_cache}, f)
        except Exception as e:
            logger.error(f"Error saving cache file {cache_file_path}: {e}")
    else:
        logger.warning(f"Could not determine best gamma for {full_method_name_for_cache} on {env_name_prefix_for_cache}. Defaulting to first gamma: {gamma_values[0]}")
        best_gamma_for_this_method = gamma_values[0] 

    return best_gamma_for_this_method, all_tuning_eval_histories, full_method_name_for_cache




def plot_gamma_tuning_curves_per_method(env_name_prefix, method_name, tuning_histories_map, output_dir, tuning_mode):
    if not tuning_histories_map: return
    plt.figure(figsize=(12, 8))
    for gamma_val in sorted(tuning_histories_map.keys()):
        histories = tuning_histories_map[gamma_val]
        if not histories: continue
        all_seeds_data = [pd.DataFrame(h, columns=['iteration', 'metric_value']) for h in histories if h]
        if not all_seeds_data: continue
        agg_df = pd.concat(all_seeds_data).groupby('iteration')['metric_value'].agg(['mean', 'std']).reset_index()
        plt.plot(agg_df['iteration'], agg_df['mean'], label=f"γ={gamma_val:.4f}", alpha=0.8)
        plt.fill_between(agg_df['iteration'], agg_df['mean'] - agg_df['std'].fillna(0), agg_df['mean'] + agg_df['std'].fillna(0), alpha=0.15)
    title_suffix = {"optuna": "Optuna Trials", "grid": "Grid Search", "none": "Fixed Gamma"}.get(tuning_mode, tuning_mode)
    plt.title(f"Gamma Values ({title_suffix}) for: {method_name} on Env: {env_name_prefix}")
    plt.xlabel("Iterations/Outer Iterations"); plt.ylabel("Avg Reward/Gain (Tuning Seeds)")
    plt.legend(title="Gamma (γ)", loc='best'); plt.grid(True, linestyle='--', alpha=0.7)
    file_path = os.path.join(output_dir, f"{env_name_prefix}_{method_name.replace(' ','_')}_{tuning_mode}_gamma_explore.png")
    try: plt.savefig(file_path); logger.info(f"Saved: {file_path}")
    except Exception as e: logger.error(f"Plot save error {file_path}: {e}")
    plt.close()

def plot_final_comparison_curves(env_name_prefix, final_results_df, output_dir, tuning_mode_suffix=""):
    if final_results_df.empty: return
    plt.figure(figsize=(15, 10))
    avg_df = final_results_df[final_results_df['gamma_used'] == 'avg']
    disc_df = final_results_df[final_results_df['gamma_used'] != 'avg']
    ax1 = plt.gca()
    if not disc_df.empty:
        for method_name in sorted(disc_df['method_name'].unique()):
            m_df = disc_df[disc_df['method_name'] == method_name]
            if m_df.empty: continue
            gamma_lbl = f"{m_df['gamma_used'].iloc[0]:.4f}" if isinstance(m_df['gamma_used'].iloc[0], float) else m_df['gamma_used'].iloc[0]
            agg_df = m_df.groupby('iteration')['metric_value'].agg(['mean', 'std']).reset_index()
            ax1.plot(agg_df['iteration'], agg_df['mean'], label=f"{method_name} (γ={gamma_lbl})")
            ax1.fill_between(agg_df['iteration'], agg_df['mean']-agg_df['std'].fillna(0), agg_df['mean']+agg_df['std'].fillna(0), alpha=0.2)
        ax1.set_ylabel("Avg Reward (Discounted)"); ax1.legend(loc='lower right', title="Discounted")
    if not avg_df.empty:
        ax2 = ax1.twinx()
        colors = plt.cm.get_cmap('viridis', len(avg_df['method_name'].unique()))
        for i, method_name in enumerate(sorted(avg_df['method_name'].unique())):
            m_df = avg_df[avg_df['method_name'] == method_name]
            if m_df.empty: continue
            agg_df = m_df.groupby('iteration')['metric_value'].agg(['mean', 'std']).reset_index()
            ax2.plot(agg_df['iteration'], agg_df['mean'], label=f"{method_name} (Avg Gain)", linestyle='--', color=colors(i))
            ax2.fill_between(agg_df['iteration'], agg_df['mean']-agg_df['std'].fillna(0), agg_df['mean']+agg_df['std'].fillna(0), alpha=0.1, color=colors(i))
        ax2.set_ylabel("Avg Gain (Average Reward)"); ax2.legend(loc='upper left', title="Average Reward")
    ax1.set_xlabel("Iterations / Outer Iterations"); ax1.set_title(f"Final Comparison on {env_name_prefix}")
    ax1.grid(True, linestyle=':', alpha=0.7); plt.tight_layout(rect=[0,0,0.9,1] if not disc_df.empty and not avg_df.empty else None)
    file_path = os.path.join(output_dir, f"{env_name_prefix}_final_comparison_all{'_'+tuning_mode_suffix if tuning_mode_suffix else ''}_curves.png")
    try: plt.savefig(file_path); logger.info(f"Saved: {file_path}")
    except Exception as e: logger.error(f"Plot save error {file_path}: {e}")
    plt.close()


def create_convergence_boxplot(env_name_prefix, convergence_data, output_dir):
    """
    Creates a boxplot showing the number of samples needed for convergence,
    excluding any methods that end with '_exact'. 
    Discounting‐free methods are colored blue; discounted methods are colored orange. 
    The y‐axis is scaled in thousands of samples.
    """
    # 1. Filter out any methods that end with '_exact'
    filtered_items = [
        (method, samples)
        for method, samples in convergence_data.items()
        if not method.endswith('_exact')
    ]

    if not filtered_items:
        logger.warning("No non‐exact methods available for boxplot.")
        return
    
    
    # Build a “long” DataFrame with one row per (method, sample)
    csv_records = []
    for method, samples in filtered_items:
        for s in samples:
            csv_records.append({
                'method_name': method,
                'samples_to_converge': s
            })
            
    df_conv = pd.DataFrame(csv_records)
    csv_filename = f"{env_name_prefix}_convergence_data.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    try:
        os.makedirs(output_dir, exist_ok=True)
        df_conv.to_csv(csv_path, index=False)
        logger.info(f"Convergence data CSV saved to {csv_path}")
    except Exception as e:
        logger.error(f"Error saving convergence CSV {csv_path}: {e}")
    # --------------------------------------------------
    

    # 2. Unzip method names and their sample lists
    methods, samples_lists = zip(*filtered_items)

    # 3. Scale all sample counts by 1000
    scaled_data = [
        [s / 1000.0 for s in samples]  
        for samples in samples_lists
    ]

    # 4. Assign colors
    colors = []
    for method in methods:
        if method.startswith("discounting_free_"):
            colors.append("#1f77b4")  
        elif method.startswith("discounted_"):            
            colors.append("#ff7f0e")  
        else:
            colors.append("#7f7f7f")  

    # 5. Plot
    plt.figure(figsize=(12, 8))
    bplot = plt.boxplot(
        scaled_data,
        vert=True,
        patch_artist=True,
        labels=list(methods),
        widths=0.6,
        # boxprops=dict(linewidth=2),
        # whiskerprops=dict(linewidth=1.5),
        # capprops=dict(linewidth=1.5),
        # medianprops=dict(linewidth=2, color='k'),

    )

    # 6. Color each box according to its method type
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)   # match the face color
        patch.set_linewidth(2)

    # 7. Color each x‐tick label to match its box
    ax = plt.gca()
    for tick_label, color in zip(ax.get_xticklabels(), colors):
        tick_label.set_color(color)

    # 8. Final tweaks: title, axis labels, rotation, grid
    plt.title(f"Samples Needed for Convergence on {env_name_prefix}")
    plt.xlabel("Method")
    plt.ylabel("Number of Samples (in thousands)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # 9. Save to file
    output_path = os.path.join(output_dir, f"{env_name_prefix}_convergence_samples_boxplot.png")
    try:
        plt.savefig(output_path)
        logger.info(f"Convergence boxplot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving convergence boxplot to {output_path}: {e}")
    finally:
        plt.close()

def run_discounting_free_polgrad(config, common_data, final_evaluation_results_data, convergence_data):
    logger.info(f"--- Running Discounting Free Experiments for Env: {common_data['env_name_prefix']} ---")
    
    discounting_free_cfg = config.get("discounting_free_params", {})
    outer_iterations = discounting_free_cfg.get("outer_iterations", 1)
    inner_iterations = discounting_free_cfg.get("inner_iterations", 100)
    
    method_configs = discounting_free_cfg.get("method_configs", [
        ("vanilla", False), ("vanilla", True), 
        ("natural", False), ("natural", True),
    ])

    for method_variant, use_sampling in tqdm(method_configs, desc="Discounting Free Methods"):
        full_method_name = f"discounting_free_{method_variant}{'_sampling' if use_sampling else '_exact'}"
        convergence_data[full_method_name] = []
        
        logger.info(f"Running: {full_method_name}")
        for seed_val in tqdm(common_data['EVALUATION_SEEDS'], desc=f"Seeds for {method_variant}_{'sampling' if use_sampling else 'exact'}", leave=False):
            eval_history, full_method_name, trained_agent , gain_samples, total_samples = run_single_config_discounting_free(
                method_variant=method_variant, use_sampling=use_sampling, seed_val=seed_val, 
                env_path=common_data['env_config_path'], agent_config_path_main=common_data['agent_config_path_main'],
                outer_iterations=outer_iterations, 
                inner_iterations=inner_iterations
            )
            
            logger.info(f"gain samples: {gain_samples}")
            logger.info(f"total samples: {total_samples}")
            
            for iter_num, policy_gain_plus_bias in enumerate(eval_history):
                final_evaluation_results_data.append({
                    'seed': seed_val, 'method_name': full_method_name, 
                    'gamma_used': 'avg', 'iteration': iter_num,
                    'metric_value': policy_gain_plus_bias, 'env_name': common_data['env_name_prefix'],
                    'experiment_type': 'discounting_free'
                })
                
            if total_samples is not None:
                convergence_data[full_method_name].append(total_samples)
            
                
            if seed_val == common_data['EVALUATION_SEEDS'][0]: # Visualize for first seed
                viz = Visualizer(common_data['env_name_prefix'], trained_agent, None, seed_val, output_dir=VISUALIZATIONS_DIR)
                theta_cfg = config.get("theta_range_params", {"min":-10,"max":10,"points":41})
                theta0_r, theta1_r = [np.linspace(theta_cfg["min"], theta_cfg["max"], theta_cfg["points"])]*2
                if config.get("visualize_landscapes"):
                    viz.visualize_gain_landscape(theta0_r, theta1_r)
                    viz.visualize_bias_landscape(theta0_r, theta1_r)
                if config.get("visualize_trajectories"):
                    viz.visualize_gain_trajectory(theta0_r, theta1_r, full_method_name)
                    if trained_agent.is_gain_converged: 
                        viz.visualize_bias_trajectory(theta0_r, theta1_r, full_method_name)
                if config.get("visualize_progressions"):
                    viz.visualize_gain_progression(full_method_name)
                    viz.visualize_gain_plus_bias_progression(full_method_name)

def run_discounted_polgrad(config, common_data, final_evaluation_results_data, convergence_data):
    logger.info(f"--- Running Discounted Reward Experiments for Env: {common_data['env_name_prefix']} ---")
    
    disc_cfg = config.get("discounted_polgrad_params", {})
    
    # Find tuning seeds
    NUM_TUNING_SEEDS_disc = config.get('num_tuning_seeds', 1)
    if NUM_TUNING_SEEDS_disc > len(common_data['ALL_SEEDS']): 
        NUM_TUNING_SEEDS_disc = len(common_data['ALL_SEEDS'])
    TUNING_SEEDS_disc = common_data['ALL_SEEDS'][:NUM_TUNING_SEEDS_disc]

    common_params_for_tuning = {
        'tuning_seeds': TUNING_SEEDS_disc, 
        'env_path': common_data['env_config_path'], 
        'agent_config_path_main': common_data['agent_config_path_main'],
        'num_iterations': disc_cfg.get('num_iterations',50), 
        'sampling_n_exp_episodes': disc_cfg.get('sampling_n_exp_episodes',1000),
        'sampling_t_max': disc_cfg.get('sampling_t_max',100), 
        'eval_interval': disc_cfg.get('eval_interval',10),
        'eval_n_exp_episodes': disc_cfg.get('eval_n_exp_episodes',10), 
        'eval_t_max': disc_cfg.get('eval_t_max',100),
        'eval_metric_window': disc_cfg.get('eval_metric_window_for_tuning',3),
        'default_gamma_if_tuning_none': disc_cfg.get('default_gamma_if_tuning_none', 0.99)
    }
    optuna_params_cfg = disc_cfg.get('optuna_params', {})
    grid_params_cfg = disc_cfg.get('grid_params', {})
    tuning_mode = disc_cfg.get("tuning_mode", "optuna")

    method_configs_disc = disc_cfg.get("method_configs", [
        ("vanilla", False, None), ("vanilla", True, 'proper'), ("vanilla", True, 'popular'),
        ("natural", False, None), ("natural", True, 'proper'), ("natural", True, 'popular'),
        
    ])
    method_to_best_gamma_map = {}
    method_to_tuning_samples_map = {}

    logger.info(f"--- Phase 1 (Discounted): Gamma Tuning ({tuning_mode}) ---")
    for method_config_tuple in method_configs_disc:
        best_gamma, histories, method_name_for_plot, tuning_samples = run_tuning_for_method(
            tuning_mode, method_config_tuple, common_params_for_tuning,
            optuna_params_cfg, grid_params_cfg
        )
        method_to_best_gamma_map[method_config_tuple] = best_gamma
        method_to_tuning_samples_map[method_config_tuple] = tuning_samples
        
        if histories:
            plot_gamma_tuning_curves_per_method(
                common_data['env_name_prefix'], method_name_for_plot, histories, 
                GAMMA_TUNING_PLOTS_DIR, tuning_mode
            )
    logger.info(f"Discounted Gamma Tuning Complete. Best Gammas: {method_to_best_gamma_map}")


    logger.info(f"--- Phase 2 (Discounted): Final Evaluation & Visualizations ---")
    for method_config_tuple, tuned_gamma in tqdm(method_to_best_gamma_map.items(), desc="Final Eval Discounted"):
        method_variant, use_sampling, sampling_type = method_config_tuple
        if tuned_gamma is None: 
            logger.error(f"Tuned gamma for {method_config_tuple} is None. Skipping eval."); continue
            
            
        full_method_name = f"discounted_{method_variant}{'_' + sampling_type if use_sampling else '_exact'}"
        convergence_data[full_method_name] = []
        
        tuning_samples = method_to_tuning_samples_map.get(method_config_tuple, 0)
        for i, seed_val in enumerate(common_data['EVALUATION_SEEDS']):
            eval_history, full_method_name, trained_agent, samples_to_converge, _ = run_single_config_discounted(
                method_variant, use_sampling, sampling_type, tuned_gamma, seed_val,
                common_data['env_config_path'], common_data['agent_config_path_main'],
                common_params_for_tuning['num_iterations'], common_params_for_tuning['sampling_n_exp_episodes'],
                common_params_for_tuning['sampling_t_max'], common_params_for_tuning['eval_interval'],
                common_params_for_tuning['eval_n_exp_episodes'], common_params_for_tuning['eval_t_max']
            )
            
            if samples_to_converge is not None:
                total_samples_with_tuning = samples_to_converge + tuning_samples
                
                if full_method_name not in convergence_data:
                    convergence_data[full_method_name] = []
                convergence_data[full_method_name].append(total_samples_with_tuning)

            for iter_num, avg_reward in eval_history:
                final_evaluation_results_data.append({
                    'seed': seed_val, 'method_name': full_method_name, 
                    'gamma_used': tuned_gamma, 'iteration': iter_num,
                    'metric_value': avg_reward, 'env_name': common_data['env_name_prefix'],
                    'experiment_type': 'discounted_polgrad'
                })
            
            if i == 0: # Visualize for first seed
                viz = Visualizer(common_data['env_name_prefix'], trained_agent, tuned_gamma, seed_val, output_dir=VISUALIZATIONS_DIR) 
                theta_cfg = config.get("theta_range_params", {"min":-10,"max":10,"points":41})
                theta0_r, theta1_r = [np.linspace(theta_cfg["min"], theta_cfg["max"], theta_cfg["points"])]*2
                temp_env = Env(common_data['env_config_path'], seed=seed_val) 
                
                if config.get("visualize_landscapes"):
                    viz.visualize_discounted_value_landscape(theta0_r, theta1_r, temp_env.initial_state_dist)
                if config.get("visualize_trajectories"):
                    viz.visualize_discounted_value_trajectory(theta0_r, theta1_r, full_method_name, temp_env.initial_state_dist)
                if config.get("visualize_progressions"):
                    viz.visualize_gain_progression(full_method_name) # Plots V(s0)
                    viz.visualize_discounted_value_progression(full_method_name)
                    

def run_experiment():
    try:
        with open(DEFAULT_EXP_CONFIG_PATH, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Experiment config file not found: {DEFAULT_EXP_CONFIG_PATH}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing experiment config file {DEFAULT_EXP_CONFIG_PATH}: {e}")
        return
    
    agent_config_path_main = os.path.join(BASE_DIR, config.get("agent_config_path", "../agent/agent.yml"))

    env_config_file_name = config.get("env_config", "../environment/env-a1.yml")
    env_config_path = os.path.join(BASE_DIR, "..", "environment", env_config_file_name)
    env_name_prefix = get_env_name_prefix(env_config_file_name)
    
    setup_paths(config, env_name_prefix=env_name_prefix) 

    common_data = {
        'env_name_prefix': env_name_prefix,
        'env_config_path': env_config_path,
        'agent_config_path_main': agent_config_path_main,
        'ALL_SEEDS': config.get('seeds', [42]),
        'EVALUATION_SEEDS': config.get('seeds', [42]) 
    }

    final_evaluation_results_data = [] 
    convergence_data = {}  
    
    if config.get("run_discounting_free_polgrad", False):
        run_discounting_free_polgrad(config, common_data, final_evaluation_results_data, convergence_data)
    
    if config.get("run_discounted_polgrad", True):
        run_discounted_polgrad(config, common_data, final_evaluation_results_data, convergence_data)

    logger.info("--- Consolidating and Plotting All Final Results ---")
    if final_evaluation_results_data:
        final_results_df = pd.DataFrame(final_evaluation_results_data)
        disc_tuning_mode_fn = config.get("discounted_reward_params", {}).get("tuning_mode", "") \
                              if config.get("run_discounted_polgrad") else ""
        
        csv_filename = f"{common_data['env_name_prefix']}_final_ALL_results.csv"
        final_results_csv_path = os.path.join(OUTPUT_DIR_BASE, csv_filename)
        try:
            final_results_df.to_csv(final_results_csv_path, index=False)
            logger.info(f"ALL final results saved to {final_results_csv_path}")
        except Exception as e: 
            logger.error(f"Error saving CSV {final_results_csv_path}: {e}")
        
        plot_final_comparison_curves(common_data['env_name_prefix'], final_results_df, 
                                     OUTPUT_DIR_BASE, disc_tuning_mode_fn)
        
        create_convergence_boxplot(common_data['env_name_prefix'], convergence_data, OUTPUT_DIR_BASE)

    else:
        logger.info(f"No final evaluation results from any experiment type for {common_data['env_name_prefix']} to plot/save.")

if __name__ == "__main__":
    run_experiment()