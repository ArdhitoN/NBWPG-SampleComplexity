import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def load_optimal_values(exh_csv):
    """
    From exhaustive search results, find the gain-optimal policies and
    among them the bias-optimal. Return (optimal_gain, optimal_bias).
    """
    df = pd.read_csv(exh_csv)
    max_gain = df['gain'].max()
    df_gain_opt = df[df['gain'] == max_gain]
    max_bias = df_gain_opt['bias'].max()
    return max_gain, max_bias

def compute_effectiveness(env_name='env-a1', output_dir=None):
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = output_dir if output_dir is not None else SCRIPT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..')) # code
    FIGS_DIR = os.path.join(PROJECT_ROOT, 'figs')                   # code/figs
    AGENT_DIR = os.path.join(PROJECT_ROOT, 'agent')                # code/agent
    EXH_DIR = os.path.join(PROJECT_ROOT, 'xh_search_results')                # code/xh_search_results

    EXPERIMENT_OUTPUT_DIR = os.path.join(FIGS_DIR, 'experiment_outputs_final')
    ENV_OUTPUT_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, env_name)  # adjust per env name
    FILE_DIR = os.path.join(ENV_OUTPUT_DIR, 'visualizations')       # where CSVs are stored

    # Environment and agent config
    ENV_CONFIG = os.path.join(PROJECT_ROOT, 'environment', f'{env_name}.yml')
    AGENT_CONFIG = os.path.join(AGENT_DIR, 'agent.yml')

    # Patterns for gain CSVs (discounted and discounting-free)
    PATTERNS = [
        os.path.join(FILE_DIR, 'bias_progression_discounted_*_gamma_*.csv'),
        os.path.join(FILE_DIR, 'bias_progression_discounting_free_*_gamma_None.csv')
    ]

    EXH_CSV = os.path.join(EXH_DIR, f'{env_name}_exhaustive_search_results.csv')
    # Load optimal values
    optimal_gain, optimal_bias = load_optimal_values(EXH_CSV)

    # Gather method files
    files = []
    for pat in PATTERNS:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        print(f"No method CSVs found matching: {PATTERNS}")
        return

    summary = []
    for fpath in files:
        method = os.path.basename(fpath).replace('.csv', '')
        if 'exact' in method:
            continue
        
        df = pd.read_csv(fpath)
        final = df.iloc[-1]
        gain_f = final['gain']
        bias_f = final['bias']

        # Gain effectiveness
        if optimal_gain != 0:
            gain_effectiveness = 1 - (optimal_gain - gain_f) / abs(optimal_gain)
        else:
            gain_effectiveness = 1.0 if gain_f == 0 else 0.0

        # Bias effectiveness
        if optimal_bias != 0:
            bias_effectiveness = 1 - (optimal_bias - bias_f) / abs(optimal_bias)
        else:
            bias_effectiveness = 1.0 if bias_f == 0 else 0.0

        # Gain difference for reference
        gain_diff = optimal_gain - gain_f
        effective_gain_flag = gain_diff <= 1

        method_name = os.path.basename(fpath).replace('.csv', '')

        summary.append({
            'method': method_name,
            'gain_final': gain_f,
            'optimal_gain': optimal_gain,
            'gain_diff': gain_diff,
            'effective_gain': effective_gain_flag,
            'gain_effectiveness': gain_effectiveness,
            'bias_final': bias_f,
            'optimal_bias': optimal_bias,
            'bias_effectiveness': bias_effectiveness
        })

    df_sum = pd.DataFrame(summary)
    df_sum['effective_bias'] = df_sum['bias_effectiveness'] >= 0.5  # Adjust threshold as needed
    
    def truncate_label(label):
        clean = label.replace('bias_progression_', '').replace('gain_progression_', '')
        clean = re.sub(r"discounting_free_", r"DF_", clean)
        clean = re.sub(r"discounted_", r"D_", clean)
        clean = re.sub(r"gamma_", r"$\\gamma$=", clean)
        # Match numbers (including decimals) after gamma_ and format to two decimal places
        clean = re.sub(r"(\$\\gamma\$=)(\d+\.\d+)", lambda m: f"{m.group(1)}{float(m.group(2)):.2f}", clean)
        # Apply method name shortcuts
        clean = clean.replace('vanilla', 'V').replace('natural', 'N').replace('proper', 'Prop').replace('popular', 'Pop')
        # Remove 'sampling' from DF_ methods
        clean = clean.replace('_sampling', '')
        return clean
    
    df_sum['label'] = df_sum['method'].apply(truncate_label)

    # Sort methods: D_ methods first, then DF_ methods, with V before N/Prop/Pop within each group
    def custom_sort_key(label):
        prefix = 0 if label.startswith('D_') else 1
        method_order = {'V': 0, 'N': 1, 'Pop': 2, 'Prop': 3}
        method = next((m for m in method_order.keys() if m in label), 'N') 
        return (prefix, method_order[method], label)
    
    df_sum['sort_key'] = df_sum['label'].apply(custom_sort_key)
    df_sum = df_sum.sort_values(by='sort_key').drop(columns='sort_key')

    out_csv = os.path.join(OUTPUT_DIR, f'{env_name}_method_effectiveness_summary.csv')
    print(f"Saved summary to {out_csv}")
    df_sum.to_csv(out_csv, index=False)

    # Visualizations
    # Set larger font sizes for ticks (3x default, assuming default 'medium' ≈ 10 points)
    BASE_TICK_SIZE = 10  
    TICK_SIZE = BASE_TICK_SIZE * 1.4
    # Set larger font sizes for title (3x default) and x-axis label 
    BASE_TITLE_SIZE = 12  
    TITLE_SIZE = BASE_TITLE_SIZE * 1.4
    BASE_LABEL_SIZE = 10  
    LABEL_SIZE = BASE_LABEL_SIZE * 1.4

    # 1. Gain effectiveness bar chart
    plt.figure(figsize=(8,6))
    colors_g = df_sum['effective_gain'].map({True: 'blue', False: 'orange'})
    # Reverse the order of labels to show D_ methods at the top
    plt.barh(df_sum['label'][::-1], df_sum['gain_effectiveness'][::-1], color=colors_g[::-1])
    plt.xlabel('Gain Effectiveness', fontsize=LABEL_SIZE)
    plt.title('Gain Effectiveness', fontsize=TITLE_SIZE)
    plt.tick_params(axis='x', labelsize=TICK_SIZE)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{env_name}_gain_effectiveness.png'))
    plt.close()

    # 2. Horizontal Bias Effectiveness (gain-effective only)
    plt.figure(figsize=(8,6))
    bias_vals = df_sum.apply(lambda r: r['bias_effectiveness'] if r['effective_gain'] else 0, axis=1)
    colors_b = df_sum.apply(lambda r: 'green' if (r.effective_gain and r.effective_bias)
                            else ('red' if r.effective_gain else 'gray'), axis=1)
    # Reverse the order of labels to show D_ methods at the top
    plt.barh(df_sum['label'][::-1], bias_vals[::-1], color=colors_b[::-1])
    plt.axvline(0.5, color='black', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Bias Effectiveness', fontsize=LABEL_SIZE)
    plt.title('Bias Effectiveness (Gain-Effective Only)', fontsize=TITLE_SIZE)
    plt.legend()
    plt.tick_params(axis='x', labelsize=TICK_SIZE)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)
    plt.savefig(os.path.join(OUTPUT_DIR, f'{env_name}_bias_effectiveness.png'))
    plt.close()

    return df_sum

if __name__ == '__main__':
    df_summary = compute_effectiveness(env_name='env-b3', output_dir='./figs')
    print(df_summary.to_string(index=False))