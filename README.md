# Sample Complexity of Policy Gradient Reinforcement Learning in Environments With Transient States

## Abstract
Reinforcement learning (RL), while useful for sequential decision-making problems, is often challenging to apply due to the high computational resources required. This research addresses the critical need for sample-efficient RL algorithms, particularly in environments with transient states. We focus on a problem setting with transient states because in real-world decision-making, choices made in these states are crucial. However, many implementations overlook these states, leading to suboptimal modeling or environment designs. We require algorithms to meet the Blackwell optimality criterion, which is the ideal criterion in our problem setting, as a prerequisite for assessing their sample complexity, since an efficient algorithm is not meaningful if it fails to solve the problem. However, while the Blackwell criterion is theoretically ideal, it is challenging to apply, necessitating approximations via two more tractable criteria: discounted reward and discounting-free bias optimality. Discounted reward algorithms, though popular, often still produce discrepancies between predicted and actual outcomes, and many implementations deviate from the theoretical principles governing sample utilization. To bridge the gap between theoretical and applied RL, we investigate two questions: (1) How does the sample complexity of discounting-free policy gradient methods compare to that of discounted policy gradient methods? (2) How do ”proper” (geometrically distributed truncation) versus ”popular” (fixed-horizon) sampling method affect performance within the discounted framework? Empirical results show that discounting-free policy gradient methods are, on average, more sample-efficient than their discounted counterparts, while in the discounted framework, proper sampling enhances sample efficiency, but popular sampling yields better convergence on average.

## How To Run:

1. Create environment with conda:

   `conda env create -f environment.yml`

2. Activate the environment

   `conda activate experiment`

3. Go to core directory
   `cd core`

3. In the directory, select an experiment to run:
   
      `python -m experiment.<experiment_executable>`

   * To find nearly Blackwell optimal policies:
   
      `python -m experiment.exhaustive_search`

   * To run the NBWPGs, its visualizations, progressions, initial sample complexity measurements:
   
      `python -m experiment.experiment_nbwpg`

   * To compute algorithm effectiveness w.r.t Blackwell:
   
      `python -m experiment compute_method_effectiveness`

   * To plot sample complexity visualizations:
   
      `python -m experiment.plot_sample_complexity -c figs/experiment_outputs_final/<ENV>/<ENV>_convergence_data.csv -o <OUTPUT_DIR>` 


   * Miscellany:
      * To plot bias progressions: 
   
         `python -m experiment.plot_bias_progression`
