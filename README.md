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
   `python -m experiment.compute_method_effectiveness`

   * To plot sample complexity visualizations:
   `python -m experiment.plot_sample_complexity -c figs/experiment_outputs_final/<ENV>/<ENV>_convergence_data.csv`


   * Miscellany:
      * To plot bias progressions: 
      `python -m experiment.plot_bias_progression`