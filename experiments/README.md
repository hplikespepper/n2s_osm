# mvN2S follow-up experiments

Run commands from `n2s_mv`. The scripts default to `python` and GPUs `0,1`;
override them with `PYTHON` and `CUDA_VISIBLE_DEVICES` when needed. Set
`DRY_RUN=1` to print training/evaluation commands without executing them.

Recommended order:

```bash
bash experiments/fleet_scalability_train.sh
bash experiments/fleet_scalability_eval.sh

bash experiments/objective_ablation_train.sh
bash experiments/objective_ablation_eval.sh

bash experiments/fixed_assignment_train.sh
bash experiments/fixed_assignment_eval.sh

bash experiments/search_convergence_eval.sh
```

Generate the Fleet Scalability and Search Convergence figures after evaluation:

```bash
bash experiments/fleet_scalability_plot.sh
bash experiments/search_convergence_plot.sh
```

Each plotting script automatically uses the latest run containing its summary
CSV and writes PNG and PDF files to that run's `figures/` directory. Fleet
Scalability is rendered as one four-panel figure. Search Convergence preserves
the full raw-value scale and adds an inset showing percentage improvement after
step 100 so that long-tail convergence remains visible. Set
`RUN_DIR` to plot a specific run, or override `PLOT_FORMATS` (`png,pdf,svg`)
and `PLOT_DPI` as needed. Plotting only reads the existing summaries and never
runs training or evaluation.

The Fleet evaluation must finish before the two ablation evaluation scripts,
because its N=20, K=2 proposed-objective result is reused through a manifest.
Training checkpoints remain under `outputs/`. Evaluation artifacts are written
under the corresponding experiment directory in `result_n2s/`.

Common overrides:

```bash
PYTHON=/path/to/rlor/bin/python CUDA_VISIBLE_DEVICES=2,3 \
  bash experiments/fleet_scalability_train.sh

VAL_SIZE=256 VAL_BATCH_SIZE=256 T_MAX=3000 \
  bash experiments/fleet_scalability_eval.sh

HISTORY_INTERVAL=100 bash experiments/search_convergence_eval.sh
```

To continue an interrupted evaluation in the same directory, set `RUN_DIR` to
that directory. Completed configurations with matching N, K, objective,
validation size, and search budget are skipped.
