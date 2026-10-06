# LKH-3 PDPTW baseline for MVN2S

`lkh3_baseline.py` reads the same `datasets/pdp_*.pkl` files as MVN2S. It
converts each instance to PDPTW, invokes the unmodified LKH-3 executable with
several seeds and diversified route-length limits, and selects the candidate
minimizing the MVN2S objective:

```text
total_distance + (num_vehicles - 1) * makespan
```

LKH itself minimizes total distance. Every start receives the same deterministic
balanced feasible initial tour. The first start is unconstrained; later starts
bound every route to different fractions of the initial makespan so that LKH
also produces balanced multi-vehicle candidates. The
makespan-aware selection happens only between valid LKH candidates, so results
should be described as
"LKH-3 PDPTW multi-start with MVN2S-objective reranking."

## Quick check

From `n2s_mv` in the environment used for MVN2S:

```bash
python -m unittest discover -s tests -p test_lkh3_baseline.py

python lkh3_baseline.py \
  --lkh_path ../LKH3/LKH-3.0.14/LKH \
  --val_dataset ./datasets/pdp_20.pkl \
  --graph_size 20 \
  --num_vehicles 2 \
  --val_size 4 \
  --max_trials 5000 \
  --num_starts 4 \
  --start_workers 4 \
  --num_workers 1 \
  --output result_temp_lkh3.json
```

`--max_trials` maps directly to LKH's `MAX_TRIALS` for every independent
start. No `TIME_LIMIT` or `TOTAL_TIME_LIMIT` is written. `solve_time` is an
observed duration only and also includes process startup, conversion, parsing,
feasibility checks, and MVN2S cost calculation.

## Full comparison batch

The batch script runs five configurations in this order: 50 nodes/5 vehicles,
100 nodes/2 vehicles, 100 nodes/5 vehicles, 20 nodes/2 vehicles, and finally
50 nodes/2 vehicles:

```bash
VAL_SIZE=256 MAX_TRIALS_LIST="5000 10000" \
NUM_STARTS=4 START_WORKERS=4 NUM_WORKERS=16 \
bash mv_lkh3.sh
```

`NUM_STARTS` is the number of candidates generated for every instance.
`START_WORKERS` controls concurrent starts inside one instance, while
`NUM_WORKERS` controls concurrent dataset instances. The default configuration
therefore runs at most `4 * 16 = 64` LKH processes at once. The four route-limit
settings are unconstrained, 0.9, 1.0, and 1.1 times the deterministic initial
route limit.

`CONFIGS` uses `nodes:vehicles` entries and can also select one configuration,
for example `CONFIGS="50:5"`. Set `RUN_DIR` only when intentionally adding to
an existing result directory; otherwise the script creates a fresh run.

Results are written below `result_lkh3/run_*`. Each JSON uses the same
`best_cost`, `best_distance_cost`, `best_makespan_cost`, `best_rec`,
`vehicle_routes`, and `coordinates` fields as the MVN2S results. It also stores
`candidate_results` for every instance, including each start's seed, route
limit, objective, distance, makespan, elapsed time, success/failure state, and
whether it was selected.
