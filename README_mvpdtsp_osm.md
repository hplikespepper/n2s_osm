# MVPDP on OSM roads (`mvpdtsp_osm`)

This variant uses one physical depot and K homogeneous vehicles. It reuses the
existing multi-vehicle successor representation, pair relocation, same-vehicle
pickup/delivery and pickup-before-delivery constraints. Capacity limits are **not**
implemented in this version: `--capacity` preserves the existing data/config field
and does not restrict route loads.

## Dataset compatibility

Existing OSM pickle files generated with `OSMOnlinePDPSDataset` remain unchanged:
`[depot, pickups..., deliveries...]`, with `coordinates`, directed `dist`,
`node2osmid`, `pairs`, and optional `path_lookup`. `--graph_size` is the number of
customers, excluding the depot, and must match the stored samples.

`--num_vehicles K` expands each sample in memory to
`[depot_0, ..., depot_K-1, pickups..., deliveries...]`. All depot copies refer to
the same physical OSM node. Coordinates, distance rows/columns and path indices
are adapted together. Thus one file supports different vehicle counts without
recomputing shortest paths. Fixed datasets are adapted lazily on sample access;
original arrays and dictionaries are never mutated. Explicit missing files fail.
`OSMMVPDPDataset` also supports `offset` and `num_samples`, as well as online data
creation using the original strongly connected road graph generator.

Cost uses only directed road distances. Empty vehicles have zero cost. Objectives:

- `--objective distance` (default): total distance D.
- `--objective only_makespan`: longest vehicle route distance M.
- `--objective proposed`: D + (K - 1) M.

M is a distance, not travel time. Distance units are inherited from the stored
matrix (normally meters for OSM edge length); distances and rewards are not
rescaled. Random and greedy initialization retain the original multi-vehicle
implementation; greedy still orders by Euclidean coordinate distance.

## Training and evaluation

Use the existing project environment with PyTorch, OSMnx, NumPy, pyproj,
NetworkX, tqdm and tensorboard_logger. Rendering dependencies such as Contextily
are not needed. Do not install these in conda base.

```bash
cd n2s_osm_marl
python run.py --problem mvpdtsp_osm --graph_size 20 --num_vehicles 3 \
  --train_dataset datasets/osm_train_20.pkl --val_dataset datasets/osm_val_20.pkl \
  --objective proposed --init_val_met random

python run.py --problem mvpdtsp_osm --graph_size 20 --num_vehicles 3 \
  --eval_only --load_path outputs/<run-directory>/epoch-0.pt \
  --val_dataset datasets/osm_val_20.pkl --objective proposed --val_m 2
```

Omit `--train_dataset` to generate training samples online using `--osm_place`.
Pre-generated data is recommended: road shortest-path preprocessing is expensive.
Online epoch seeds vary as in the existing single-vehicle OSM training flow.
Actor/Critic architecture, coordinate feature encoding, rollout coordinate
augmentation and PPO hyperparameters are unchanged. Rollout expansion also
expands the road matrix. A checkpoint used for evaluation must match network
configuration and sequence length; this extension does not guarantee trained
Euclidean checkpoints have good road-network performance.

## Output and reproducibility

Evaluation writes `mvpdtsp_osm_results_<timestamp>.json` and a terminal summary.
Results include vehicle routes, successor table (`best_rec`), objective, total
road distance, longest route distance, per-vehicle road distances, initial cost
components, and optional best-so-far component history.

`source_dataset` identifies the source pickle. Each result's `dataset_index`
identifies the sample in that file; `original_total_nodes` includes the single
physical depot and `adapted_total_nodes` includes K logical depot copies. Full
road matrices and path dictionaries are not embedded in JSON. Preserve the source
pickle and its corresponding GraphML for later road-route reconstruction.
Unlike old Euclidean exports, OSM exports do not label road distance as vehicle
completion time. Distributed validation transports road matrices alongside
solutions, restores dataset order and removes sampler-padding duplicates.

No existing single-vehicle data, training configuration or road-length
preprocessing is changed. Multi-vehicle real-map rendering and OR-Tools road
baselines are separate follow-up work.

## Offline verification

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_mvpdtsp_osm*.py' -v
```

Tests cover K=1/2/3 adaptation, nonmutation, directed costs and all objectives,
empty vehicles, unchanged initialization, cross-vehicle pair moves, malformed
inputs, offset selection, distributed ordering/padding, augmented rollout and
JSON cost consistency. A two-process CPU Gloo simulation checks both collective
and file-shard aggregation with different sample matrices and sampler padding.
Tests use synthetic distances and do not request maps.
