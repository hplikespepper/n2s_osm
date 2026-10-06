#!/usr/bin/env python3
"""Small, dependency-free helpers shared by the manual experiment scripts."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def result_files(result_dir):
    return sorted(Path(result_dir).glob('*_results_*.json'))


def result_matches(path, args):
    data = load_json(path)
    return (
        data.get('graph_size') == args.graph_size
        and data.get('num_vehicles') == args.num_vehicles
        and data.get('objective') == args.objective
        and data.get('val_size') == args.val_size
        and data.get('T_max') == args.t_max
        and len(data.get('instances', [])) == args.val_size
    )


def result_complete(args):
    for path in result_files(args.result_dir):
        if result_matches(path, args):
            print(path)
            return 0
    return 1


def result_file_valid(args):
    if result_matches(Path(args.result_file), args):
        print(args.result_file)
        return 0
    return 1


def training_state(args):
    run_dir = Path(args.run_dir)
    args_path = run_dir / 'args.json'
    if not args_path.is_file():
        print('mismatch')
        return 2
    data = load_json(args_path)
    expected = {
        'problem': args.problem,
        'graph_size': args.graph_size,
        'num_vehicles': args.num_vehicles,
        'objective': args.objective,
        'epoch_end': args.epoch_end,
    }
    if any(data.get(key) != value for key, value in expected.items()):
        print('mismatch')
        return 2

    record_path = run_dir / 'training_time.json'
    if record_path.is_file() and load_json(record_path).get('status') == 'completed':
        print('complete')
        return 0
    epochs = []
    for path in run_dir.glob('epoch-*.pt'):
        try:
            epochs.append(int(path.stem.split('-', 1)[1]))
        except (IndexError, ValueError):
            pass
    if epochs and max(epochs) >= args.epoch_end - 1:
        print('complete')
        return 0
    print('incomplete')
    return 1


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(args):
    output = Path(args.output)
    source = Path(args.source).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'role': args.role,
        'source_result': os.path.relpath(source, output.parent.resolve()),
        'source_sha256': sha256(source),
    }
    with open(output, 'w') as f:
        json.dump(payload, f, indent=2)
    print(output)
    return 0


def parse_entries(values):
    entries = []
    for value in values:
        if '=' not in value:
            raise ValueError(f'Expected LABEL=PATH, got: {value}')
        label, path = value.split('=', 1)
        entries.append((label, Path(path)))
    return entries


def stats(values):
    values = [float(x) for x in values]
    count = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if count > 1 else 0.0
    return count, mean, std, std / math.sqrt(count)


def summarize_standard(mode, entries, output):
    rows = []
    for label, path in entries:
        data = load_json(path)
        instances = data['instances']
        k = int(data['num_vehicles'])
        metrics = {
            'distance': [x['best_distance_cost'] for x in instances],
            'makespan': [x['best_makespan_cost'] for x in instances],
            'objective': [x['best_cost'] for x in instances],
            'proposed': [x.get('proposed_cost', x['best_distance_cost'] + (k - 1) * x['best_makespan_cost']) for x in instances],
            'active_vehicles': [x.get('active_vehicles', sum(v > 1e-8 for v in x['best_vehicle_distance_costs'])) for x in instances],
            'utilization': [x.get('vehicle_utilization', sum(v > 1e-8 for v in x['best_vehicle_distance_costs']) / k) for x in instances],
        }
        row = {
            'experiment': mode,
            'variant': label,
            'graph_size': data['graph_size'],
            'num_vehicles': k,
            'objective_name': data.get('objective', instances[0].get('objective', 'legacy')),
            'result_file': str(path),
        }
        for name, values in metrics.items():
            count, mean, std, se = stats(values)
            row['instances'] = count
            row[f'mean_{name}'] = mean
            row[f'std_{name}'] = std
            row[f'se_{name}'] = se
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_convergence(entries, output):
    rows = []
    for label, path in entries:
        data = load_json(path)
        instances = data['instances']
        expected_steps = data.get('component_history_steps')
        if not expected_steps:
            raise ValueError(f'No component history in {path}')
        for step_index, step in enumerate(expected_steps):
            metrics = {
                'objective': [x['component_history']['best_cost'][step_index] for x in instances],
                'distance': [x['component_history']['best_distance_cost'][step_index] for x in instances],
                'makespan': [x['component_history']['best_makespan_cost'][step_index] for x in instances],
            }
            row = {
                'variant': label,
                'graph_size': data['graph_size'],
                'num_vehicles': data['num_vehicles'],
                'objective_name': data['objective'],
                'step': step,
                'result_file': str(path),
            }
            for name, values in metrics.items():
                count, mean, std, se = stats(values)
                row['instances'] = count
                row[f'mean_{name}'] = mean
                row[f'std_{name}'] = std
                row[f'se_{name}'] = se
            rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args):
    entries = parse_entries(args.entry)
    if not entries:
        raise ValueError('At least one --entry is required')
    output = Path(args.output)
    if args.mode == 'convergence':
        summarize_convergence(entries, output)
    else:
        summarize_standard(args.mode, entries, output)
    print(output)
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)

    complete = sub.add_parser('result-complete')
    complete.add_argument('--result-dir', required=True)
    complete.add_argument('--graph-size', type=int, required=True)
    complete.add_argument('--num-vehicles', type=int, required=True)
    complete.add_argument('--objective', required=True)
    complete.add_argument('--val-size', type=int, required=True)
    complete.add_argument('--t-max', type=int, required=True)
    complete.set_defaults(func=result_complete)

    valid = sub.add_parser('result-file-valid')
    valid.add_argument('--result-file', required=True)
    valid.add_argument('--graph-size', type=int, required=True)
    valid.add_argument('--num-vehicles', type=int, required=True)
    valid.add_argument('--objective', required=True)
    valid.add_argument('--val-size', type=int, required=True)
    valid.add_argument('--t-max', type=int, required=True)
    valid.set_defaults(func=result_file_valid)

    training = sub.add_parser('training-state')
    training.add_argument('--run-dir', required=True)
    training.add_argument('--problem', required=True)
    training.add_argument('--graph-size', type=int, required=True)
    training.add_argument('--num-vehicles', type=int, required=True)
    training.add_argument('--objective', required=True)
    training.add_argument('--epoch-end', type=int, default=200)
    training.set_defaults(func=training_state)

    manifest = sub.add_parser('write-manifest')
    manifest.add_argument('--source', required=True)
    manifest.add_argument('--output', required=True)
    manifest.add_argument('--role', required=True)
    manifest.set_defaults(func=write_manifest)

    summary = sub.add_parser('summarize')
    summary.add_argument('--mode', choices=['fleet', 'objective', 'fixed', 'convergence'], required=True)
    summary.add_argument('--entry', action='append', default=[])
    summary.add_argument('--output', required=True)
    summary.set_defaults(func=summarize)
    return parser


if __name__ == '__main__':
    try:
        parsed = build_parser().parse_args()
        sys.exit(parsed.func(parsed))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
