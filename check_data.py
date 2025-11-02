#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速查看训练数据集的字段和节点信息

Usage:
    python check_data.py --dataset datasets/osm_train_20.pkl
"""
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description='Quick check OSM training dataset')
    parser.add_argument('--dataset', type=str, default='datasets/osm_train_20.pkl',
                        help='Path to dataset pickle file')
    args = parser.parse_args()

    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
    sample = dataset[0]
    print(f"字段列表: {list(sample.keys())}")
    if 'coordinates' in sample:
        coords = sample['coordinates']
        num_nodes = coords.shape[0]
        num_pairs = (num_nodes - 1) // 2
        print(f"节点总数: {num_nodes}")
        print(f"Depot: 1")
        print(f"Pickup/Delivery对数: {num_pairs}")

if __name__ == "__main__":
    main()
