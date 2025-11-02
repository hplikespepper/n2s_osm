#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将结果文件中的邻接表形式路径转换为实际访问顺序。

Usage:
    python get_path_order.py --results results/pdtsp_results_pt_199.json --index 0
"""

import argparse
import json


def adjacency_list_to_path(adj_list):
    """
    将邻接表形式的路径转换为节点访问顺序。
    
    Args:
        adj_list: 邻接表，adj_list[i] 表示节点i指向的下一个节点
    
    Returns:
        path: 节点访问顺序列表
    """
    if not adj_list:
        return []
    
    path = [0]  # 从depot (节点0) 开始
    current = 0
    visited = set([0])
    
    while len(path) <= len(adj_list):
        next_node = adj_list[current]
        if next_node in visited and next_node == 0:
            # 回到depot，路径结束
            path.append(0)
            break
        if next_node in visited:
            # 检测到环路
            print(f"Warning: Cycle detected at node {next_node}")
            break
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    
    return path


def main(args):
    # 加载结果文件
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # 检查实例索引
    instance_idx = args.index
    if instance_idx >= len(results['instances']):
        print(f"Error: Instance index {instance_idx} out of range.")
        print(f"Available instances: 0-{len(results['instances'])-1}")
        return
    
    # 获取实例数据并转换
    instance = results['instances'][instance_idx]
    adj_list = instance['best_path']
    path = adjacency_list_to_path(adj_list)
    
    # 输出结果
    print(f"Instance {instance_idx}:")
    print(f"Adjacency list: {adj_list}")
    print(f"Path order: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将邻接表形式的路径转换为实际访问顺序')
    parser.add_argument('--results', type=str, required=True,
                        help='结果JSON文件路径')
    parser.add_argument('--index', type=int, required=True,
                        help='要转换的实例索引')
    
    args = parser.parse_args()
    main(args)
