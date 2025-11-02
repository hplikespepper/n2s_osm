#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查验证集中保存的最短路径是否正确。
可视化第一个实例中节点0到节点1的预计算最短路径。

Usage:
    python check_vis.py
"""

import pickle
import matplotlib.pyplot as plt
import osmnx as ox
from data.osm_utils import build_drive_graph

# 配置
VAL_DATASET_FILE = "./datasets/osm_val_20.pkl"
OSM_PLACE = "Boca Raton, Florida, USA"
OUTPUT_FILE = "check_path_0_to_1.png"


def visualize_saved_path(G, sample, node_from, node_to, output_path):
    """
    可视化验证集中保存的从node_from到node_to的最短路径。
    
    Args:
        G: OSMnx图
        sample: 数据集样本，包含path_lookup和node2osmid
        node_from: 起始节点索引
        node_to: 目标节点索引
        output_path: 输出文件路径
    """
    print(f"\n检查节点 {node_from} 到节点 {node_to} 的路径:")
    
    # 获取OSM节点ID
    osm_from = sample['node2osmid'][node_from]
    osm_to = sample['node2osmid'][node_to]
    print(f"  节点 {node_from} 对应 OSM ID: {osm_from}")
    print(f"  节点 {node_to} 对应 OSM ID: {osm_to}")
    
    # 从path_lookup中获取保存的路径
    if (node_from, node_to) in sample['path_lookup']:
        saved_path = sample['path_lookup'][(node_from, node_to)]
        print(f"  保存的路径长度: {len(saved_path)} 个OSM节点")
        print(f"  保存的路径: {saved_path}")
        
        # 检查路径的起点和终点
        if saved_path[0] != osm_from:
            print(f"  ⚠️  警告: 路径起点 {saved_path[0]} 与预期 {osm_from} 不匹配!")
        if saved_path[-1] != osm_to:
            print(f"  ⚠️  警告: 路径终点 {saved_path[-1]} 与预期 {osm_to} 不匹配!")
    else:
        print(f"  ⚠️  错误: 在path_lookup中找不到路径 ({node_from}, {node_to})!")
        return
    
    # 获取保存的距离
    dist_matrix = sample['dist']
    saved_distance = dist_matrix[node_from, node_to]
    print(f"  保存的距离: {saved_distance:.2f}")
    
    # 重新计算最短路径进行对比
    try:
        computed_path = ox.distance.shortest_path(G, osm_from, osm_to, weight='length')
        print(f"  重新计算的路径长度: {len(computed_path)} 个OSM节点")
        
        # 计算路径的实际长度
        path_length = 0
        for i in range(len(computed_path) - 1):
            u, v = computed_path[i], computed_path[i+1]
            # 在多重图中找到边
            if G.has_edge(u, v):
                edge_data = G[u][v]
                if isinstance(edge_data, dict):
                    # 单条边
                    if 'length' in edge_data:
                        path_length += edge_data['length']
                    else:
                        # 多条边的情况，取第一条
                        path_length += list(edge_data.values())[0]['length']
                else:
                    # 多重边，取第一条
                    path_length += list(edge_data.values())[0]['length']
        
        print(f"  重新计算的距离: {path_length:.2f}")
        
        # 比较路径
        if saved_path == computed_path:
            print(f"  ✓ 路径一致!")
        else:
            print(f"  ⚠️  路径不一致!")
            print(f"     保存路径与重算路径的差异:")
            print(f"     - 保存路径首尾: {saved_path[0]} -> ... -> {saved_path[-1]}")
            print(f"     - 重算路径首尾: {computed_path[0]} -> ... -> {computed_path[-1]}")
        
        # 比较距离
        if abs(saved_distance - path_length) < 0.01:
            print(f"  ✓ 距离一致!")
        else:
            print(f"  ⚠️  距离不一致! 差异: {abs(saved_distance - path_length):.2f}")
            
    except Exception as e:
        print(f"  ⚠️  重新计算路径时出错: {e}")
        computed_path = None
    
    # 可视化
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 绘制基础道路网络
    ox.plot_graph(G, node_size=2, edge_color='lightgray', edge_linewidth=0.5,
                  show=False, close=False, ax=ax)
    
    # 绘制保存的路径（红色）
    xs_saved = [G.nodes[n]['x'] for n in saved_path]
    ys_saved = [G.nodes[n]['y'] for n in saved_path]
    ax.plot(xs_saved, ys_saved, linewidth=4, alpha=0.7, color='red', 
            label=f'保存的路径 (长度={saved_distance:.2f})', zorder=3)
    
    # 如果重新计算的路径存在且不同，也绘制出来（蓝色虚线）
    if computed_path is not None and saved_path != computed_path:
        xs_computed = [G.nodes[n]['x'] for n in computed_path]
        ys_computed = [G.nodes[n]['y'] for n in computed_path]
        ax.plot(xs_computed, ys_computed, linewidth=3, alpha=0.6, color='blue',
                linestyle='--', label=f'重新计算的路径 (长度={path_length:.2f})', zorder=2)
    
    # 标记起点（绿色星形）
    start_x = G.nodes[osm_from]['x']
    start_y = G.nodes[osm_from]['y']
    ax.scatter(start_x, start_y, s=400, c='green', marker='*', 
               edgecolors='black', linewidths=2, label=f'起点 (节点{node_from})', zorder=5)
    
    # 标记终点（红色星形）
    end_x = G.nodes[osm_to]['x']
    end_y = G.nodes[osm_to]['y']
    ax.scatter(end_x, end_y, s=400, c='red', marker='*', 
               edgecolors='black', linewidths=2, label=f'终点 (节点{node_to})', zorder=5)
    
    # 添加标题和图例
    title = f'验证集第一个实例: 节点{node_from}到节点{node_to}的路径检查\n'
    title += f'OSM ID: {osm_from} -> {osm_to}'
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=11)
    
    # 保存图片
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n✓ 可视化已保存到: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("检查验证集中保存的最短路径")
    print("=" * 60)
    
    # 加载验证数据集
    print(f"\n加载验证数据集: {VAL_DATASET_FILE}")
    with open(VAL_DATASET_FILE, 'rb') as f:
        val_dataset = pickle.load(f)
    
    print(f"  数据集大小: {len(val_dataset)} 个实例")
    
    # 获取第一个实例
    sample = val_dataset[0]
    print(f"\n第一个实例信息:")
    print(f"  节点数量: {len(sample['node2osmid'])}")
    print(f"  坐标形状: {sample['coordinates'].shape if 'coordinates' in sample else 'N/A'}")
    print(f"  距离矩阵形状: {sample['dist'].shape if 'dist' in sample else 'N/A'}")
    print(f"  路径查找表大小: {len(sample['path_lookup']) if 'path_lookup' in sample else 0}")
    
    # 加载OSM图
    print(f"\n加载OSM道路网络: {OSM_PLACE}")
    G = build_drive_graph(OSM_PLACE)
    print(f"  图节点数: {len(G.nodes)}")
    print(f"  图边数: {len(G.edges)}")
    
    # 可视化节点0到节点1的路径
    visualize_saved_path(G, sample, node_from=0, node_to=1, output_path=OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("检查完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
