#!/usr/bin/env python
"""
读取并检查 OSM 验证数据集的内容
"""
import pickle
import numpy as np

# 定义文件路径
file_path = 'datasets/osm_val.pkl'

print(f"正在读取文件: {file_path}\n")

# 使用 'rb' 模式打开文件，表示以二进制读取模式
with open(file_path, 'rb') as f:
    # 使用 pickle.load() 来反序列化文件内容
    data = pickle.load(f)

# 检查数据类型和基本信息
print("=" * 80)
print("数据集基本信息:")
print("=" * 80)
print(f"数据类型: {type(data)}")

if isinstance(data, (list, tuple)):
    print(f"总共有 {len(data)} 个数据样本\n")
    
    # 检查第一个样本的结构
    if len(data) > 0:
        first_sample = data[0]
        print("=" * 80)
        print("第一个样本的结构:")
        print("=" * 80)
        print(f"样本类型: {type(first_sample)}")
        
        if isinstance(first_sample, dict):
            print(f"包含的键: {list(first_sample.keys())}\n")
            
            # 详细检查每个字段
            for key, value in first_sample.items():
                print(f"字段 '{key}':")
                print(f"  类型: {type(value)}")
                
                if isinstance(value, np.ndarray):
                    print(f"  形状: {value.shape}")
                    print(f"  数据类型: {value.dtype}")
                    if value.size > 0:
                        print(f"  值范围: [{np.min(value):.6f}, {np.max(value):.6f}]")
                elif isinstance(value, (list, tuple)):
                    print(f"  长度: {len(value)}")
                    if len(value) > 0:
                        print(f"  第一个元素类型: {type(value[0])}")
                        if len(value) <= 5:
                            print(f"  内容: {value}")
                elif isinstance(value, dict):
                    print(f"  键的数量: {len(value)}")
                    if len(value) <= 5:
                        print(f"  内容: {value}")
                else:
                    print(f"  值: {value}")
                print()
        
        # 显示前两个样本的详细信息
        print("=" * 80)
        print("前两个样本的详细数据:")
        print("=" * 80)
        for i, sample in enumerate(data[:2]):
            print(f"\n{'*' * 40}")
            print(f"样本 {i+1}:")
            print(f"{'*' * 40}")
            
            if isinstance(sample, dict):
                # 显示坐标信息
                if 'coordinates' in sample:
                    coords = sample['coordinates']
                    print(f"\n坐标 (coordinates):")
                    print(f"  形状: {coords.shape}")
                    print(f"  前3个点: \n{coords[:3]}")
                
                # 显示距离矩阵信息
                if 'dist' in sample:
                    dist = sample['dist']
                    print(f"\n距离矩阵 (dist):")
                    print(f"  形状: {dist.shape}")
                    print(f"  前3x3子矩阵: \n{dist[:3, :3]}")
                
                # 显示配对信息
                if 'pairs' in sample:
                    pairs = sample['pairs']
                    print(f"\n配对信息 (pairs):")
                    print(f"  配对数量: {len(pairs)}")
                    print(f"  配对: {pairs}")
                
                # 显示其他字段
                if 'capacity' in sample:
                    print(f"\n容量 (capacity): {sample['capacity']}")
                
                if 'multi_start' in sample:
                    print(f"多起点 (multi_start): {sample['multi_start']}")
                
                if 'disable_geo_aug' in sample:
                    print(f"禁用地理增强 (disable_geo_aug): {sample['disable_geo_aug']}")
                
                if 'node2osmid' in sample:
                    print(f"\nOSM节点映射 (node2osmid):")
                    print(f"  类型: {type(sample['node2osmid'])}")
                    print(f"  节点数: {len(sample['node2osmid'])}")
                    if len(sample['node2osmid']) <= 5:
                        print(f"  内容: {sample['node2osmid']}")
                
                if 'path_lookup' in sample:
                    print(f"\n路径查找表 (path_lookup):")
                    print(f"  类型: {type(sample['path_lookup'])}")
                    if isinstance(sample['path_lookup'], dict):
                        print(f"  条目数: {len(sample['path_lookup'])}")
                        # 显示一个示例路径
                        for key in list(sample['path_lookup'].keys())[:1]:
                            path = sample['path_lookup'][key]
                            print(f"  示例路径 {key}: 长度={len(path)}, 前5个节点={path[:5]}")
            else:
                print(sample)
    
    # 统计信息
    print("\n" + "=" * 80)
    print("数据集统计信息:")
    print("=" * 80)
    
    if len(data) > 0 and isinstance(data[0], dict):
        # 统计坐标维度
        if 'coordinates' in data[0]:
            shapes = [sample['coordinates'].shape for sample in data]
            print(f"坐标形状: {set(shapes)}")
        
        # 统计配对数量
        if 'pairs' in data[0]:
            pair_counts = [len(sample['pairs']) for sample in data]
            print(f"配对数量范围: {min(pair_counts)} - {max(pair_counts)}")
            print(f"平均配对数: {np.mean(pair_counts):.2f}")

else:
    # 如果数据不是列表或元组，则打印数据类型并尝试打印整个数据
    print(f"数据类型为: {type(data)}")
    print("数据内容（或部分内容）:")
    print(data)

print("\n" + "=" * 80)
print("检查完成!")
print("=" * 80)
