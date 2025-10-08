#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校验 PDTSP_OSM 定义是否正确：
1) 能在线从 OSM 构建训练样本（50节点=25对+仓库）；
2) batch 中包含 coordinates / dist / path_lookup / node2osmid / pairs 等字段；
3) 使用 PDTSP_OSM.get_costs 以“路网距离”计算路径长度；
4) 在 OSM 地图上绘制一条示意路线（顺序环游）并保存 PNG。
"""

import argparse
import os
import torch

# === 按你之前新增的文件路径导入 ===
from problems.problem_pdtsp_osm import PDTSP_OSM
from data.osm_pdp_dataset import OSMOnlinePDPSDataset
from data.osm_utils import build_drive_graph
from utils.viz_osmnx import plot_solution_on_osm

def make_successor_table_from_seq(seq):
    """
    由访问序列（如 [0, 1, 2, ..., N, 0]）构造“后继表”rec（大小 N+1），
    使 PDTSP_OSM.get_costs 可计算该路线的总长度。
    """
    n1 = len(seq) - 1  # 最后一个等于首点 0
    rec = torch.zeros(n1, dtype=torch.long)
    for i in range(n1):
        u = seq[i]
        v = seq[i + 1]
        rec[u] = v
    return rec.unsqueeze(0)  # [B=1, N+1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", type=str, default="Boca Raton, Florida, USA")
    ap.add_argument("--graph_size", type=int, default=50)   # 50 节点 = 25 对
    ap.add_argument("--capacity", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out_png", type=str, default="figs/pdtsp_osm_check.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    # 1) 在线训练样本（从 OSM 生成）
    ds = OSMOnlinePDPSDataset(
        place=args.place,
        graph_size=args.graph_size,
        capacity=args.capacity,
        size=1,                      # 只要一个样本校验
        seed=args.seed,
        disable_geo_aug=True,        # 禁用旋转/翻转
        multi_start=4,
        pair_permute_aug=True        # 节点重标号增强
    )
    sample = ds[0]

    # 2) 基本字段检查
    required_keys = ["coordinates", "dist", "path_lookup", "node2osmid", "pairs", "capacity"]
    print("[check] sample keys:", list(sample.keys()))
    for k in required_keys:
        assert k in sample, f"missing key in sample: {k}"

    B = 1
    coords = sample["coordinates"]                     # [N+1, 2]
    dist = sample["dist"]                              # [N+1, N+1]
    n1 = coords.shape[0]                               # N+1
    print(f"[check] coordinates shape: {tuple(coords.shape)}  (expect {(args.graph_size+1, 2)})")
    print(f"[check] dist shape        : {tuple(dist.shape)}    (expect {(args.graph_size+1, args.graph_size+1)})")
    assert n1 == args.graph_size + 1, "graph_size 与样本节点数不一致"

    # 3) 用“顺序环游”构造一条示意路线，并计算真实路网长度
    seq = list(range(n1)) + [0]       # [0,1,2,...,N,0]
    rec = make_successor_table_from_seq(seq)  # [1, N+1]

    # get_costs 需要 batch 维度；把 sample 包装成 batch（B=1）
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in sample.items()}
    # 代价使用 batch['dist']（真实路网长度）
    cost = PDTSP_OSM.get_costs(batch, rec).item()
    print(f"[check] route length on real map: {cost:.1f} (units = meters if dist is 'length')")

    # 4) 在真实地图上画出这条示意路线
    G = build_drive_graph(args.place)
    # plot_solution_on_osm 期望的是单样本的 dict，所以把张量去掉 batch 维度
    single_sample = {k: (v[0] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    plot_solution_on_osm(G, single_sample, seq, args.out_png)
    print(f"[check] saved map visualization to: {args.out_png}")

    print("[check] PDTSP_OSM is loadable and working (data fields, cost, plotting all OK).")

if __name__ == "__main__":
    main()
