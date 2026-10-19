#!/usr/bin/env python
"""Draw and describe the largest strongly connected graph selected for training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import osmnx as ox

HERE = Path(__file__).resolve().parent
PLACE = 'Boca Raton, Florida, USA'


def direction(data):
    value = data.get('oneway')
    values = value if isinstance(value, (list, tuple)) else [value]
    normalized = []
    for item in values:
        if isinstance(item, (bool, np.bool_)):
            normalized.append(bool(item))
        elif str(item).lower() in ('true', 'yes', '1', '-1'):
            normalized.append(True)
        elif str(item).lower() in ('false', 'no', '0'):
            normalized.append(False)
        else:
            return 'unknown'
    return 'oneway' if all(normalized) else 'twoway' if not any(normalized) else 'mixed'


def edge_summary(graph):
    physical = ox.convert.to_undirected(graph)
    counts = Counter(direction(data) for _, _, data in graph.edges(data=True))
    segments = Counter(direction(data) for _, _, data in physical.edges(data=True))
    return physical, counts, segments


def describe(graph, heading):
    physical, directed, segments = edge_summary(graph)
    out = [heading, '-' * 72,
           f'节点总数: {graph.number_of_nodes()}',
           f'有向边总数（MultiDiGraph arcs，含平行边和自环）: {graph.number_of_edges()}',
           f'  单向道路对应有向边: {directed["oneway"]}',
           f'  双向道路对应有向边: {directed["twoway"]}',
           f'  单/双向混合属性边: {directed["mixed"]}',
           f'  方向属性未知边: {directed["unknown"]}',
           f'去除反向重复后的道路段总数: {physical.number_of_edges()}',
           f'  单向道路段: {segments["oneway"]}',
           f'  双向道路段: {segments["twoway"]}',
           f'  混合方向道路段: {segments["mixed"]}',
           f'  未知方向道路段: {segments["unknown"]}',
           f'自环有向边数: {nx.number_of_selfloops(graph)}',
           f'强连通分量数: {nx.number_strongly_connected_components(graph)}',
           f'弱连通分量数: {nx.number_weakly_connected_components(graph)}',
           f'有向图密度: {nx.density(graph):.8f}',
           f'道路段总长度（反向去重，米）: {sum(d["length"] for _, _, d in physical.edges(data=True)):.3f}',
           f'有向边总长度（双向道路通常重复计入，米）: {sum(d["length"] for _, _, d in graph.edges(data=True)):.3f}']
    lengths = [d['length'] for _, _, d in physical.edges(data=True)]
    if lengths:
        out.append(f'道路段长度最小/平均/中位数/最大（米）: {min(lengths):.3f} / {np.mean(lengths):.3f} / {np.median(lengths):.3f} / {max(lengths):.3f}')
    degrees = np.array([d for _, d in graph.degree()])
    out.append(f'节点有向总度最小/平均/最大: {degrees.min()} / {degrees.mean():.4f} / {degrees.max()}')
    out.append(f'零入度节点数: {sum(d == 0 for _, d in graph.in_degree())}')
    out.append(f'零出度节点数: {sum(d == 0 for _, d in graph.out_degree())}')
    for attr in ('highway', 'name'):
        counter = Counter()
        for _, _, data in physical.edges(data=True):
            values = data.get(attr, '(未标注)')
            for value in values if isinstance(values, list) else [values]:
                counter[str(value)] += 1
        out.append(f'按 {attr} 属性统计道路段（多值边每个属性值各计一次）:')
        out.extend(f'  {key}: {value}' for key, value in counter.most_common())
    out.append('OSMnx basic_stats（保留官方指标字段名，长度单位为米）:')
    out.extend(f'  {key}: {value}' for key, value in ox.basic_stats(graph).items())
    return out


def draw_graph(graph, output, title):
    palette = {'oneway': '#d95f02', 'twoway': '#377eb8', 'mixed': '#984ea3', 'unknown': '#888888'}
    colors = [palette[direction(d)] for _, _, d in graph.edges(data=True)]
    fig, ax = ox.plot_graph(graph, figsize=(14, 14), bgcolor='white',
                            node_color='#333333', node_size=2,
                            edge_color=colors, edge_linewidth=.7,
                            show=False, close=False)
    ax.set_title(title, fontsize=14, weight='bold', pad=12)
    ax.legend(handles=[Line2D([0], [0], color=palette['oneway'], label='One-way road'),
                       Line2D([0], [0], color=palette['twoway'], label='Two-way road')],
              loc='lower right', fontsize=11)
    fig.savefig(output, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--place', default=PLACE)
    parser.add_argument('--refresh', action='store_true', help='Rebuild graph from OSMnx responses (cached responses reused)')
    args = parser.parse_args()
    cache = HERE / 'cache'
    cache.mkdir(exist_ok=True)
    # Seed the self-contained download cache with the project's existing OSM
    # responses, so the historical map can be reconstructed without new queries.
    old_cache = HERE.parent / 'cache'
    for source in old_cache.glob('*.json'):
        target = cache / source.name
        if not target.exists():
            shutil.copy2(source, target)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache)
    graph_path = HERE / 'boca_raton_drive_all.graphml'
    if graph_path.exists() and not args.refresh:
        graph = ox.load_graphml(graph_path)
        if graph.graph.get('analysis_place') != args.place:
            raise ValueError('Snapshot place differs; use --refresh for a different place')
        print(f'Loaded fixed graph: {graph_path}', flush=True)
    else:
        print(f'Building complete drivable network for {args.place}', flush=True)
        graph = ox.graph_from_place(args.place, network_type='drive', simplify=True, retain_all=True)
        graph.graph['analysis_place'] = args.place
        ox.save_graphml(graph, graph_path)
    # Match training selection: largest weak component, then largest SCC.
    weak_nodes = max(nx.weakly_connected_components(graph), key=len)
    weak_graph = graph.subgraph(weak_nodes)
    strong_nodes = max(nx.strongly_connected_components(weak_graph), key=len)
    graph = graph.subgraph(strong_nodes).copy()
    assert nx.is_strongly_connected(graph)
    ox.save_graphml(graph, HERE / 'boca_raton_drive_largest_scc.graphml')
    xs = [d['x'] for _, d in graph.nodes(data=True)]
    ys = [d['y'] for _, d in graph.nodes(data=True)]
    lines = ['Boca Raton 训练/评估使用的最大强连通驾驶路网统计', '=' * 72,
             f'地点: {args.place}', f'报告生成时间（UTC）: {datetime.now(timezone.utc).isoformat()}',
             f'OSMnx: {ox.__version__}; NetworkX: {nx.__version__}',
             '统计对象: 与训练构图相同规则筛选出的最大强连通分量。',
             '筛选规则: 最大弱连通分量 -> 其中最大强连通分量。',
             '以下全部节点、边、方向、长度和道路属性统计仅针对筛选后的图。',
             '图类型: 有向多重图 MultiDiGraph; network_type=drive; simplify=True。',
             '节点是拓扑简化后保留的交叉口/道路端点等，不是 OSM 原始形状点总数。',
             '有向边按可行驶方向计数；双向道路通常对应两条反向有向边。',
             '道路段使用 OSMnx to_undirected 按 OSM way/几何去除反向重复；平行独立道路仍保留。',
             '单/双向按 OSMnx oneway 属性统计，不仅依据是否存在任意反向边。',
             '长度指标保留缓存简化图的原始 length，不重复调用 add_edge_lengths。',
             '本报告未修改训练图或数据；历史 pickle 的原始图版本需以其配套快照确认。',
             '缓存可能来自历史 OSM 响应；报告时间不是地图数据更新时间。',
             f'经度范围: {min(xs):.8f} 至 {max(xs):.8f}',
             f'纬度范围: {min(ys):.8f} 至 {max(ys):.8f}', '']
    lines += describe(graph, '最大强连通分量统计')
    lines += ['', '完整节点和边属性、OSM ID、道路几何见 boca_raton_drive_largest_scc.graphml。']
    report = HERE / 'boca_raton_network_info.txt'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    title = (f'Boca Raton: Largest Strongly Connected Drivable Network\n'
             f'{len(graph):,} nodes | {graph.number_of_edges():,} directed edges')
    draw_graph(graph, HERE / 'boca_raton_road_network.png', title)
    # Retain the previous explicit SCC filename as an identical convenience copy.
    shutil.copy2(HERE / 'boca_raton_road_network.png', HERE / 'boca_raton_largest_scc.png')
    print(f'Training graph: {len(graph)} nodes, {graph.number_of_edges()} directed edges; SCCs: 1', flush=True)
    print(f'Report: {report}', flush=True)


if __name__ == '__main__':
    main()
