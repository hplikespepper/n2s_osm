# Boca Raton 最大强连通驾驶路网统计

统计对象仅为训练/评估采样所用筛选规则得到的最大强连通分量：
先保留最大弱连通分量，再选其中最大的强连通分量。

从项目根目录运行（使用 rlor 环境）：

```bash
bash n2s_osm_marl/boca_raton_imfo/run.sh
```

结果文件：

- `boca_raton_road_network.png`：最大强连通驾驶路网，橙色单向、蓝色双向。
- `boca_raton_largest_scc.png`：同一图片的显式 SCC 文件名副本。
- `boca_raton_network_info.txt`：仅针对最大强连通分量的 UTF-8 中文统计报告。
- `boca_raton_drive_largest_scc.graphml`：上述图的完整节点、边、OSM ID 和几何属性。

当前最大强连通图有 3077 个节点、7286 条有向边；
反向去重后有 4606 个道路段，其中单向 1926 段、双向 2680 段。
双向道路通常贡献两条有向边；平行且几何不同的独立道路段仍保留。
方向分类依据 OSMnx `oneway` 属性，反向去重使用 `to_undirected`。

`boca_raton_drive_all.graphml` 和 `cache/` 仅保留为构图输入和历史数据缓存，
不作为当前统计对象。默认复用该输入快照；`--refresh` 从 OSMnx 响应重新构图，
但已有响应缓存仍会复用，不代表重新获取最新 OSM 数据。

统计采用拓扑简化的 `drive` 路网，不包含所有步道、自行车道和 OSM 原始形状点。
长度指标保留输入简化图的原始 `length`，不重复计算训练边权，亦不修改训练数据。
报告生成日期不代表地图数据日期。历史数据具体来源需以生成时的配套路网快照确认。
