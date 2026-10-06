# 真实地图路径可视化

`vis_osm_real.py` 输出标准 OSM 或 CartoDB Positron 底图上的静态 PNG。
使用原有 `best_path` 邻接表、验证集 `node2osmid`、`path_lookup` 和 `pairs`。
路线仍连接 OSM 路径节点，不使用道路边的曲线几何，不改变训练或成本计算。

## 环境和运行

在已有项目 Python 环境中安装绘图依赖（生成验证集仍需要项目的 PyTorch）：

```bash
cd n2s_osm_marl
python -m pip install -r requirements-vis-osm-real.txt
python vis_osm_real.py --results results/pdtsp_results_20260209_202314.json --val_dataset datasets/osm_val_20.pkl --index 0 --basemap osm
# Positron 需要先在环境中设置 CARTO_BASEMAP_API_KEY，勿将真实密钥写入代码或提交仓库。
python vis_osm_real.py --results results/pdtsp_results_20260209_202314.json --val_dataset datasets/osm_val_20.pkl --index 0 --basemap positron
```

默认写入 `visualizations/instance_0_cost_<成本>_osm.png` 或 `_positron.png`。
底图包含版权来源标注，标准 OSM 使用 Mapnik，Positron 使用 CARTO `rastertiles/light_all` 端点。
首次下载瓦片需要网络，地图详细程度由缩放级别决定。

CARTO 当前要求 API key（见 [官方说明](https://www.carto.com/basemaps/apikey/)）。
通过官方页面获取密钥后，在运行环境设置 `CARTO_BASEMAP_API_KEY`。
未设置时，Positron 绘图会明确失败，避免将“API key required”提示瓦片误当成地图。
本次实际网络验收验证了 OSM；Positron 渲染逻辑通过模拟底图检查，完整在线底图需有效密钥后验证。

## 参数

| 参数 | 默认值 / 行为 |
| --- | --- |
| `--results` | 与旧脚本相同的历史默认路径；实际运行建议显式指定存在的文件 |
| `--val_dataset` | `datasets/osm_val_20.pkl` |
| `--index` | `0`；结果和验证集应有相同实例顺序，无偏移 |
| `--osm_place` | `Boca Raton, Florida, USA`，仅首次构建旧数据配套路网时使用 |
| `--output_dir` | `visualizations` |
| `--basemap` | `osm`，可选 `positron` |
| `--graphml` | 未指定时自动加载验证集同名 `.graphml`；显式指定的文件加载失败时停止 |
| `--extent` | `route`：完整路线和节点范围；`place`：整个固定路网范围 |
| `--zoom` | `auto`；也可指定地图源支持的整数缩放级别 |
| `--dpi` | `200` |
| `--tile_cache_dir` | `cache/vis_osm_real`，持久缓存相同范围/缩放下下载的瓦片 |

所有文件路径相对于运行目录。图层统一投影为 EPSG:3857，范围每侧留 10% 边距，
窄/退化范围按至少 200 米展开。标题中的 Objective 是结果文件的优化目标值，
Task nodes 包含 depot，不代表行驶距离。

## 固定路网

```bash
python create_osm_val_dataset.py --output datasets/new_val.pkl --num_samples 10
# 同时得到 datasets/new_val.graphml，保存生成这些样本时实际使用的 dataset.G
python vis_osm_real.py --results results/<匹配结果>.json --val_dataset datasets/new_val.pkl --basemap osm
```

旧验证集没有配套 GraphML 时，脚本复用现有 OSMnx 下载缓存并构建路网，
检查选中实例任务节点、路径端点和有向边后，保存同名 GraphML。后续直接读取固定文件，
不再在线重建路网。首次固定的路网不能证明与旧数据生成时完全一致；不同实例也会逐次检查。
在线地图瓦片与路网可能来自不同时间版本。

无效索引、未闭合或未覆盖全部任务节点的解、缺失 OSM 节点/路径边、不可达路段，
以及底图加载失败都会停止输出并返回非零退出码。固定的有问题 GraphML 不会自动刷新；
可通过 `--graphml` 提供正确文件。不会使用虚构直连路段替代缺失道路。

## 验证

```bash
python -m unittest discover -s tests -p 'test_vis_osm_real.py' -v
```

单元测试使用小型固定有向路网和模拟底图，不需要网络；实际两种底图应另外运行上述命令检查。
本版仅支持旧 `vis_osm.py` 的单 depot PDTSP 结果，不解码多车辆结果。
