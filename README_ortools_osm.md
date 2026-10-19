# OSM 两车 OR-Tools 对照组

在项目根目录运行，使用独立的 `ortools` conda 环境：

```bash
conda activate ortools
python n2s_osm_marl/ortools_osm.py
```

默认读取脚本所在目录的 `datasets/osm_val_20.pkl`，求解前 100 个实例、
20 个客户节点、2 辆车，目标 `proposed = D + M`。使用数据集内的有向
道路距离，保证同车取送、先取后送；允许空车，与当前 N2S 一样不添加
容量约束。每个实例搜索限时 15 秒，默认串行。初始解策略为
`PATH_CHEAPEST_ARC`，与合成数据的 `ortools_baseline.py` 一致；
启用完整取送对重定位算子，并使用 `GUIDED_LOCAL_SEARCH`。

结果及统计摘要保存到 `n2s_osm_marl/result_ortools/`。例如：

```bash
python n2s_osm_marl/ortools_osm.py \
  --val_size 100 --time_limit 60 --num_workers 4
```

`--offset` 可以选择数据集起始样本；实例的 `dataset_index` 始终对应原始
pickle 中的索引。`--output` 指定 JSON 文件，`--output_dir` 指定默认输出
目录。结果按原始评估顺序保存，不受并行完成顺序影响。

JSON 保持 `problem=mvpdtsp_osm`，含 `instances`、`best_rec`、车辆路线、
总道路距离、最长路线距离、每车距离和源数据集等字段。OR-Tools 没有
N2S 的 rollout 步数和随机初始解，因此不填充 `T_max` 或 `initial_*`。
不输出把道路距离称为完成时间的字段。

可直接使用已有绘图程序，也可以借助前 5 个实例的 shell 脚本：

```bash
conda activate rlor
RESULTS_FILE=./result_ortools/mvpdtsp_osm_results_ortools_<timestamp>.json \
  bash n2s_osm_marl/vis_osm_real.sh
```

此命令的图片仍保存到 `result_n2s/mvpdtsp_osm20_mv2/visualization/`。
若要将对照组图片单独保存，直接调用 Python：

```bash
cd n2s_osm_marl
/home/data/phuang/miniconda3/envs/rlor/bin/python vis_osm_real.py \
  --results result_ortools/mvpdtsp_osm_results_ortools_<timestamp>.json \
  --val_dataset datasets/osm_val_20.pkl --index 0 \
  --output_dir result_ortools/visualization
```

搜索使用 OR-Tools 整数距离，默认 `--scale 1000`，即在原数据单位下
四舍五入至 0.001。导出的成本则复用 `MVPDTSP_OSM` 的 float32 原道路
矩阵计算，与 N2S 统计口径相同；`solver_integer_objective` 和
`solver_scaled_objective` 记录搜索目标的整数值及还原值。限时启发式搜索
不提供最优性证明。建模参考 [OR-Tools 取送约束文档](https://developers.google.com/optimization/routing/pickup_delivery)
和 [车辆路线全局跨度目标文档](https://developers.google.com/optimization/routing/vrp)。

求解环境包含 CPU 版 PyTorch、OR-Tools 和 OSM 数据读取所需依赖，
与训练和绘图使用的 `rlor` 完全分开。当前 `rlor` 环境的 OR-Tools 9.12
与 protobuf 3.20.3 不兼容，所以仅在 `ortools` 中求解，在 `rlor` 中绘图。
环境重建及测试命令如下：

```bash
conda env create -f n2s_osm_marl/environment_ortools.yml
conda activate ortools
python -m unittest discover \
  -s n2s_osm_marl/tests -p test_ortools_osm.py -v
```
