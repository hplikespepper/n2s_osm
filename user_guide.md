## Multi-Vehicle PDTSP (Centralized Policy) — Quick Guide

### 方案要点（精简版）
- **结构**：保持 Transformer Encoder–Decoder 不变。
- **解表示**：多车使用“多个闭合子回路（Multiple Disjoint Cycles）”，通过**克隆多个 Depot**实现分车路径。
- **节点索引约定**：
	- Depot 克隆：`0..(m-1)`
	- Pickup：`m..(m+K-1)`
	- Delivery：`(m+K)..(m+2K-1)`
	- 其中 $m=\text{num\_vehicles}$，$K=\text{num\_pairs}$
- **动作空间**：Removal 选择一个订单；Reinsertion 同时插入该订单的 pickup 与 delivery。
- **目标**：最小化所有车辆路径总长度（sum of tour length）。

### 约束（当前实现）
- **允许跨车迁移**：移除一个订单后，可将该订单整体插回到任意车辆路径段。
- **禁止跨车拆分**：pickup 和 delivery 必须插在**同一辆车**路径内。
- **先取后送**：delivery 不允许出现在 pickup 之前。

### 关键实现位置
- 问题定义与数据集：
	- [problems/problem_mvpdtsp.py](problems/problem_mvpdtsp.py)
	- 关键函数：`get_vehicle_id()`、`get_swap_mask()`、`get_initial_solutions()`
- 解码器多车兼容：
	- [nets/graph_layers.py](nets/graph_layers.py)
	- `get_swap_mask(..., rec)` 使用当前解 `rec` 计算同车约束
- Actor 输入与位置编码：
	- [nets/actor_network.py](nets/actor_network.py)
	- [nets/graph_layers.py](nets/graph_layers.py)
- 运行入口与参数：
	- [run.py](run.py)
	- [options.py](options.py)

### 使用命令
> 注意：`graph_size` 表示 **pickup+delivery 总节点数**（例如 20 表示 10 单）。

**训练**（默认 2 辆车）：
```bash
python run.py --problem mvpdtsp --graph_size 20 --num_vehicles 2 \
	--batch_size 600 --epoch_size 12000 --epoch_end 200 --run_name 'mvpdtsp_20'
```

简单测试：
python run.py --problem mvpdtsp --graph_size 20 --num_vehicles 2 --batch_size 64 --epoch_size 256 --epoch_end 1 --T_
train 5 --T_max 10 --run_name 'temp_tsp20'

**训练（多车）**：
```bash
python run.py --problem mvpdtsp --graph_size 50 --num_vehicles 4 \
	--batch_size 600 --epoch_size 12000 --epoch_end 200
```

**验证/推理**：
```bash
python run.py --problem mvpdtsp --graph_size 20 --num_vehicles 2 \
	--eval_only --val_size 1000 --val_batch_size 1000
```

### 备注
- `num_vehicles` 默认值为 2。
- 当前实现保留 `capacity` 参数但**未启用载重约束**。


### 01/19/2026
# 评估时绘制初始的路径（用于检查初始解是否合理）
- 参数“--print_solution”

# 当前的mvpdtsp存在一个逻辑漏洞，因为优化目标只是最小化多车辆的行驶距离，但使用多辆车的距离极大概率是大于一辆车的，因为还存在从仓库出发和回仓库的距离，导致模型学到的就是将所有请求都分配给一辆车，另一辆车不行动。因此：加入所有车辆路线的最大完成时间
- 快速训练命令（基于你给的命令，启用 makespan）：
	python run.py --problem mvpdtsp --graph_size 20 --num_vehicles 2 --batch_size 64 --epoch_size 256 --epoch_end 1 --T_train 5 --T_max 10 --run_name 'temp_tsp20' --makespan
- 快速评估命令（启用 makespan 目标；按需加模型路径）：
	python run.py --eval_only --problem mvpdtsp --graph_size 20 --num_vehicles 2 --val_size 256 --val_batch_size 256 --T_max 10 --run_name 'mv20_eval_epoch198' --load_path outputs/mvpdtsp_20/mvpdtsp20_makespan_20260119T182324/epoch-198.pt --makespan
- 完整评估：
	python run.py --eval_only --problem mvpdtsp --graph_size 20 --num_vehicles 2 --val_size 256 --val_batch_size 256 --T_max 3000 --run_name 'mv20_eval_epoch198' --load_path outputs/mvpdtsp_20/mvpdtsp20_makespan_log_20260130T200924/epoch-198.pt --makespan
	**一定要调大T_max到3000**

可视化：
1. 使用best_rec(邻接表示)
	python vis_mvpdtsp_rec.py --instance_id 0 --results_file ./results/mvpdtsp_results_mv_198.json --save_path visualizations/mv_198.png
or
2. 使用解码的顺序表示
	python vis_mvpdtsp.py --instance_id 1 --results_file ./results/mvpdtsp_results_mv_198.json --save_path visualizations/mv_198.png

02/02/2026
# 更新了训练代码，新增记录训练日志，以及增加tensorboard日志的指标
# 创建了analysis_figure.py用于绘制训练中各指标的变化
python analysis_figure.py --result_file ./outputs/mvpdtsp_20/mvpdtsp20_makespan_log_20260130T200924/mvpdtsp_epoch_metrics.jsonl --save_figure ./figure


02/08/2026
# 对比试验使用ortools:
	- (存在一个问题 ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. tensorboard 2.11.0 requires protobuf<4,>=3.9.2, but you have protobuf 5.29.6 which is incompatible. 安装ortools会导致tensorboard无法使用，理由是protobuf冲突)
	
	多车：
	python ortools_baseline.py --val_size 256 --time_limit 15
	单车： ortools_pdtsp.py

	# 可视化代码同n2s一样

# 查看结果统计：
	python stat_best_cost.py --json_path ./results/mvpdtsp_results_epoch_198.json


03/05/2026
# 创建蒙特卡洛基准：
	# 快速测试（5个实例，1000次采样）
	python MonteCarlo_mvpdp.py --val_size 5 --num_mc_samples 1000

	# 完整运行（1000个实例，10000次采样）
	python MonteCarlo_mvpdp.py --val_size 1000 --num_mc_samples 10000

	# 自定义输出路径
	python MonteCarlo_mvpdp.py --output results/mc_result.json

	# 贪心+随机扰动模式
	python MonteCarlo_mvpdp.py --greedy --val_size 1000 --num_mc_samples 10000

	# 调整扰动比例（0=纯贪心，1=接近纯随机，默认0.3）
	python MonteCarlo_mvpdp.py --greedy --perturb_ratio 0.2

03/09/2026
# 新增对蒙特卡洛结果的统计分析脚本（特别的，包含solve time，其他的结果中似乎没有这项）
python analysis_json.py results/mvpdtsp_results_mc_20260309_150953.json