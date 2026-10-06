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

	50个节点：
	CUDA_VISIBLE_DEVICES=2 python run.py --eval_only --problem mvpdtsp --graph_size 50 --num_vehicles 2 --val_size 256 --val_batch_size 256 --T_max 3000 --run_name 'mv50_eval_epoch199' --load_path outputs/mvpdtsp_50/mvpdtsp50_makespan_log_20260311T220152/epoch-199.pt --makespan --val_dataset ./datasets/pdp_50.pkl 

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

# 3 vehicles
	-bash mv_3_exp.sh
	-CUDA_VISIBLE_DEVICES=2 GRAPH_SIZES="50 100" VAL_SIZE=1000 VAL_BATCH_SIZE=1000 PRINT_SOLUTION=1 bash mv_3_eval.sh
# 05/06/2026
	发现重大bug，若只是用 dis + makespan，在2车的情况下适用，但是扩展到更多车的情况下，makespan的边际效应就很小了，导致塌缩到只是用2辆车。因此做出调整：
	distance + (num_vehicles - 1) * makespan

# 09/15/2026
	修正了n2s_mv/MonteCarlo_mvpdp.py的一些小问题（不影响结果），并开启了并行运行的功能。
	新增mv_mc.sh用于多车辆实验
	新增了mv_ortools.sh用于多车辆实验
	可视化（通用）：
	python vis_mvpdtsp.py \
	--results_file result_mc/run_20260915_112436_3UijNR/mc_50_mv2.json \
	--instance_id 0 \
	--save_path result_mc/run_20260915_112436_3UijNR/mc_50_mv2_instance_0.png


# 09/17/2026
	更新了ortools_baseline.py: 新增"--pair_relocate", choices=["full", "light"], default="full",full为完整算子，而light是轻量版算子：
	| 对比项 | 完整版 | 轻量版 |
	| 调整对象 | 一对取货点和送货点 | 同样是一对取送货点 |
	| 插入位置 | 分别枚举取货、送货位置的组合 | 根据已有请求对等结构，限制位置组合 |
	| 候选数量 | 较多 | 较少 |
	| 单轮搜索开销 | 较大 | 通常较小 |
	| 跨车辆移动 | 支持 | 支持 |
	| 全局最优保证 | 没有 | 没有 |

	* 轻量版：用规则关联两个位置
		当前版本的轻量配置实际包含两种移动：
		- LightPairRelocateOperator：借助目标路线已有请求对的位置安排新请求对。例如把新取货点放到另一取货点后，再把对应送货点放到另一送货点后。
		- GroupPairAndRelocate：将取货和送货组成相邻的 P → D，一起移到某个位置。
		因此，轻量版并不强制所有请求都取完立刻送，也仍然允许交错取送；只是减少了单次移动时尝试的位置组合。

	mv_ortools.sh为轻量版实验脚本
	mv_ortools_full.sh为对100节点额外进行的完整版算子实验，特别的，time_limit为900，需要的时间很长

