# PDTSP_2V: 双车辆取送货路径规划问题

## 问题描述

PDTSP_2V (Pickup and Delivery TSP with 2 Vehicles) 是基于标准PDTSP问题的扩展，使用两辆车来解决取送货路径规划任务。

### 核心特点

1. **车辆配置**：两辆车共享同一个depot（起始点和终点）
2. **任务分配**：根据pickup点的y坐标自动分配：
   - **车辆1**：服务y坐标 >= 0.5的pickup点（上半区）
   - **车辆2**：服务y坐标 < 0.5的pickup点（下半区）
3. **移动约束**：
   - 两辆车可以在整个地图上自由移动
   - 每辆车只能拾取其分配区域的货物
   - 送货点可以在任意位置（因为存在跨区域送货的情况）
4. **优化目标**：最小化两辆车的总路径长度之和

### 约束条件

- ✅ 每个pickup-delivery对必须由同一辆车完成
- ✅ pickup必须在对应的delivery之前
- ✅ 每辆车只能服务其分配区域的pickup点
- ✅ 所有节点必须被访问恰好一次
- ❌ 无容量约束（当前版本）

## 实现架构

### 统一求解方案

本实现采用**统一求解**架构，即将两辆车的路径合并为一个统一的解表示：

```
解表示 (successor format):
rec[i] = j 表示节点i之后访问节点j

示例：rec = [1, 3, 5, 2, 6, 4, 0, ...]
表示路径：depot(0) -> 1 -> 3 -> 5 -> 2 -> 6 -> 4 -> depot(0)
```

### 关键设计

#### 1. 数据预处理
- 在`get_initial_solutions`中自动计算`vehicle_assignment`
- `vehicle_assignment[i]`表示节点i属于哪辆车（1或2），depot为0

#### 2. 网络架构
- **单一Actor-Critic网络**处理整个问题
- 输入维度固定（graph_size个节点），与车辆分配无关
- 自动适应不同的上下半区节点分布

#### 3. 动态Mask机制
- 在decoder的`get_swap_mask`中应用vehicle约束
- 确保改进操作（node removal & reinsertion）不违反车辆分配
- Mask逻辑：
  ```python
  # 只允许节点在其所属车辆的上下文中移动
  valid_position = (position_vehicle == node_vehicle) OR (position == depot)
  ```

## 代码改动说明

### 新增文件

1. **`problems/problem_pdtsp_2v.py`** (完整实现)
   - `PDTSP_2V`类：问题定义和求解逻辑
   - `PDTSP2VDataset`类：数据集加载
   - 关键方法：
     - `split_by_y_coordinate()`: 根据y坐标分配车辆
     - `get_initial_solutions()`: 生成初始解（三种策略：random/greedy/p2d）
     - `get_real_mask()`: 生成带车辆约束的mask
     - `insert_star()`: 执行节点重插入操作
     - `get_costs()`: 计算总路径长度
     - `get_costs_separate()`: 分别计算每辆车的路径长度

2. **`generate_pdtsp_2v_data.py`**
   - 数据生成脚本
   - 生成标准PDTSP格式的数据（车辆分配在求解时动态计算）

3. **`train_pdtsp_2v.sh`**
   - 完整训练pipeline
   - 包含数据生成、训练配置

4. **`eval_pdtsp_2v.sh`**
   - 模型评估脚本

### 修改文件

1. **`run.py`**
   ```python
   # 新增导入
   from problems.problem_pdtsp_2v import PDTSP_2V
   
   # 在load_problem中添加
   'pdtsp_2v': PDTSP_2V,
   ```

2. **`nets/graph_layers.py`**
   - `MultiHeadDecoder.forward()`: 添加`vehicle_assignment`参数
   - 在`get_swap_mask`调用中传递vehicle约束：
     ```python
     mask_table = problem.get_swap_mask(
         action_removal + 1, 
         visited_order_map, 
         vehicle_assignment if problem.NAME == 'pdtsp_2v' else top2
     )
     ```

3. **`nets/actor_network.py`**
   - `Actor.forward()`: 添加`vehicle_assignment`参数并传递给decoder

4. **`agent/ppo.py`**
   - 在所有`agent.actor()`调用处添加vehicle_assignment传递：
     ```python
     vehicle_assignment = batch.get('vehicle_assignment', None) \
                         if hasattr(problem, 'NAME') and problem.NAME == 'pdtsp_2v' \
                         else None
     exchange = self.actor(..., vehicle_assignment=vehicle_assignment)
     ```
   - 修改位置：
     - `rollout()`: 推理时
     - `train_batch()`: warm-up阶段
     - `train_batch()`: 采样轨迹
     - `train_batch()`: PPO更新循环

## 使用指南

### 1. 环境准备

确保已安装所有依赖：
```bash
cd n2s_osm
pip install -r requirements.txt  # 如果有的话
```

### 2. 生成数据

#### 手动生成
```bash
# 训练数据
python generate_pdtsp_2v_data.py \
    --graph_size 20 \
    --num_samples 50000 \
    --output ./datasets/pdtsp_2v_train_20.pkl \
    --seed 1234

# 验证数据
python generate_pdtsp_2v_data.py \
    --graph_size 20 \
    --num_samples 1000 \
    --output ./datasets/pdtsp_2v_val_20.pkl \
    --seed 5678
```

#### 使用训练脚本自动生成
```bash
# 训练脚本会自动检查并生成缺失的数据集
bash train_pdtsp_2v.sh
```

### 3. 训练模型

#### 使用默认配置
```bash
bash train_pdtsp_2v.sh
```

#### 自定义训练
```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --problem pdtsp_2v \
    --graph_size 20 \
    --train_dataset ./datasets/pdtsp_2v_train_20.pkl \
    --val_dataset ./datasets/pdtsp_2v_val_20.pkl \
    --batch_size 512 \
    --epoch_size 10240 \
    --n_epochs 100 \
    --lr_model 1e-4 \
    --lr_critic 1e-4 \
    --max_grad_norm 0.5 \
    --run_name pdtsp_2v_20 \
    --checkpoint_epochs 10
```

#### 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--graph_size` | 节点数（不含depot） | 20/50/100 |
| `--batch_size` | 批大小 | 512 (20节点), 256 (50节点) |
| `--epoch_size` | 每epoch的样本数 | 10240 |
| `--n_epochs` | 训练轮数 | 100 |
| `--lr_model` | Actor学习率 | 1e-4 |
| `--lr_critic` | Critic学习率 | 1e-4 |
| `--max_grad_norm` | 梯度裁剪 | 0.5 |
| `--T_train` | 训练时改进步数 | 50 |
| `--T_max` | 推理时改进步数 | 50 |

### 4. 评估模型

```bash
# 使用评估脚本
bash eval_pdtsp_2v.sh

# 或手动评估
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_2v \
    --graph_size 20 \
    --val_dataset ./datasets/pdtsp_2v_val_20.pkl \
    --load_path ./outputs/pdtsp_2v_20/epoch-99.pt \
    --eval_only \
    --val_m 1
```

### 5. 不同规模问题

#### 20节点问题（10对pickup-delivery）
```bash
python generate_pdtsp_2v_data.py --graph_size 20 --num_samples 50000 \
    --output ./datasets/pdtsp_2v_train_20.pkl

CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --problem pdtsp_2v \
    --graph_size 20 \
    --batch_size 512 \
    --train_dataset ./datasets/pdtsp_2v_train_20.pkl
```

#### 50节点问题（25对pickup-delivery）
```bash
python generate_pdtsp_2v_data.py --graph_size 50 --num_samples 50000 \
    --output ./datasets/pdtsp_2v_train_50.pkl

CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --problem pdtsp_2v \
    --graph_size 50 \
    --batch_size 256 \
    --train_dataset ./datasets/pdtsp_2v_train_50.pkl
```

#### 100节点问题（50对pickup-delivery）
```bash
python generate_pdtsp_2v_data.py --graph_size 100 --num_samples 50000 \
    --output ./datasets/pdtsp_2v_train_100.pkl

CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --problem pdtsp_2v \
    --graph_size 100 \
    --batch_size 128 \
    --train_dataset ./datasets/pdtsp_2v_train_100.pkl
```

## 技术细节

### 车辆分配算法

```python
def split_by_y_coordinate(self, batch):
    # 提取pickup点坐标
    pickup_coords = batch['coordinates'][:, 1:half_size+1, :]
    pickup_y = pickup_coords[:, :, 1]
    
    # 基于y坐标阈值(0.5)分配
    pickup_assignment = torch.where(
        pickup_y >= 0.5,
        torch.ones_like(pickup_y).long(),      # 车辆1
        torch.ones_like(pickup_y).long() * 2   # 车辆2
    )
    
    # delivery继承对应pickup的车辆分配
    vehicle_assignment[:, half_size+1:] = pickup_assignment
```

### 初始解生成策略

1. **Random** (`init_val_met='random'`)
   - 每辆车独立生成随机可行解
   - 快速但质量较低

2. **Greedy** (`init_val_met='greedy'`)
   - 最近邻启发式
   - 每辆车从depot开始，每次选择最近的可行节点
   - 质量较高但可能陷入局部最优

3. **Pickup-to-Delivery** (`init_val_met='p2d'`)
   - 每个pickup后立即delivery
   - 最简单的可行解
   - 通常路径较长

推荐使用 `greedy` 作为训练时的初始解策略。

### Mask机制

在节点重插入（reinsertion）操作中，mask确保：

1. **时序约束**：pickup必须在delivery之前
2. **车辆约束**：节点只能插入到其所属车辆的路径段中

```python
# 伪代码
for each candidate insertion position (i, j):
    if node belongs to vehicle_1:
        valid = (position_i in vehicle_1_path) AND (position_j in vehicle_1_path)
    if position is depot:
        valid = True  # depot总是可达
```

### 成本计算

```python
# 总成本（优化目标）
total_cost = vehicle1_distance + vehicle2_distance

# 可以通过get_costs_separate()获取每辆车的单独成本
v1_cost, v2_cost, total = problem.get_costs_separate(batch, solution)
```

## 与标准PDTSP的对比

| 特性 | PDTSP | PDTSP_2V |
|------|-------|----------|
| 车辆数 | 1 | 2 |
| 任务分配 | 无需分配 | 基于y坐标自动分配 |
| 移动约束 | 无 | 车辆只能拾取分配区域的货物 |
| 解表示 | 单一路径 | 统一路径（包含两车） |
| 网络架构 | 标准Actor-Critic | 相同，添加vehicle mask |
| 训练复杂度 | 基准 | 略高（mask计算） |
| 应用场景 | 单车路径规划 | 多区域协同配送 |

## 扩展方向

### 短期扩展

1. **容量约束**
   - 为每辆车添加容量限制
   - 在mask中添加容量检查

2. **时间窗约束**
   - 为pickup/delivery添加时间窗
   - 修改feasibility检查

3. **动态车辆数**
   - 支持3+辆车
   - 更灵活的区域划分策略

### 长期扩展

1. **自适应区域划分**
   - 学习最优的区域划分策略
   - 而非固定的y=0.5阈值

2. **异构车辆**
   - 不同车辆有不同的容量/速度
   - 考虑车辆特性的任务分配

3. **跨车辆协作**
   - 允许任务在车辆间转移
   - 更复杂的协同优化

## 常见问题

### Q1: 训练时车辆分配不平衡怎么办？

A: 这是正常现象。由于数据是随机生成的，某些实例可能出现8:2或3:7的分配比例。网络设计（统一求解 + 动态mask）天然支持这种不平衡，会自动适应。

### Q2: 可以修改车辆分配阈值吗？

A: 可以。在`problem_pdtsp_2v.py`的`split_by_y_coordinate`方法中修改：
```python
pickup_assignment = torch.where(pickup_y >= 0.6,  # 改为0.6
                               torch.ones_like(pickup_y).long(),
                               torch.ones_like(pickup_y).long() * 2)
```

### Q3: 为什么不使用分开求解方案？

A: 统一求解的优势：
- ✅ 单一网络，训练效率高
- ✅ 自动适应不同节点分布
- ✅ 代码改动最小
- ✅ 可扩展到N辆车

分开求解的缺点：
- ❌ 需要两组网络（内存翻倍）
- ❌ 无法batch处理不同尺寸
- ❌ 训练复杂度高

### Q4: 如何可视化两辆车的路径？

A: 可以使用`get_costs_separate()`方法提取每辆车的路径，然后绘制：
```python
v1_cost, v2_cost, total = problem.get_costs_separate(batch, solution)
# 根据vehicle_assignment分离路径并可视化
```

### Q5: 训练收敛慢怎么办？

A: 尝试：
1. 增大warm_up轮数 (`--warm_up 2`)
2. 调整学习率 (`--lr_model 5e-5`)
3. 使用greedy初始解 (`--init_val_met greedy`)
4. 增加batch_size
5. 调整梯度裁剪 (`--max_grad_norm 1.0`)

## 文件清单

```
n2s_osm/
├── problems/
│   └── problem_pdtsp_2v.py          # PDTSP_2V问题定义 (新增)
├── nets/
│   ├── actor_network.py             # Actor网络 (修改: +vehicle_assignment)
│   └── graph_layers.py              # Decoder (修改: +vehicle_assignment)
├── agent/
│   └── ppo.py                       # PPO训练 (修改: 传递vehicle_assignment)
├── generate_pdtsp_2v_data.py        # 数据生成脚本 (新增)
├── train_pdtsp_2v.sh                # 训练脚本 (新增)
├── eval_pdtsp_2v.sh                 # 评估脚本 (新增)
├── run.py                           # 主程序 (修改: 添加pdtsp_2v支持)
└── problem_pdtsp_2v.md              # 本文档 (新增)
```

## 性能参考

基于N2S方法的预期性能（需要实际训练验证）：

| 规模 | 训练时间 | 推理时间 | Gap vs 最优解 |
|------|---------|---------|--------------|
| 20节点 | ~2-3h | ~10s/1000实例 | 待测 |
| 50节点 | ~4-6h | ~30s/1000实例 | 待测 |
| 100节点 | ~8-12h | ~60s/1000实例 | 待测 |

*注：时间基于NVIDIA RTX 3090 / A100*

## 引用

如果使用此代码，请引用原始N2S论文以及本项目：

```bibtex
@article{n2s,
  title={Neural Improvement Heuristics for Graph Combinatorial Optimization},
  author={...},
  journal={...},
  year={...}
}
```

## 联系方式

- Issue tracker: [GitHub Issues]
- 技术讨论: [相关论坛/讨论组]

---

**最后更新**: 2025-12-01  
**版本**: 1.0  
**维护者**: [Your Name]
