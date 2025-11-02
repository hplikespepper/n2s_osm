# OSM-PDP N2S 完整使用指南

本文档提供了基于真实地图（OpenStreetMap）的取送货问题（PDP）使用 #### 示例：生成 20 节点验证数据集
```bash
python create_osm_val_dataset.py \
    --graph_size 20 \
    --num_samples 100 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_val_20.pkl
```
### 3. 训练 50 节点模型（使用预生成数据集）

```bash
CUDA_VISIBLE_DEVICES=6 python run.py \
    --problem pdtsp_osm \
    --graph_size 50 \
    --train_dataset './datasets/osm_train_50_50k.pkl' \
    --batch_size 256 \
    --epoch_size 12000 \
    --epoch_end 50 \
    --T_train 500 \
    --val_dataset './datasets/osm_val_50.pkl' \
    --val_size 100 \
    --val_m 2 \
    --run_name osm_train_50 \
    --lr_model 4e-5 \
    --lr_critic 1e-5 \
    --checkpoint_epochs 5
```

### 4. 训练 100 节点模型（使用预生成数据集）

```bash
CUDA_VISIBLE_DEVICES=7 python run.py \
    --problem pdtsp_osm \
    --graph_size 100 \
    --train_dataset './datasets/osm_train_100_20k.pkl' \
    --batch_size 64 \
    --epoch_size 12000 \
    --epoch_end 50 \
    --T_train 1000 \
    --val_dataset './datasets/osm_val_100.pkl' \
    --val_size 100 \
    --val_m 2 \
    --run_name osm_train_100 \
    --lr_model 2e-5 \
    --lr_critic 5e-6 \
    --checkpoint_epochs 5
```

### 5. 双GPU分布式训练（使用预生成数据集）

```bash
CUDA_VISIBLE_DEVICES=6,7 python run.py \
    --problem pdtsp_osm \
    --graph_size 50 \
    --train_dataset './datasets/osm_train_50_50k.pkl' \
    --batch_size 128 \
    --epoch_size 24000 \
    --epoch_end 50 \
    --T_train 500 \
    --val_dataset './datasets/osm_val_50.pkl' \
    --val_size 100 \
    --val_m 2 \
    --run_name osm_train_50_ddp \
    --lr_model 4e-5 \
    --lr_critic 1e-5
```

**注意**：
- 目前仅支持2个GPU（4+GPU会出现NCCL错误，可能是OSMnx并发访问问题）
- 使用预生成数据集可以完全避免在线生成的并发问题
- 双GPU训练时batch_size=128（单GPU的一半），但总batch变大（128x2=256）

---

## 性能对比

| 方式 | 速度 (iter/s) | GPU利用率 | epoch耗时 (12k samples) | 总训练时间 (50 epochs) |
|------|---------------|-----------|------------------------|----------------------|
| 在线生成 | 1-2 | ~0% | ~2-3小时 | **100-150小时** ❌ |
| 预生成数据 | 50-100 | >90% | ~2-5分钟 | **2-4小时** ✅ |

**速度提升**: **100-1000倍** 🚀

---数据集
```bash
python create_osm_val_dataset.py \
    --graph_size 50 \
    --num_samples 100 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_val_50.pkl
```

#### 示例：生成 100 节点验证数据集
```bash
python create_osm_val_dataset.py \
    --graph_size 100 \
    --num_samples 100 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_val_100.pkl
```

### 2. 生成训练数据集 ⚡ **强烈推荐，可提速100-1000倍！**

**为什么需要预生成训练数据集？**

在线生成OSM数据非常慢（每个样本0.1-1秒），导致：
- GPU利用率接近0%（GPU在等待CPU生成数据）
- 训练速度极慢（1-2 iter/s）
- 浪费计算资源

预生成训练数据集后：
- 训练速度提升 **100-1000倍** （50-100 iter/s）
- GPU利用率 >90%
- 一次生成，多次使用

#### 命令格式
```bash
python create_osm_training_dataset.py \
    --graph_size <图规模> \
    --num_samples <样本数量> \
    --place "<OSM地点字符串>" \
    --output <输出文件路径>
```

#### 示例：生成 20 节点大规模训练集（10万样本）
```bash
python create_osm_training_dataset.py \
    --graph_size 20 \
    --num_samples 100000 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_train_20_100k.pkl
```

**预计耗时**：1-3小时（取决于CPU性能），但只需生成一次！

#### 示例：生成 50 节点训练集（5万样本）
```bash
python create_osm_training_dataset.py \
    --graph_size 50 \
    --num_samples 50000 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_train_50_50k.pkl
```

#### 示例：生成 100 节点训练集（2万样本）
```bash
python create_osm_training_dataset.py \
    --graph_size 100 \
    --num_samples 20000 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_train_100_20k.pkl
```

**建议样本数量**：
- graph_size=20: 50k-100k samples
- graph_size=50: 30k-50k samples  
- graph_size=100: 10k-20k samples

### 3. 查看数据集内容eighborhood Search (N2S) 进行训练和测试的完整流程。

## 目录
- [环境准备](#环境准备)
- [数据生成](#数据生成)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [结果可视化](#结果可视化)
- [路径设置说明](#路径设置说明)
- [常用参数说明](#常用参数说明)
- [完整示例](#完整示例)

---

## 环境准备

### 1. 激活 Python 环境
确保已安装所有必需的依赖包（如 PyTorch, OSMnx, NetworkX 等）。

### 2. 目录结构
```
pdp_n2s_oms/
├── datasets/          # 数据集存储目录
├── outputs/           # 模型检查点保存目录
├── logs/              # TensorBoard 日志目录
├── results/           # 验证结果保存目录（硬编码）
├── run.py             # 主训练/评估脚本
├── create_osm_val_dataset.py  # 生成验证数据集
├── read_data_mos.py   # 查看数据集内容
└── options.py         # 命令行参数定义
```

---

## 数据生成

### 1. 生成验证数据集

#### 命令格式
```bash
python create_osm_val_dataset.py \
    --graph_size <图规模> \
    --num_samples <样本数量> \
    --place "<OSM地点字符串>" \
    --output <输出文件路径>
```

#### 示例：生成 20 节点验证数据集
```bash
python create_osm_val_dataset.py \
    --graph_size 20 \
    --num_samples 100 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_val_20.pkl
python create_osm_val_dataset.py \
    --graph_size 20 \
    --num_samples 100 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_val_20.pkl
```

#### 示例：生成 50 节点验证数据集
```bash
python create_osm_val_dataset.py \
    --graph_size 50 \
    --num_samples 100 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_val_50.pkl
```

#### 示例：生成 100 节点验证数据集
```bash
python create_osm_val_dataset.py \
    --graph_size 100 \
    --num_samples 100 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_val_100.pkl
```

### 2. 查看数据集内容

使用 `read_data_mos.py` 检查生成的数据集是否正确：

```bash
python read_data_mos.py
```

**注意**：需要先在脚本中修改 `file_path` 变量为要检查的数据集路径。

---

## 模型训练

### 训练方式对比

#### 🐌 方式1：在线生成数据（不推荐，极慢）
```bash
# 没有 --train_dataset 参数，每个epoch都重新生成数据
python run.py --problem pdtsp_osm --graph_size 20 ...
```
- **速度**：1-2 iter/s
- **GPU利用率**：接近0%
- **适用场景**：快速测试（小epoch_size）

#### ⚡ 方式2：使用预生成数据集（**强烈推荐**，快100-1000倍）
```bash
# 使用 --train_dataset 参数加载预生成数据
python run.py --problem pdtsp_osm --graph_size 20 \
    --train_dataset './datasets/osm_train_20_100k.pkl' ...
```
- **速度**：50-100 iter/s
- **GPU利用率**：>90%
- **适用场景**：正式训练

---

### 1. 基础训练命令

#### 命令格式（使用预生成数据集）
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python run.py \
    --problem pdtsp_osm \
    --graph_size <图规模> \
    --train_dataset <训练集路径> \
    --batch_size <批次大小> \
    --epoch_size <每轮样本数> \
    --epoch_end <总轮数> \
    --T_train <训练步数> \
    --val_dataset <验证集路径> \
    --val_size <验证集大小> \
    --run_name <运行名称> \
    [其他参数]
```

### 2. 训练 20 节点模型

#### 快速测试（在线生成，用于调试）
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 20 \
    --batch_size 4 \
    --epoch_size 20 \
    --epoch_end 1 \
    --T_train 10 \
    --val_dataset './datasets/osm_val_20.pkl' \
    --val_size 2 \
    --val_m 1 \
    --run_name osm_test_20 \
    --no_tb \
    --no_saving
```

#### 完整训练（使用预生成数据集）⚡ **推荐**
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 20 \
    --train_dataset './datasets/osm_train_20_100k.pkl' \
    --batch_size 600 \
    --epoch_size 12000 \
    --epoch_end 50 \
    --T_train 250 \
    --val_dataset './datasets/osm_val_20.pkl' \
    --val_size 100 \
    --val_m 2 \
    --run_name osm_train_20 \
    --lr_model 8e-5 \
    --lr_critic 2e-5 \
    --checkpoint_epochs 5
```

### 3. 训练 50 节点模型（使用预生成数据集）

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 50 \
    --batch_size 400 \
    --epoch_size 10000 \
    --epoch_end 100 \
    --T_train 300 \
    --val_dataset './datasets/osm_val_50.pkl' \
    --val_size 100 \
    --val_m 2 \
    --run_name osm_train_50 \
    --lr_model 8e-5 \
    --lr_critic 2e-5 \
    --checkpoint_epochs 10
```

### 4. 训练 100 节点模型

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 100 \
    --batch_size 200 \
    --epoch_size 8000 \
    --epoch_end 150 \
    --T_train 400 \
    --val_dataset './datasets/osm_val_100.pkl' \
    --val_size 100 \
    --val_m 2 \
    --run_name osm_train_100 \
    --lr_model 6e-5 \
    --lr_critic 1.5e-5 \
    --checkpoint_epochs 15
```

### 5. 从检查点恢复训练

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 20 \
    --resume outputs/pdtsp_osm_20/osm_train_20_20251007T120000 \
    --epoch_start 10
```

---

## 模型评估

### 1. 评估已训练模型

#### 命令格式
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python run.py \
    --problem pdtsp_osm \
    --graph_size <图规模> \
    --eval_only \
    --load_path <模型路径> \
    --val_dataset <验证集路径> \
    --val_size <验证集大小> \
    --T_max <推理步数> \
    --val_m <数据增强倍数> \
    --no_saving \
    --no_tb
```

#### 示例：评估 20 节点模型
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 20 \
    --eval_only \
    --load_path outputs/pdtsp_osm_20/osm_train_20_20251007T120000/epoch-50.pt \
    --val_dataset './datasets/osm_val_20.pkl' \
    --val_size 100 \
    --T_max 1500 \
    --val_m 4 \
    --no_saving \
    --no_tb
```

#### 示例：评估 50 节点模型
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 50 \
    --eval_only \
    --load_path outputs/pdtsp_osm_50/osm_train_50_20251007T140000/epoch-100.pt \
    --val_dataset './datasets/osm_val_50.pkl' \
    --val_size 100 \
    --T_max 2000 \
    --val_m 4 \
    --no_saving \
    --no_tb
```

---

## 结果可视化

### 1. 可视化评估结果

使用 `vis_osm.py` 脚本可视化评估结果到真实地图上。

#### 命令格式
```bash
python vis_osm.py \
    --results <结果JSON文件路径> \
    --val_dataset <验证数据集路径> \
    --index <实例索引> \
    --osm_place "<OSM地点字符串>" \
    --output_dir <可视化输出目录>
```

#### 示例：可视化第 0 个实例
```bash
python vis_osm.py \
    --results results/pdtsp_results_20251007_171709.json \
    --val_dataset ./datasets/osm_val_20.pkl \
    --index 0 \
    --osm_place "Boca Raton, Florida, USA" \
    --output_dir visualizations
```

#### 示例：可视化第 3 个实例
```bash
python vis_osm.py \
    --results results/pdtsp_results_20251007_171709.json \
    --index 3 \
    --output_dir visualizations
```

#### 通过代码控制可视化实例

也可以直接修改 `vis_osm.py` 脚本顶部的配置变量：

```python
# ==================== Configuration ====================
# Control which instance to visualize
INSTANCE_INDEX = 0  # 修改此变量来可视化不同的实例

# OSM place (should match the training data)
OSM_PLACE = "Boca Raton, Florida, USA"

# Paths
RESULTS_FILE = "results/pdtsp_results_20251007_171709.json"
VAL_DATASET_FILE = "./datasets/osm_val_20.pkl"
OUTPUT_DIR = "visualizations"
# ======================================================
```

然后直接运行：
```bash
python vis_osm.py
```

#### 可视化输出

- **输出目录**：`visualizations/`（可通过 `--output_dir` 修改）
- **文件命名**：`instance_{index}_cost_{cost}.png`
- **图例说明**：
  - 🟢 绿色星星：Depot（仓库/起点）
  - 🔵 蓝色圆圈：Pickup 节点（取货点）
  - 🟠 橙色方块：Delivery 节点（送货点）
  - 🔴 红色路径：求解的路径

#### 批量可视化

可以编写循环脚本批量生成所有实例的可视化：

```bash
# 可视化前 5 个实例
for i in {0..4}; do
    python vis_osm.py --index $i --results results/pdtsp_results_20251007_171709.json
done
```

---

## 路径设置说明

### 1. 模型保存路径（可配置）

模型检查点的保存路径由以下参数控制：

```
<output_dir>/<problem>_<graph_size>/<run_name>_<timestamp>/epoch-<N>.pt
```

**相关参数：**
- `--output_dir`：输出目录，默认为 `outputs`
- `--problem`：问题类型，设为 `pdtsp_osm`
- `--graph_size`：图规模，如 20、50、100
- `--run_name`：运行名称，自定义标识
- `--checkpoint_epochs`：每 N 轮保存一次检查点（默认 1）
- `--no_saving`：禁用模型保存

**示例路径：**
```
outputs/pdtsp_osm_20/osm_train_20_20251007T120000/epoch-50.pt
```

### 2. 验证结果保存路径（硬编码）

验证结果的保存路径在代码中**硬编码**为：

```
results/pdtsp_results_<timestamp>.json
```

- 保存位置：`agent/utils.py` 第 155 行和第 198 行
- 格式：`results/pdtsp_results_YYYYMMDD_HHMMSS.json`
- **当前无法通过命令行参数修改**

**示例路径：**
```
results/pdtsp_results_20251007_143929.json
```

### 3. TensorBoard 日志路径（可配置）

TensorBoard 日志的保存路径：

```
<log_dir>/<problem>_<graph_size>/<run_name>_<timestamp>/
```

**相关参数：**
- `--log_dir`：日志目录，默认为 `logs`
- `--no_tb`：禁用 TensorBoard 日志

**示例路径：**
```
logs/pdtsp_osm_20/osm_train_20_20251007T120000/
```

### 4. 数据集路径（可配置）

**验证数据集：**
- 参数：`--val_dataset`
- 默认：`./datasets/pdp_20.pkl`
- 推荐：使用 OSM 数据集时设置为 `./datasets/osm_val_<graph_size>.pkl`

**训练数据集：**
- 在线生成，无需预先准备
- 参数 `--osm_place` 控制 OSM 地点

---

## 常用参数说明

### 核心参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--problem` | 问题类型 | `pdtsp` | `pdtsp_osm` (使用 OSM) |
| `--graph_size` | 图节点数 | 20 | 20/50/100 |
| `--osm_place` | OSM 地点字符串 | `"Boca Raton, Florida, USA"` | 根据需求设置 |
| `--capacity` | 车辆容量 | 3 | 3 |

### 训练参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--batch_size` | 批次大小 | 600 | 20节点: 600<br>50节点: 400<br>100节点: 200 |
| `--epoch_size` | 每轮样本数 | 12000 | 根据资源调整 |
| `--epoch_end` | 总训练轮数 | 200 | 20节点: 50<br>50节点: 100<br>100节点: 150 |
| `--T_train` | 训练步数 | 250 | 20节点: 250<br>50节点: 300<br>100节点: 400 |
| `--lr_model` | Actor 学习率 | 8e-5 | 8e-5 (较小问题)<br>6e-5 (较大问题) |
| `--lr_critic` | Critic 学习率 | 2e-5 | 2e-5 (较小问题)<br>1.5e-5 (较大问题) |
| `--lr_decay` | 学习率衰减 | 0.985 | 0.985 |

### 验证参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--val_dataset` | 验证集路径 | `./datasets/pdp_20.pkl` | `./datasets/osm_val_<size>.pkl` |
| `--val_size` | 验证集样本数 | 1000 | 训练时: 100<br>评估时: 1000 |
| `--val_m` | 数据增强倍数 | 1 | 训练时: 1-2<br>评估时: 4 |
| `--T_max` | 推理最大步数 | 1500 | 20节点: 1500<br>50节点: 2000<br>100节点: 2500 |

### 路径参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output_dir` | 模型输出目录 | `outputs` |
| `--log_dir` | TensorBoard 日志目录 | `logs` |
| `--run_name` | 运行名称标识 | `run_name` |
| `--checkpoint_epochs` | 保存检查点间隔 | 1 |

### 开关参数

| 参数 | 说明 | 效果 |
|------|------|------|
| `--no_cuda` | 禁用 GPU | 使用 CPU 训练 |
| `--no_tb` | 禁用 TensorBoard | 不记录日志 |
| `--no_saving` | 禁用保存模型 | 不保存检查点 |
| `--eval_only` | 仅评估模式 | 加载模型并评估 |
| `--no_progress_bar` | 禁用进度条 | 减少终端输出 |

---

## 完整示例

### 示例 1：从零开始训练 20 节点 OSM 模型

```bash
# 步骤 1: 生成验证数据集
python create_osm_val_dataset.py \
    --graph_size 20 \
    --num_samples 100 \
    --place "Boca Raton, Florida, USA" \
    --output ./datasets/osm_val_20.pkl

# 步骤 2: 检查数据集
python read_data_mos.py  # 需修改脚本中的 file_path

# 步骤 3: 开始训练
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 20 \
    --batch_size 600 \
    --epoch_size 12000 \
    --epoch_end 50 \
    --T_train 250 \
    --val_dataset './datasets/osm_val_20.pkl' \
    --val_size 100 \
    --val_m 2 \
    --run_name osm_train_20 \
    --checkpoint_epochs 5

# 步骤 4: 评估最佳模型
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 20 \
    --eval_only \
    --load_path outputs/pdtsp_osm_20/osm_train_20_<timestamp>/epoch-50.pt \
    --val_dataset './datasets/osm_val_20.pkl' \
    --val_size 100 \
    --T_max 1500 \
    --val_m 4 \
    --no_saving \
    --no_tb
```

### 示例 2：在多个 GPU 上训练 50 节点模型

```bash
# 生成数据集
python create_osm_val_dataset.py \
    --graph_size 50 \
    --num_samples 100 \
    --output ./datasets/osm_val_50.pkl

# 使用 GPU 1 和 2 训练（DDP 会自动启用）
CUDA_VISIBLE_DEVICES=1,2 python run.py \
    --problem pdtsp_osm \
    --graph_size 50 \
    --batch_size 400 \
    --epoch_size 10000 \
    --epoch_end 100 \
    --T_train 300 \
    --val_dataset './datasets/osm_val_50.pkl' \
    --val_size 100 \
    --val_m 2 \
    --run_name osm_train_50_multi_gpu \
    --checkpoint_epochs 10
```

### 示例 3：快速测试新配置

```bash
# 生成小规模测试数据
python create_osm_val_dataset.py \
    --graph_size 20 \
    --num_samples 10 \
    --output ./datasets/osm_val_test.pkl

# 快速训练测试
CUDA_VISIBLE_DEVICES=0 python run.py \
    --problem pdtsp_osm \
    --graph_size 20 \
    --batch_size 10 \
    --epoch_size 50 \
    --epoch_end 2 \
    --T_train 20 \
    --val_dataset './datasets/osm_val_test.pkl' \
    --val_size 10 \
    --val_m 1 \
    --run_name quick_test \
    --no_tb \
    --no_saving
```

---

## 结果文件说明

### 1. 训练检查点
```
outputs/pdtsp_osm_<graph_size>/<run_name>_<timestamp>/
├── epoch-1.pt          # 第 1 轮检查点
├── epoch-5.pt          # 第 5 轮检查点
├── epoch-10.pt         # 第 10 轮检查点
└── args.json           # 训练参数配置
```

### 2. 验证结果
```
results/
└── pdtsp_results_<timestamp>.json  # 包含所有验证实例的结果
```

**JSON 文件结构：**
```json
{
  "timestamp": "20251007_143929",
  "problem": "pdtsp",
  "graph_size": 20,
  "T_max": 1500,
  "val_size": 100,
  "instances": [
    {
      "instance_id": 0,
      "best_cost": 123.45,
      "best_path": [0, 1, 2, ...],
      "path_length": 41,
      "coordinates": [[x1, y1], [x2, y2], ...]
    },
    ...
  ]
}
```

### 3. TensorBoard 日志
```
logs/pdtsp_osm_<graph_size>/<run_name>_<timestamp>/
└── events.out.tfevents.*
```

**查看日志：**
```bash
tensorboard --logdir logs/pdtsp_osm_20/
```

---

## 注意事项

1. **GPU 内存**：较大的图规模（50、100 节点）需要更多显存，建议相应调小 `batch_size`

2. **OSM 数据缓存**：首次使用某个地点时会下载 OSM 数据并缓存到 `cache/` 目录，后续会直接使用缓存

3. **结果路径**：当前验证结果保存路径 `results/` 是硬编码的，无法通过参数修改

4. **时间戳**：每次运行会自动添加时间戳到 `run_name`，避免覆盖之前的结果

5. **分布式训练**：当检测到多个 GPU 时会自动启用 DDP，除非使用 `--no_DDP` 禁用

6. **验证频率**：训练过程中每轮结束后都会进行验证

7. **强连通分量修复**：代码已修复 OSM 图连通性问题。系统会自动检测并只使用最大强连通分量，确保所有节点之间都是可达的，避免出现距离为 1000000000 的异常值。如果看到警告信息 "Graph was not strongly connected"，这是正常的，表明系统已自动处理

---

## 常见问题

### Q1: 如何修改结果保存路径？
**A:** 当前结果路径硬编码在 `agent/utils.py` 第 155 行，需要手动修改代码。

### Q2: 训练中断后如何恢复？
**A:** 使用 `--resume` 参数指定检查点目录：- 🟢 绿色星星：Depot（仓库/起点）
  - 🔵 蓝色圆圈：Pickup 节点（取货点）
  - 🟠 橙色方块：Delivery 节点（送货点）
  - 🔴 红色路径：求解的路径
```bash
--resume outputs/pdtsp_osm_20/osm_train_20_20251007T120000 \
--epoch_start 10
```

### Q3: 如何使用自定义的 OSM 地点？
**A:** 生成数据集时通过 `--place` 参数指定，训练时通过 `--osm_place` 参数指定：
```bash
--osm_place "Manhattan, New York, USA"
```

### Q4: 评估时显存不足怎么办？
**A:** 减小 `--val_batch_size`：
```bash
--val_batch_size 100  # 默认为 1000
```

### Q5: 如何只保存最优模型？
**A:** 增大 `--checkpoint_epochs` 的值，或在训练后手动删除不需要的检查点。

### Q6: 为什么某些实例的评估 cost 异常大（如 1000000000）？
**A:** 这个问题已在最新版本中修复。旧版本数据集可能包含不可达的节点对。解决方法：
1. **重新生成数据集**（推荐）：使用修复后的代码重新生成验证数据集
2. **使用检查脚本**：运行 `python check_generated_data.py` 检查数据集质量
3. **过滤异常实例**：使用 `python analyze_distance_issue.py` 找出有问题的实例并排除

修复说明：代码现在会自动使用 OSM 图的最大强连通分量，确保所有节点之间都是可达的。

### Q7: 可视化时如何切换不同的实例？
**A:** 使用 `--index` 参数指定实例索引，或直接修改 `vis_osm.py` 中的 `INSTANCE_INDEX` 变量：
```bash
python vis_osm.py --index 2  # 可视化第 2 个实例
```

### Q8: 可视化提示找不到 OSM 图怎么办？
**A:** 确保 `--osm_place` 参数与生成数据集时使用的地点名称一致，且网络连接正常（首次使用会下载地图数据）。

### Q9: 如何批量可视化所有实例？
**A:** 使用 bash 循环：
```bash
for i in {0..9}; do python vis_osm.py --index $i; done
```

---

## 附录：OSM 地点示例

常用的 OSM 地点字符串格式：

- `"Boca Raton, Florida, USA"`
- `"Manhattan, New York, USA"`
- `"Boston, Massachusetts, USA"`
- `"San Francisco, California, USA"`
- `"London, England"`
- `"Paris, France"`
- `"Tokyo, Japan"`

**注意**：地点名称必须是 OSMnx 能够识别的有效字符串。

---

## 更新日志

- **2025-10-09**: 
  - **重要修复**：修复距离矩阵异常值问题
  - 自动使用 OSM 图的最大强连通分量，确保所有节点可达
  - 避免出现距离为 1000000000 的异常值
  - 添加数据质量检查脚本 `check_generated_data.py` 和 `analyze_distance_issue.py`
  - **建议重新生成所有验证和训练数据集**

- **2025-10-07**: 
  - 初始版本，包含完整的训练和评估流程
  - 添加结果可视化功能 (`vis_osm.py`)
  - 修复 `--val_size` 参数不生效的问题
  - 修复训练时保存验证结果的问题（现在仅在 `--eval_only` 时保存）

## 2025-10-16
- 更新了可视化代码vis_osm.py，之前存在问题，读取的best_path未完成从邻接表转为实际访问顺序的功能，导致绘制的路径明显不对。
- 新建了chek_vis.py，用于检查验证集中各节点间的最短路径是否正确
- 新建了get_path_order.py，用于直接读取结果中实例的最佳路径并转为实际访问顺序