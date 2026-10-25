# MVPDTSP 容量实验

以下命令均在 `n2s_mv_cvrp` 目录执行。每单在取货到送货期间占用 1 个容量单位；不传容量参数即无容量约束。一次运行只使用一组容量配置。

完整训练脚本均默认使用 GPU 6、7，依次训练 50 和 100 节点各 200 个 epoch。训练每个 epoch 后验证，保留验证成绩最好的 10 个 `epoch-*.pt`，另存一份最佳权重为 `optimal-<epoch>.pt`；训练结束后用这份最佳权重评估 256 个实例，四组评估均使用 `T_max=3000`：

```bash
./mv_cvrp/mvpdp_cv_2.sh  # 2 车：50 节点容量 10；100 节点容量 15
./mv_cvrp/mvpdp_cv_5.sh  # 5 车：50 节点容量 5；100 节点容量 7
```

可在命令前设置 `CUDA_VISIBLE_DEVICES` 指定其他 GPU。两个脚本均独立包含训练和评估命令，文件顶部有分别对应 50、100 节点的参数区；各自串行运行两组配置。权重保存在 `mv_cvrp/output/`，日志与评估 JSON 保存在 `mv_cvrp/result/`。
中断或服务器重启后，重新执行同一脚本即可从 `resume-latest.pt` 记录的最近完整 epoch 自动续训；前 10 名与最佳权重的记录也会恢复。已完成训练但尚未评估的配置会直接评估，已完成评估的配置会跳过。可更换 GPU 编号，但应保持两张 GPU；修改实验参数后若要开始独立实验，运行 `NEW_RUN=1 ./mv_cvrp/mvpdp_cv_2.sh`（5 车脚本同理）。

```bash
# 20 节点、2 车、每车容量 3：GPU 6/7，完整训练和评估 1 个 epoch
./mv_cvrp/train_cvrp_1v.sh

# 2 车异构容量示例
VEHICLE_CAPACITIES="7 3" ./mv_cvrp/train_cvrp_1v.sh

# 5 车同构容量示例
python run.py --problem mvpdtsp --graph_size 20 --num_vehicles 5 --capacity 7 \
  --batch_size 64 --epoch_size 256 --epoch_end 1 --T_train 5 --T_max 10 \
  --val_dataset ../pdp_n2s/datasets/pdp_20.pkl --val_size 16 --val_batch_size 16 --no_DDP \
  --output_dir mv_cvrp/output/5v7 --results_dir mv_cvrp/result/5v7 \
  --log_dir mv_cvrp/output/5v7/tensorboard

# 5 车异构容量示例：车辆 0～4 分别为 7、7、7、3、3
python run.py --problem mvpdtsp --graph_size 20 --num_vehicles 5 \
  --vehicle_capacities 7 7 7 3 3 \
  --batch_size 64 --epoch_size 256 --epoch_end 1 --T_train 5 --T_max 10 \
  --val_dataset ../pdp_n2s/datasets/pdp_20.pkl --val_size 16 --val_batch_size 16 --no_DDP \
  --output_dir mv_cvrp/output/5v77733 --results_dir mv_cvrp/result/5v77733 \
  --log_dir mv_cvrp/output/5v77733/tensorboard

# 推理时使用与权重相同的容量配置；替换 checkpoint 路径
python run.py --problem mvpdtsp --graph_size 20 --num_vehicles 5 \
  --vehicle_capacities 7 7 7 3 3 --eval_only --load_path PATH/TO/CHECKPOINT.pt \
  --val_dataset ../pdp_n2s/datasets/pdp_20.pkl --val_size 16 --val_batch_size 16 \
  --T_max 10 --no_DDP --results_dir mv_cvrp/result/5v77733_eval \
  --output_dir mv_cvrp/output/5v77733_eval --log_dir mv_cvrp/output/5v77733_eval/tensorboard
```

权重和 TensorBoard 文件保存在 `mv_cvrp/output/`，评估 JSON、命令记录及运行日志保存在 `mv_cvrp/result/`。`--capacity` 与 `--vehicle_capacities` 不能同时指定；容量必须为正整数，列表长度必须等于车辆数。结果 JSON 包含 `vehicle_capacities` 和各车 `vehicle_peak_loads`。
