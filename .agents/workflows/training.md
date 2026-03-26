---
description: 模型训练工作流 — 确保每次训练的数据、图表、权重都被保存
---

# 模型训练工作流

每次执行模型训练时，**必须**确保训练产出物被完整保存。

## 训练前检查

1. 确认训练数据集路径存在
2. 确认 `reports/training_archives/` 目录可写

## 执行训练

项目中有 4 个训练脚本，所有脚本均已内置自动归档功能：

// turbo-all

```bash
# 分类模型初始训练
python scripts/train_yolo_cls.py

# 目标检测模型训练
python scripts/train_yolo.py

# 夜间数据微调
python scripts/finetune_yolo_cls.py

# 领域自适应微调
python scripts/finetune_domain_gap.py
```

## 训练后自动保存（已内置）

训练结束后，脚本会自动调用 `save_training_results.archive_training_run()`，执行以下操作：

1. 在 `reports/training_archives/{脚本名}_{YYYYMMDD_HHMMSS}/` 创建归档目录
2. 复制所有训练产物：
   - `weights/best.pt` — 最优模型权重
   - `weights/last.pt` — 最后一轮模型权重
   - `results.csv` — 每个 epoch 的训练指标
   - `results.png` — 训练曲线图（loss、accuracy 等）
   - `confusion_matrix.png` — 混淆矩阵
   - `confusion_matrix_normalized.png` — 归一化混淆矩阵
   - `args.yaml` — 训练参数配置
   - `train_batch*.jpg` — 训练批次可视化
   - `val_batch*.jpg` — 验证批次可视化
   - 各种曲线图（F1_curve、PR_curve、P_curve、R_curve）
3. 生成 `training_summary.txt` 训练摘要报告

## 手动归档（如果需要）

```bash
python scripts/save_training_results.py --test
```

## 重要规则

> **每次训练都必须确认 `reports/training_archives/` 下生成了对应的归档目录。训练结果不可丢失！**
