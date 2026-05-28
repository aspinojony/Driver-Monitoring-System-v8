"""一次性把图表引用与数据修正写回 毕业论文.md（绕过 Edit 工具的权限问题）"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
md = ROOT / "毕业论文.md"
text = md.read_text(encoding="utf-8")

replacements = [
    ("[图 4-4：Stage B 20 轮训练损失与 Top-1 曲线及早停位置]",
     """Stage B 的训练动态如图 4-4 与图 4-4b 所示，两阶段的整体 Top-1 演化对比如图 4-4c 所示。

![图 4-4 Stage B 训练全量指标](assets/figures/fig_4_4_stage_b_results.png)

**图 4-4** Stage B 训练全量指标曲线（Ultralytics 原始输出）

![图 4-4b Stage B 中文版三联图](assets/figures/fig_4_4b_stage_b_curves_zh.png)

**图 4-4b** Stage B 中文版关键指标三联图（最佳 epoch=3，Top-1=65.99%）

![图 4-4c Stage A 与 Stage B 串联 Top-1 演化曲线](assets/figures/fig_4_4b_stage_ab_compare.png)

**图 4-4c** Stage A 与 Stage B 累计 Top-1 演化对比。蓝线为 Stage A 全量预训练 30 轮，红线为 Stage B 冻结 backbone 微调 17 轮。Stage A 在 epoch 18 达到 67.45% 峰值，Stage B 在 epoch 3 达到 65.99% 峰值后早停触发"""),

    ("[图 5-1：8 类验证集混淆矩阵热力图]",
     """![图 5-1 验证集混淆矩阵（原始计数）](assets/figures/fig_5_1_confusion_matrix.png)

**图 5-1** 1000 张 subject-independent 验证集上的 8 类混淆矩阵（原始计数）

![图 5-1b 归一化混淆矩阵](assets/figures/fig_5_1b_confusion_matrix_normalized.png)

**图 5-1b** 归一化混淆矩阵（按真实类别归一化，单元格为召回率）。对角线值越大、非对角线值越小表示分类越准确"""),

    ("[图 5-2：Per-class F1 柱状排序图]",
     """![图 5-2 8 类驾驶行为的精确率/召回率/F1 对比](assets/figures/fig_5_2_per_class_metrics.png)

**图 5-2** 8 类驾驶行为的精确率、召回率、F1 值对比（按 F1 由高至低排序）"""),

    ("[表 4-2：8 类训练集与验证集样本数分布]",
     """[表 4-2：8 类训练集与验证集样本数分布]

![图 4-1b 验证集 8 类样本数分布](assets/figures/fig_4_1b_val_class_distribution.png)

**图 4-1b** Subject-Independent 切分后验证集 8 类样本数分布（共 1000 张）"""),

    ("表5-7 推理增强4件套的消融对比\n\n值得注意的是，",
     """表5-7 推理增强4件套的消融对比

![图 5-3a 推理增强 4 件套消融柱状对比](assets/figures/fig_5_3a_ablation_inference_enhance.png)

**图 5-3a** 推理增强 4 件套的消融对比（Top-1 准确率、宏 F1 与单图推理耗时三维度可视化）

值得注意的是，"""),

    ("Stage B的训练曲线呈现与Stage A明显不同的特征：由于Backbone已经收敛，Stage B的训练损失波动较小（从0.107轻微下降至0.088），验证集Top-1在64%–66%区间内稳定振荡。在epoch 7处达到当前最佳，之后10轮验证集Top-1未能超越epoch 7，触发早停。最终发布权重为epoch 7的best.pt，",
     "Stage B的训练曲线呈现与Stage A明显不同的特征：由于Backbone已经收敛，Stage B的训练损失波动较小（从0.107轻微下降至0.088），验证集Top-1在64%–66%区间内稳定振荡。验证集Top-1在epoch 3处达到最佳65.99%，此后14轮训练验证指标未能超越该值，最终在epoch 17触发patience=10的早停机制（计数从最佳点起算）。最终发布权重为epoch 3的best.pt，"),

    ("""| 1 | 0.107 | 65.3% | 1.918 | 2.77e-04 | 加载 Stage A best.pt |
| 5 | 0.102 | 65.2% | 2.469 | 7.54e-04 | 平稳 |
| 7 | 0.097 | 65.9% | 2.301 | 5.96e-04 | **最佳** |
| 10 | 0.093 | 64.5% | 2.373 | 4.85e-04 | 未超越 epoch 7 |
| 15 | 0.090 | 64.9% | 2.313 | 1.78e-04 | 未超越 epoch 7 |
| 17 | 0.089 | 64.7% | 2.305 | 1.05e-04 | patience=10 触发早停 |""",
     """| 1 | 0.107 | 65.26% | 1.918 | 2.77e-04 | 加载 Stage A best.pt |
| 2 | 0.103 | 64.20% | 2.150 | 5.51e-04 | 短暂下行 |
| 3 | 0.110 | **65.99%** | 2.217 | 8.12e-04 | **最佳**（早停参考点） |
| 5 | 0.102 | 65.16% | 2.469 | 7.54e-04 | 平稳 |
| 7 | 0.096 | 65.89% | 2.188 | 6.63e-04 | 接近最佳但未超越 |
| 10 | 0.093 | 64.46% | 2.373 | 4.85e-04 | 持续未超越 |
| 15 | 0.090 | 64.87% | 2.313 | 1.78e-04 | 持续未超越 |
| 17 | 0.090 | 65.45% | 2.332 | 8.71e-05 | patience=10 触发早停 |"""),

    ("""| 15 | 0.326 | 64.1% | 2.627 | 4.64e-04 | Val Loss 峰值 |
| 20 | 0.223 | 67.3% | 2.037 | 2.53e-04 | **Top-1 峰值** |""",
     """| 15 | 0.326 | 64.07% | 2.627 | 4.64e-04 | Val Loss 峰值 |
| 18 | 0.250 | **67.45%** | 1.945 | 3.35e-04 | **Top-1 峰值** |
| 20 | 0.223 | 67.27% | 2.037 | 2.53e-04 | 峰值附近平稳 |"""),

    ("第20轮Top-1达到峰值约67.3%，此后开始小幅波动至训练结束。最终发布的Stage A最佳权重为patience机制保留的epoch 20权重，",
     "第18轮Top-1达到峰值约67.45%（第20轮接近为67.27%），此后小幅波动至训练结束。最终发布的Stage A最佳权重为patience机制保留的epoch 18权重，"),
]

ok = 0
missing = []
for old, new in replacements:
    if old in text:
        text = text.replace(old, new, 1)
        ok += 1
        print(f"[ok]   {repr(old[:60])}")
    else:
        missing.append(old)
        print(f"[miss] {repr(old[:60])}")

md.write_text(text, encoding="utf-8")
print(f"\n{ok}/{len(replacements)} 替换完成。文件已写回。")
if missing:
    print(f"\n未匹配的 {len(missing)} 项需要手动确认:")
    for m in missing:
        print("  ---")
        print(f"  {m[:200]}")
