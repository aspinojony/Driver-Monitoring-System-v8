"""为毕业论文批量生成补充图表。

生成内容：
  - 图 5-2: Per-class F1/Precision/Recall 柱状图（中文标签）
  - 图 4-3b: Stage A 简化训练曲线（中文标签，单 PNG 含 loss/acc/val_loss/lr）
  - 图 4-4b: Stage B 简化训练曲线
  - 图 4-x: Stage A vs Stage B Top-1 对比曲线（单图叠加）
  - 图 5-3a: 推理增强 4 件套 Top-1 与耗时对比
  - 图 3-x: 8 类训练样本数分布柱状图（如可获取）

输出目录: assets/figures/
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体配置（macOS）
plt.rcParams["font.sans-serif"] = ["Heiti SC", "Hiragino Sans GB", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EVAL_JSON = ROOT / "runs/classify/dms_v2_final/eval_metrics.json"
STAGE_A_CSV = ROOT / "runs/classify/dms_v2_stage_a/results.csv"
STAGE_B_CSV = ROOT / "runs/classify/dms_v2_stage_b/results.csv"


CHN_NAME = {
    "Normal_Driving": "正常驾驶",
    "Texting": "发短信",
    "Talking_on_Phone": "打电话",
    "Operating_Radio": "操作中控",
    "Drinking": "喝水",
    "Reaching_Behind": "向后取物",
    "Hair_and_Makeup": "整理仪容",
    "Talking_to_Passenger": "与乘客交谈",
}


def gen_per_class_metrics():
    """图 5-2: Per-class P/R/F1 柱状图"""
    data = json.loads(EVAL_JSON.read_text())
    rows = data["_per_class"]
    rows = sorted(rows, key=lambda r: -r["F1"])

    labels = [CHN_NAME.get(r["class"], r["class"]) for r in rows]
    P = [r["P"] * 100 for r in rows]
    R = [r["R"] * 100 for r in rows]
    F1 = [r["F1"] * 100 for r in rows]

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    ax.bar(x - width, P, width, label="精确率 Precision", color="#3b82f6")
    ax.bar(x, R, width, label="召回率 Recall", color="#10b981")
    ax.bar(x + width, F1, width, label="F1 值", color="#ef4444")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("百分比 (%)")
    ax.set_ylim(0, 110)
    ax.set_title("图 5-2  8 类驾驶行为的精确率 / 召回率 / F1 值对比", fontsize=13, pad=10)
    ax.legend(loc="upper right", ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for i, (p, r, f) in enumerate(zip(P, R, F1)):
        ax.text(i + width, f + 1.5, f"{f:.1f}", ha="center", fontsize=8, color="#ef4444")

    plt.tight_layout()
    out = FIG_DIR / "fig_5_2_per_class_metrics.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out.name}")


def gen_stage_curves_zh(csv_path, title, out_name):
    """图 4-3b / 4-4b: 单阶段训练曲线（中文标签）"""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=160)

    ax = axes[0]
    ax.plot(df["epoch"], df["train/loss"], color="#3b82f6", label="训练损失")
    ax.plot(df["epoch"], df["val/loss"], color="#ef4444", label="验证损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练 / 验证损失")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linestyle="--")

    ax = axes[1]
    top1 = df["metrics/accuracy_top1"] * 100
    top5 = df["metrics/accuracy_top5"] * 100
    ax.plot(df["epoch"], top1, color="#10b981", label="Top-1 准确率", linewidth=2)
    ax.plot(df["epoch"], top5, color="#f59e0b", label="Top-5 准确率", linestyle="--")
    best_idx = top1.idxmax()
    ax.axvline(df["epoch"].iloc[best_idx], color="gray", linestyle=":", alpha=0.6)
    ax.scatter([df["epoch"].iloc[best_idx]], [top1.iloc[best_idx]],
               color="red", zorder=5, s=40, label=f"最佳 epoch={df['epoch'].iloc[best_idx]}, Top-1={top1.iloc[best_idx]:.1f}%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("准确率 (%)")
    ax.set_title("验证集准确率")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")

    ax = axes[2]
    ax.plot(df["epoch"], df["lr/pg0"], color="#8b5cf6")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("学习率")
    ax.set_title("余弦退火学习率")
    ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / out_name
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out.name}")


def gen_stage_ab_compare():
    """补充图: Stage A vs Stage B Top-1 对比曲线"""
    a = pd.read_csv(STAGE_A_CSV)
    b = pd.read_csv(STAGE_B_CSV)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    ax.plot(a["epoch"], a["metrics/accuracy_top1"] * 100, color="#3b82f6",
            label=f"Stage A (全量预训练, {len(a)} 轮)", linewidth=2, marker="o", markersize=4)
    ax.plot(np.array(b["epoch"]) + len(a), b["metrics/accuracy_top1"] * 100,
            color="#ef4444",
            label=f"Stage B (冻结 backbone 微调, 早停 epoch {b['metrics/accuracy_top1'].idxmax()+1})",
            linewidth=2, marker="s", markersize=4)

    a_best = a["metrics/accuracy_top1"].max() * 100
    b_best = b["metrics/accuracy_top1"].max() * 100
    ax.axhline(a_best, color="#3b82f6", linestyle=":", alpha=0.5)
    ax.axhline(b_best, color="#ef4444", linestyle=":", alpha=0.5)
    ax.text(2, a_best + 0.5, f"Stage A 最佳 {a_best:.2f}%", color="#3b82f6", fontsize=9)
    ax.text(len(a) + 1, b_best + 0.5, f"Stage B 最佳 {b_best:.2f}%", color="#ef4444", fontsize=9)

    ax.axvline(len(a), color="gray", linestyle="--", alpha=0.5)
    ax.text(len(a) + 0.5, 15, "↑ 切换至 Stage B", color="gray", fontsize=9)

    ax.set_xlabel("累计 Epoch")
    ax.set_ylabel("验证集 Top-1 准确率 (%)")
    ax.set_title("图 4-4b  两阶段迁移学习的 Top-1 准确率演化", fontsize=13, pad=10)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0, 75)

    plt.tight_layout()
    out = FIG_DIR / "fig_4_4b_stage_ab_compare.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out.name}")


def gen_ablation_chart():
    """图 5-3a: 推理增强 4 件套消融对比"""
    data = json.loads(EVAL_JSON.read_text())
    labels_full = ["Baseline (no enhance)", "+ TTA", "+ TTA + Temp=1.5", "+ TTA + Temp + Prior×1.2"]
    labels_zh = ["Baseline\n基线", "+ TTA\n翻转增强", "+ TTA + 温度\nT=1.5", "+ 全部 4 件套\n(Normal 先验×1.2)"]
    acc = [data[k]["accuracy"] * 100 for k in labels_full]
    f1 = [data[k]["macro_F1"] * 100 for k in labels_full]
    secs = [data[k]["wall_seconds"] / data[k]["num_samples"] * 1000 for k in labels_full]

    x = np.arange(len(labels_zh))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=160)
    bars1 = ax1.bar(x - width / 2, acc, width, label="Top-1 准确率", color="#3b82f6")
    bars2 = ax1.bar(x + width / 2, f1, width, label="宏 F1", color="#10b981")
    ax1.set_ylabel("百分比 (%)")
    ax1.set_ylim(60, 75)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_zh, fontsize=10)
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    for bars, vals in [(bars1, acc), (bars2, f1)]:
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                     ha="center", fontsize=8.5)

    ax2 = ax1.twinx()
    ax2.plot(x, secs, color="#ef4444", marker="D", linewidth=2, label="单图耗时")
    ax2.set_ylabel("单图推理耗时 (ms)", color="#ef4444")
    ax2.tick_params(axis="y", labelcolor="#ef4444")
    for xi, yi in zip(x, secs):
        ax2.text(xi, yi + 0.3, f"{yi:.1f} ms", color="#ef4444", ha="center", fontsize=8.5)

    ax1.set_title("图 5-3a  推理增强 4 件套的消融对比（Top-1 / Macro-F1 / 单图耗时）",
                  fontsize=13, pad=10)

    plt.tight_layout()
    out = FIG_DIR / "fig_5_3a_ablation_inference_enhance.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out.name}")


def gen_class_distribution():
    """图 4-1b: 8 类训练 / 验证集样本数分布（用 eval per_class support 近似 val 分布）"""
    data = json.loads(EVAL_JSON.read_text())
    rows = data["_per_class"]
    rows = sorted(rows, key=lambda r: -r["support"])

    labels = [CHN_NAME.get(r["class"], r["class"]) for r in rows]
    support = [r["support"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
    bars = ax.bar(labels, support, color=[
        "#3b82f6", "#ef4444", "#10b981", "#f59e0b",
        "#8b5cf6", "#ec4899", "#14b8a6", "#6366f1"
    ])
    ax.set_ylabel("验证集样本数")
    ax.set_title("图 4-1b  Subject-Independent 验证集的 8 类样本数分布（共 1000 张）",
                 fontsize=12.5, pad=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for b, s in zip(bars, support):
        ax.text(b.get_x() + b.get_width() / 2, s + 4, str(s),
                ha="center", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = FIG_DIR / "fig_4_1b_val_class_distribution.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out.name}")


if __name__ == "__main__":
    print(f"输出目录: {FIG_DIR}")
    gen_per_class_metrics()
    gen_stage_curves_zh(STAGE_A_CSV, "图 4-3b  Stage A（YOLOv8s-cls + CBAM 全量预训练，30 轮）训练动态", "fig_4_3b_stage_a_curves_zh.png")
    gen_stage_curves_zh(STAGE_B_CSV, "图 4-4b  Stage B（冻结 backbone 前 10 层，分类头微调，20 轮）训练动态", "fig_4_4b_stage_b_curves_zh.png")
    gen_stage_ab_compare()
    gen_ablation_chart()
    gen_class_distribution()
    print("\n全部图表已生成。")
