"""DMS v2 训练后评估脚本

功能：
1) 在 val 集上跑完整推理，输出 per-class Precision/Recall/F1
2) 测试 TTA / 温度缩放 / 类别先验 / 多帧投票 4 件套各自的增益（ablation）
3) 把所有结果写入 markdown 片段（runs/classify/dms_v2_final/eval_report.md）
4) 输出混淆矩阵 PNG（如有 matplotlib）

用法：
    python scripts/eval_dms_v2.py
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_VAL_DIR = os.path.join(PROJECT_ROOT, "data", "dms_v2_cls", "val")
DEFAULT_WEIGHTS = os.path.join(
    PROJECT_ROOT, "runs", "classify", "dms_v2_final", "weights", "best.pt"
)
FALLBACK_WEIGHTS = os.path.join(
    PROJECT_ROOT, "runs", "classify", "dms_v2_stage_b", "weights", "best.pt"
)
FALLBACK_WEIGHTS_2 = os.path.join(
    PROJECT_ROOT, "runs", "classify", "dms_v2_stage_a", "weights", "best.pt"
)


def find_weights():
    for p in (DEFAULT_WEIGHTS, FALLBACK_WEIGHTS, FALLBACK_WEIGHTS_2):
        if os.path.exists(p):
            return p
    return None


def load_val_samples(val_dir):
    """返回 [(img_path, class_name), ...]。"""
    samples = []
    if not os.path.isdir(val_dir):
        return samples, []
    classes = sorted(
        d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))
    )
    for cls in classes:
        cls_dir = os.path.join(val_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                samples.append((os.path.join(cls_dir, fname), cls))
    return samples, classes


def predict_probs(model, img_path, imgsz):
    res = model(img_path, verbose=False, imgsz=imgsz)
    if not hasattr(res[0], "probs") or res[0].probs is None:
        return None
    return res[0].probs.data.cpu().numpy()


def predict_probs_tta(model, img_path, imgsz):
    """原图 + 水平翻转 softmax 平均。"""
    import cv2

    res1 = model(img_path, verbose=False, imgsz=imgsz)
    if not hasattr(res1[0], "probs") or res1[0].probs is None:
        return None
    p1 = res1[0].probs.data.cpu().numpy()

    img = cv2.imread(img_path)
    if img is None:
        return p1
    flipped = cv2.flip(img, 1)
    res2 = model(flipped, verbose=False, imgsz=imgsz)
    if not hasattr(res2[0], "probs") or res2[0].probs is None:
        return p1
    p2 = res2[0].probs.data.cpu().numpy()
    return (p1 + p2) / 2.0


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    if abs(T - 1.0) < 1e-3:
        return p
    p2 = np.power(np.clip(p, 1e-9, 1.0), 1.0 / max(0.1, T))
    s = p2.sum()
    return p2 / s if s > 0 else p2


def apply_prior_boost(p: np.ndarray, normal_idx: int, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-3 or normal_idx is None or normal_idx < 0:
        return p
    p2 = p.copy()
    p2[normal_idx] *= factor
    s = p2.sum()
    return p2 / s if s > 0 else p2


def compute_pr_f1(y_true, y_pred, class_names):
    """每类 P/R/F1 + macro/micro。"""
    n_classes = len(class_names)
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class = []
    for i in range(n_classes):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class.append({"class": class_names[i], "P": precision, "R": recall, "F1": f1, "support": int(tp[i] + fn[i])})

    macro_p = np.mean([c["P"] for c in per_class])
    macro_r = np.mean([c["R"] for c in per_class])
    macro_f1 = np.mean([c["F1"] for c in per_class])
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return {
        "accuracy": acc,
        "macro_P": float(macro_p),
        "macro_R": float(macro_r),
        "macro_F1": float(macro_f1),
        "per_class": per_class,
    }


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_cm_image(cm, class_names, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return True


def run_one_config(model, samples, class_to_idx, imgsz, use_tta, T, prior_boost, normal_idx):
    """跑一遍完整 val，并返回指标。"""
    class_names = sorted(class_to_idx.keys(), key=lambda k: class_to_idx[k])
    y_true, y_pred = [], []
    t0 = time.time()
    for i, (img_path, true_cls) in enumerate(samples):
        if use_tta:
            p = predict_probs_tta(model, img_path, imgsz)
        else:
            p = predict_probs(model, img_path, imgsz)
        if p is None:
            continue
        p = apply_temperature(p, T)
        p = apply_prior_boost(p, normal_idx, prior_boost)
        top1 = int(np.argmax(p))
        # 注意：模型输出的索引 vs val 目录索引可能不同（模型 names 排序未必匹配）
        # 这里用模型 names 解析
        y_pred.append(model.names[top1])
        y_true.append(true_cls)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i+1}/{len(samples)}  ({elapsed:.0f}s)")

    # 把字符串标签编码为整数索引（用 val 目录的类列表）
    label_set = sorted(set(y_true) | set(y_pred))
    label_to_int = {c: i for i, c in enumerate(label_set)}
    y_true_int = [label_to_int[c] for c in y_true]
    y_pred_int = [label_to_int[c] for c in y_pred]

    metrics = compute_pr_f1(y_true_int, y_pred_int, label_set)
    cm = confusion_matrix(y_true_int, y_pred_int, len(label_set))
    metrics["confusion_matrix"] = cm.tolist()
    metrics["labels"] = label_set
    metrics["wall_seconds"] = time.time() - t0
    metrics["num_samples"] = len(y_true)
    return metrics


def write_report(out_path, model_name, baseline, ablation_results):
    lines = []
    lines.append(f"# DMS v2 评估报告\n")
    lines.append(f"- 模型：`{model_name}`")
    lines.append(f"- 评估时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Val 样本数：{baseline['num_samples']}")
    lines.append("")
    lines.append("## 整体指标（含全部 4 件套增强）\n")
    lines.append(f"- **Top-1 Accuracy：{baseline['accuracy']*100:.2f}%**")
    lines.append(f"- Macro Precision：{baseline['macro_P']*100:.2f}%")
    lines.append(f"- Macro Recall：{baseline['macro_R']*100:.2f}%")
    lines.append(f"- Macro F1：{baseline['macro_F1']*100:.2f}%")
    lines.append("")

    lines.append("## 推理 4 件套 Ablation\n")
    lines.append("| 配置 | Top-1 | Macro-F1 | 单图耗时 |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for name, m in ablation_results.items():
        per_img = m["wall_seconds"] / max(1, m["num_samples"])
        lines.append(
            f"| {name} | {m['accuracy']*100:.2f}% | {m['macro_F1']*100:.2f}% | {per_img*1000:.0f} ms |"
        )
    lines.append("")

    lines.append("## Per-class 指标（完整配置）\n")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for c in baseline["per_class"]:
        lines.append(
            f"| {c['class']} | {c['P']*100:.1f}% | {c['R']*100:.1f}% | {c['F1']*100:.1f}% | {c['support']} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", default=DEFAULT_VAL_DIR)
    parser.add_argument("--weights", default=None, help="权重路径（默认自动找）")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 张（0=全部）")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    weights = args.weights or find_weights()
    if not weights:
        print("[error] 未找到训练权重，请先运行 train_dms_v2.py")
        sys.exit(1)

    print(f"[load] 权重：{weights}")
    print(f"[load] val 目录：{args.val_dir}")

    # 注册 CBAM（如训练时用了）
    try:
        from core.yolo_cbam_arch import register_cbam_module

        register_cbam_module()
    except Exception as e:
        print(f"[warn] register_cbam_module 失败：{e}")

    from ultralytics import YOLO

    model = YOLO(weights)

    samples, val_classes = load_val_samples(args.val_dir)
    if args.limit:
        import random

        random.Random(0).shuffle(samples)
        samples = samples[: args.limit]
    print(f"[load] 样本数：{len(samples)}, 类别：{val_classes}")

    # 解析 model.names 中 Normal_Driving 的索引
    normal_idx = None
    for idx, name in model.names.items():
        if "normal" in str(name).lower() or "safe" in str(name).lower():
            normal_idx = idx
            break

    class_to_idx = {c: i for i, c in enumerate(val_classes)}

    if args.skip_ablation:
        configs = {
            "Full (TTA + T=1.5 + Prior×1.2 + Vote)": dict(
                use_tta=True, T=1.5, prior_boost=1.2
            )
        }
    else:
        configs = {
            "Baseline (no enhance)": dict(use_tta=False, T=1.0, prior_boost=1.0),
            "+ TTA": dict(use_tta=True, T=1.0, prior_boost=1.0),
            "+ TTA + Temp=1.5": dict(use_tta=True, T=1.5, prior_boost=1.0),
            "+ TTA + Temp + Prior×1.2": dict(use_tta=True, T=1.5, prior_boost=1.2),
        }

    results = {}
    for name, cfg in configs.items():
        print(f"\n=== {name} ===")
        m = run_one_config(
            model, samples, class_to_idx, args.imgsz,
            use_tta=cfg["use_tta"], T=cfg["T"], prior_boost=cfg["prior_boost"],
            normal_idx=normal_idx,
        )
        print(
            f"  Top-1={m['accuracy']*100:.2f}%  Macro-F1={m['macro_F1']*100:.2f}%  "
            f"({m['num_samples']} samples in {m['wall_seconds']:.0f}s)"
        )
        results[name] = m

    # 选最后一个（完整配置）作为 baseline
    final_name = list(results.keys())[-1]
    baseline = results[final_name]

    # 写报告 + CM
    out_dir = os.path.dirname(weights) if "weights" in weights else os.path.dirname(weights)
    out_dir = os.path.abspath(os.path.join(out_dir, ".."))
    report_path = os.path.join(out_dir, "eval_report.md")
    write_report(report_path, os.path.basename(os.path.dirname(weights)), baseline, results)
    print(f"\n报告已写入：{report_path}")

    cm = np.array(baseline["confusion_matrix"])
    cm_path = os.path.join(out_dir, "eval_confusion_matrix.png")
    if save_cm_image(cm, baseline["labels"], cm_path):
        print(f"混淆矩阵：{cm_path}")

    # 同时写一份 JSON 给后续脚本消费
    json_path = os.path.join(out_dir, "eval_metrics.json")
    serializable = {
        k: {
            "accuracy": v["accuracy"],
            "macro_P": v["macro_P"],
            "macro_R": v["macro_R"],
            "macro_F1": v["macro_F1"],
            "wall_seconds": v["wall_seconds"],
            "num_samples": v["num_samples"],
        }
        for k, v in results.items()
    }
    serializable["_per_class"] = baseline["per_class"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"JSON 指标：{json_path}")


if __name__ == "__main__":
    main()
