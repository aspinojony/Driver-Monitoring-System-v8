"""
训练结果自动归档工具

每次模型训练结束后调用 archive_training_run()，
将所有训练产物（权重、图表、CSV、配置）复制到带时间戳的归档目录中，
并生成训练摘要报告。

归档路径: reports/training_archives/{脚本名}_{YYYYMMDD_HHMMSS}/
"""

import os
import shutil
import glob
from datetime import datetime


def archive_training_run(run_dir, script_name, extra_info=None):
    """
    将一次训练的所有产物归档到 reports/training_archives/ 下。

    Args:
        run_dir: YOLO 训练产出的目录路径（包含 weights/, results.csv 等）
        script_name: 调用此函数的训练脚本名称（如 'train_yolo_cls'）
        extra_info: 可选的额外信息字典，会写入摘要报告

    Returns:
        archive_dir: 归档目录的绝对路径
    """
    if not os.path.exists(run_dir):
        print(f"⚠️ 警告：训练输出目录不存在: {run_dir}")
        print("跳过归档。")
        return None

    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 创建归档目录: reports/training_archives/{脚本名}_{时间戳}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{script_name}_{timestamp}"
    archive_dir = os.path.join(
        project_root, "reports", "training_archives", archive_name
    )
    os.makedirs(archive_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"📦 开始归档训练结果...")
    print(f"   来源: {run_dir}")
    print(f"   目标: {archive_dir}")
    print(f"{'=' * 60}")

    # 要归档的文件模式列表
    patterns_to_archive = [
        # 模型权重
        "weights/best.pt",
        "weights/last.pt",
        # 训练指标
        "results.csv",
        "args.yaml",
        # 训练曲线和图表
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        # 训练批次可视化图片
        "train_batch*.jpg",
        # 验证批次可视化图片
        "val_batch*.jpg",
        # F1/PR/P/R 曲线（如果有）
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        # labels 相关
        "labels.jpg",
        "labels_correlogram.jpg",
    ]

    archived_files = []
    for pattern in patterns_to_archive:
        full_pattern = os.path.join(run_dir, pattern)
        matched_files = glob.glob(full_pattern)
        for src_file in matched_files:
            # 计算相对路径以保持目录结构
            rel_path = os.path.relpath(src_file, run_dir)
            dst_file = os.path.join(archive_dir, rel_path)
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)
            archived_files.append(rel_path)

    # 生成训练摘要报告
    summary_path = os.path.join(archive_dir, "training_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("           训练结果归档摘要报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"训练脚本: {script_name}\n")
        f.write(f"归档时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"原始目录: {run_dir}\n")
        f.write(f"归档目录: {archive_dir}\n")
        f.write(f"\n{'─' * 40}\n")
        f.write(f"归档文件列表 ({len(archived_files)} 个文件):\n")
        f.write(f"{'─' * 40}\n")
        for af in sorted(archived_files):
            f.write(f"  ✓ {af}\n")

        # 如果有 results.csv，提取最终指标
        results_csv_path = os.path.join(run_dir, "results.csv")
        if os.path.exists(results_csv_path):
            f.write(f"\n{'─' * 40}\n")
            f.write("最终训练指标 (最后一个 epoch):\n")
            f.write(f"{'─' * 40}\n")
            try:
                with open(results_csv_path, "r") as csvf:
                    lines = csvf.readlines()
                    if len(lines) >= 2:
                        headers = [h.strip() for h in lines[0].split(",")]
                        last_values = [v.strip() for v in lines[-1].split(",")]
                        for header, value in zip(headers, last_values):
                            f.write(f"  {header}: {value}\n")
            except Exception as e:
                f.write(f"  (读取 results.csv 失败: {e})\n")

        # 如果有 args.yaml，记录训练参数
        args_yaml_path = os.path.join(run_dir, "args.yaml")
        if os.path.exists(args_yaml_path):
            f.write(f"\n{'─' * 40}\n")
            f.write("训练参数 (args.yaml):\n")
            f.write(f"{'─' * 40}\n")
            try:
                with open(args_yaml_path, "r") as yf:
                    f.write(yf.read())
            except Exception as e:
                f.write(f"  (读取 args.yaml 失败: {e})\n")

        # 额外信息
        if extra_info:
            f.write(f"\n{'─' * 40}\n")
            f.write("额外信息:\n")
            f.write(f"{'─' * 40}\n")
            for key, value in extra_info.items():
                f.write(f"  {key}: {value}\n")

    archived_files.append("training_summary.txt")

    print(f"\n✅ 归档完成！共保存 {len(archived_files)} 个文件")
    print(f"📁 归档路径: {archive_dir}")
    print(f"📄 训练摘要: {summary_path}")
    print(f"{'=' * 60}\n")

    return archive_dir


def _test_archive():
    """
    使用已有的训练输出目录做归档测试。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 尝试用已有的 domain_adapted_cls_final 做测试
    test_candidates = [
        os.path.join(project_root, "runs", "classify", "domain_adapted_cls_final"),
        os.path.join(project_root, "data", "weights", "yolov8n_driver_cls"),
    ]

    test_dir = None
    for candidate in test_candidates:
        if os.path.exists(candidate):
            test_dir = candidate
            break

    if test_dir is None:
        print("❌ 没有找到可用于测试的训练输出目录")
        return

    print(f"🧪 使用已有目录进行归档测试: {test_dir}")
    result = archive_training_run(
        run_dir=test_dir,
        script_name="archive_test",
        extra_info={"备注": "这是一次归档功能测试"},
    )

    if result:
        print(f"\n🎉 归档测试成功！请查看: {result}")
    else:
        print("\n❌ 归档测试失败")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        _test_archive()
    else:
        print("用法: python save_training_results.py --test")
        print(
            "在训练脚本中使用: from save_training_results import archive_training_run"
        )
