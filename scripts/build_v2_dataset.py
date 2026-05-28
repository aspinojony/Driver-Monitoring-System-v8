"""构建 v2 训练数据集（dms_v2_cls）

合并策略：
1) StateFarm 原始数据（state-farm-distracted-driver-detection/imgs/train/cN/）
   + 按 driver_imgs_list.csv 的 subject 严格切 train/val（避免数据泄露）
   + 类别重映射：c1+c3 → Texting, c2+c4 → Talking_on_Phone（左右合并）
2) AUC Distracted Driver V2（data/auc_dd_v2/）
   + 自动探测目录布局：split (train+test) / flat (cN) / per-subject
   + 类别重映射：注意 AUC 的类别 ID 与 StateFarm 不同
3) 自录数据（data/desk_domain_v2/manifest.csv）
   + 按 session_id 切 train/val

输出：data/dms_v2_cls/{train,val}/{8 个类别}/

8 类：
    Normal_Driving, Texting, Talking_on_Phone, Operating_Radio,
    Drinking, Reaching_Behind, Hair_and_Makeup, Talking_to_Passenger
"""

import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# StateFarm 原始数据
SF_ROOT = os.path.join(PROJECT_ROOT, "state-farm-distracted-driver-detection")
SF_DRIVER_LIST = os.path.join(SF_ROOT, "driver_imgs_list.csv")
SF_IMGS_DIR = os.path.join(SF_ROOT, "imgs", "train")

# AUC Distracted Driver V2
# 期望目录之一：
#   data/auc_dd_v2/{train,test}/cN/...     (split 模式，最常见)
#   data/auc_dd_v2/cN/...                   (flat 模式)
#   data/auc_dd_v2/{p001,p002,...}/cN/...   (per-subject 模式)
AUC_ROOT = os.path.join(PROJECT_ROOT, "data", "auc_dd_v2")

# 自录数据
DESK_V2_ROOT = os.path.join(PROJECT_ROOT, "data", "desk_domain_v2")
DESK_MANIFEST = os.path.join(DESK_V2_ROOT, "manifest.csv")

# 输出
OUT_ROOT = os.path.join(PROJECT_ROOT, "data", "dms_v2_cls")

# StateFarm 类别 → v2 类别
SF_TO_V2 = {
    "c0": "Normal_Driving",
    "c1": "Texting",  # 原 Texting_Right
    "c2": "Talking_on_Phone",  # 原 Talking_on_Phone_Right
    "c3": "Texting",  # 原 Texting_Left → 合并
    "c4": "Talking_on_Phone",  # 原 Talking_on_Phone_Left → 合并
    "c5": "Operating_Radio",
    "c6": "Drinking",
    "c7": "Reaching_Behind",
    "c8": "Hair_and_Makeup",
    "c9": "Talking_to_Passenger",
}

# AUC DD V2 类别 → v2 类别（⚠️ ID 顺序与 StateFarm 不同！）
# 参考 Eraqi et al., 2019 "Driver Distraction Identification with an Ensemble of Convolutional Neural Networks"
AUC_TO_V2 = {
    "c0": "Normal_Driving",
    "c1": "Talking_on_Phone",  # Phone Right
    "c2": "Talking_on_Phone",  # Phone Left → 合并
    "c3": "Texting",  # Text Right
    "c4": "Texting",  # Text Left → 合并
    "c5": "Operating_Radio",  # Adjusting Radio
    "c6": "Drinking",
    "c7": "Hair_and_Makeup",  # ← 注意：AUC 的 c7 是 Hair/Makeup（StateFarm 是 Reaching Behind）
    "c8": "Reaching_Behind",
    "c9": "Talking_to_Passenger",
}

V2_CLASSES = sorted(set(SF_TO_V2.values()))


def reset_output():
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    for split in ("train", "val"):
        for cls in V2_CLASSES:
            os.makedirs(os.path.join(OUT_ROOT, split, cls), exist_ok=True)


def split_subjects(val_ratio: float, seed: int):
    """按 subject 切 train/val，避免同人不同帧泄露。"""
    if not os.path.exists(SF_DRIVER_LIST):
        print(f"[warn] 未找到 {SF_DRIVER_LIST}，跳过 StateFarm 数据合并")
        return None, None

    subjects = set()
    with open(SF_DRIVER_LIST, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subjects.add(row["subject"])

    subjects = sorted(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)

    val_count = max(1, int(round(len(subjects) * val_ratio)))
    val_subjects = set(subjects[:val_count])
    train_subjects = set(subjects[val_count:])
    return train_subjects, val_subjects


def copy_statefarm(train_subjects, val_subjects, max_per_class_per_split=None):
    """从 StateFarm 原始 imgs/train/cN/ 复制到 dms_v2_cls，按 subject 切分。"""
    if train_subjects is None:
        return defaultdict(int), defaultdict(int)

    # 建立 img → (subject, classname) 索引
    img_meta = {}
    with open(SF_DRIVER_LIST, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_meta[row["img"]] = (row["subject"], row["classname"])

    train_count = defaultdict(int)
    val_count = defaultdict(int)

    for sf_class in sorted(SF_TO_V2.keys()):
        v2_class = SF_TO_V2[sf_class]
        src_dir = os.path.join(SF_IMGS_DIR, sf_class)
        if not os.path.isdir(src_dir):
            continue

        files = sorted(os.listdir(src_dir))
        random.Random(0).shuffle(files)

        for fname in files:
            meta = img_meta.get(fname)
            if meta is None:
                continue
            subject, _ = meta
            if subject in train_subjects:
                split = "train"
                if max_per_class_per_split and train_count[v2_class] >= max_per_class_per_split:
                    continue
                train_count[v2_class] += 1
            elif subject in val_subjects:
                split = "val"
                if max_per_class_per_split and val_count[v2_class] >= max_per_class_per_split:
                    continue
                val_count[v2_class] += 1
            else:
                continue

            src = os.path.join(src_dir, fname)
            dst_name = f"sf_{sf_class}_{subject}_{fname}"
            dst = os.path.join(OUT_ROOT, split, v2_class, dst_name)
            try:
                os.symlink(src, dst)  # 用软链接避免 17k 张实拷贝
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)

    return train_count, val_count


def copy_desk_v2(val_session_ratio: float, seed: int):
    """读 manifest.csv，按 session_id 切 train/val，合并到 dms_v2_cls。"""
    if not os.path.exists(DESK_MANIFEST):
        print(f"[info] 未找到自录 manifest（{DESK_MANIFEST}），跳过自录数据合并")
        return defaultdict(int), defaultdict(int)

    rows_by_session = defaultdict(list)
    with open(DESK_MANIFEST, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_session[row["session_id"]].append(row)

    sessions = sorted(rows_by_session.keys())
    if len(sessions) < 2:
        print(
            f"[warn] 自录 session 数仅 {len(sessions)} 个，无法按 session 切。"
            f"将随机抽 {val_session_ratio*100:.0f}% 文件作为 val。"
        )
        # 退化方案
        val_sessions = set()
    else:
        rng = random.Random(seed)
        rng.shuffle(sessions)
        val_count = max(1, int(round(len(sessions) * val_session_ratio)))
        val_sessions = set(sessions[:val_count])

    train_count = defaultdict(int)
    val_count = defaultdict(int)

    rng = random.Random(seed + 1)
    for session, rows in rows_by_session.items():
        for row in rows:
            v2_class = row["class"]
            src = os.path.join(DESK_V2_ROOT, v2_class, row["filename"])
            if not os.path.exists(src):
                continue
            if session in val_sessions:
                split = "val"
                val_count[v2_class] += 1
            elif len(sessions) < 2:
                # 退化随机切
                split = "val" if rng.random() < val_session_ratio else "train"
                if split == "val":
                    val_count[v2_class] += 1
                else:
                    train_count[v2_class] += 1
            else:
                split = "train"
                train_count[v2_class] += 1

            dst = os.path.join(OUT_ROOT, split, v2_class, f"desk_{row['filename']}")
            try:
                os.symlink(src, dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)

    return train_count, val_count


# ===========================================================================
# AUC Distracted Driver V2 处理
# ===========================================================================
def discover_auc_layout(auc_root: str):
    """探测 AUC DD V2 数据集的目录布局。

    返回值之一：
      - "split"        ：data/auc_dd_v2/{train,test}/cN/
      - "flat"         ：data/auc_dd_v2/cN/
      - "per-subject"  ：data/auc_dd_v2/{p001,...}/cN/
      - None           ：未找到合规目录
    """
    if not os.path.isdir(auc_root):
        return None

    # split：train/test 都在
    train_dir = os.path.join(auc_root, "train")
    test_dir = os.path.join(auc_root, "test")
    val_dir = os.path.join(auc_root, "val")
    has_train = os.path.isdir(train_dir)
    has_test = os.path.isdir(test_dir) or os.path.isdir(val_dir)
    if has_train and has_test:
        return "split"

    # flat：顶级有 cN
    if os.path.isdir(os.path.join(auc_root, "c0")):
        return "flat"

    # per-subject：顶级是 p* / subject* / driver*
    entries = sorted(
        d for d in os.listdir(auc_root) if os.path.isdir(os.path.join(auc_root, d))
    )
    if entries and any(
        d.lower().startswith(("p0", "p1", "p2", "subject", "driver", "user"))
        for d in entries
    ):
        # 进一步检查内部是否有 cN 子目录
        sample = os.path.join(auc_root, entries[0])
        if os.path.isdir(os.path.join(sample, "c0")):
            return "per-subject"

    return None


def _link_or_copy(src: str, dst: str):
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _is_image(fname: str) -> bool:
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))


def copy_auc(
    auc_root: str,
    val_ratio: float,
    seed: int,
    max_per_class_per_split=None,
):
    """合并 AUC DD V2 到 dms_v2_cls，自动适配三种布局。"""
    if not os.path.isdir(auc_root):
        print(f"[info] 未找到 AUC 数据目录 {auc_root}，跳过 AUC")
        return defaultdict(int), defaultdict(int)

    layout = discover_auc_layout(auc_root)
    if layout is None:
        print(
            f"[warn] {auc_root} 存在但布局不识别。期望:\n"
            f"        - {auc_root}/train/cN/  与  {auc_root}/test/cN/  (split)\n"
            f"        - {auc_root}/cN/                                 (flat)\n"
            f"        - {auc_root}/{{p001,...}}/cN/                    (per-subject)"
        )
        return defaultdict(int), defaultdict(int)

    print(f"[info] AUC 目录布局识别为: {layout}")

    train_count = defaultdict(int)
    val_count = defaultdict(int)
    rng = random.Random(seed + 7)

    def add_file(src_path: str, v2_class: str, target_split: str, prefix: str = "auc"):
        if (
            max_per_class_per_split
            and (train_count if target_split == "train" else val_count)[v2_class]
            >= max_per_class_per_split
        ):
            return
        fname = os.path.basename(src_path)
        dst_name = f"{prefix}_{v2_class}_{fname}"
        dst = os.path.join(OUT_ROOT, target_split, v2_class, dst_name)
        _link_or_copy(src_path, dst)
        if target_split == "train":
            train_count[v2_class] += 1
        else:
            val_count[v2_class] += 1

    if layout == "split":
        # 直接用作者已切好的 train / test 划分
        for src_split, target_split in (("train", "train"), ("test", "val"), ("val", "val")):
            split_dir = os.path.join(auc_root, src_split)
            if not os.path.isdir(split_dir):
                continue
            for cN, v2_class in AUC_TO_V2.items():
                src_dir = os.path.join(split_dir, cN)
                if not os.path.isdir(src_dir):
                    continue
                files = sorted(f for f in os.listdir(src_dir) if _is_image(f))
                rng.shuffle(files)
                for fname in files:
                    add_file(os.path.join(src_dir, fname), v2_class, target_split)

    elif layout == "flat":
        # 没有 subject 信息，只能随机切
        print(
            "[warn] AUC flat 模式：无 subject 信息，按 val_ratio={:.0%} 随机切。"
            "如可能请使用 split 或 per-subject 布局以获得更可靠的指标。".format(val_ratio)
        )
        for cN, v2_class in AUC_TO_V2.items():
            src_dir = os.path.join(auc_root, cN)
            if not os.path.isdir(src_dir):
                continue
            files = sorted(f for f in os.listdir(src_dir) if _is_image(f))
            rng.shuffle(files)
            for fname in files:
                target_split = "val" if rng.random() < val_ratio else "train"
                add_file(os.path.join(src_dir, fname), v2_class, target_split)

    elif layout == "per-subject":
        # 严格按 subject 切
        subjects = sorted(
            d
            for d in os.listdir(auc_root)
            if os.path.isdir(os.path.join(auc_root, d))
            and os.path.isdir(os.path.join(auc_root, d, "c0"))
        )
        rng_sub = random.Random(seed + 11)
        rng_sub.shuffle(subjects)
        val_n = max(1, int(round(len(subjects) * val_ratio)))
        val_subjects = set(subjects[:val_n])
        train_subjects = set(subjects[val_n:])
        print(
            f"      AUC train subjects={len(train_subjects)}, val subjects={len(val_subjects)}"
        )
        print(f"      val: {sorted(val_subjects)}")
        for subj in subjects:
            target_split = "val" if subj in val_subjects else "train"
            for cN, v2_class in AUC_TO_V2.items():
                src_dir = os.path.join(auc_root, subj, cN)
                if not os.path.isdir(src_dir):
                    continue
                files = sorted(f for f in os.listdir(src_dir) if _is_image(f))
                rng.shuffle(files)
                for fname in files:
                    add_file(
                        os.path.join(src_dir, fname),
                        v2_class,
                        target_split,
                        prefix=f"auc_{subj}",
                    )

    return train_count, val_count


def print_stats(title, sf_train, sf_val, auc_train, auc_val, desk_train, desk_val):
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)
    header = (
        f"{'class':<22}{'SF tr':>8}{'SF val':>8}{'AUC tr':>8}{'AUC val':>9}"
        f"{'desk tr':>9}{'desk val':>10}{'TOTAL':>10}"
    )
    print(header)
    print("-" * 86)
    totals = {"sf_t": 0, "sf_v": 0, "auc_t": 0, "auc_v": 0, "d_t": 0, "d_v": 0}
    for cls in V2_CLASSES:
        s_t, s_v = sf_train[cls], sf_val[cls]
        a_t, a_v = auc_train[cls], auc_val[cls]
        d_t, d_v = desk_train[cls], desk_val[cls]
        total = s_t + s_v + a_t + a_v + d_t + d_v
        totals["sf_t"] += s_t
        totals["sf_v"] += s_v
        totals["auc_t"] += a_t
        totals["auc_v"] += a_v
        totals["d_t"] += d_t
        totals["d_v"] += d_v
        print(
            f"{cls:<22}{s_t:>8}{s_v:>8}{a_t:>8}{a_v:>9}{d_t:>9}{d_v:>10}{total:>10}"
        )
    print("-" * 86)
    print(
        f"{'TOTAL':<22}{totals['sf_t']:>8}{totals['sf_v']:>8}{totals['auc_t']:>8}"
        f"{totals['auc_v']:>9}{totals['d_t']:>9}{totals['d_v']:>10}"
        f"{sum(totals.values()):>10}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf-val-ratio", type=float, default=0.20, help="StateFarm subject val 比例")
    parser.add_argument("--auc-val-ratio", type=float, default=0.20, help="AUC val 比例（仅在 flat/per-subject 模式生效）")
    parser.add_argument("--desk-val-ratio", type=float, default=0.25, help="自录 session val 比例")
    parser.add_argument(
        "--max-sf-per-class",
        type=int,
        default=0,
        help="单类 StateFarm 上限（0=不限制）。建议设 800-1200 削弱 StateFarm 占比",
    )
    parser.add_argument(
        "--max-auc-per-class",
        type=int,
        default=0,
        help="单类 AUC 上限（0=不限制）。建议设 800-1200",
    )
    parser.add_argument(
        "--auc-root",
        type=str,
        default=AUC_ROOT,
        help=f"AUC DD V2 根目录（默认 {AUC_ROOT}）",
    )
    parser.add_argument(
        "--skip-auc", action="store_true", help="跳过 AUC（即便目录存在）"
    )
    parser.add_argument(
        "--skip-sf", action="store_true", help="跳过 StateFarm"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1/4] 重置输出目录 {OUT_ROOT}")
    reset_output()

    if args.skip_sf:
        print("[2/4] 已跳过 StateFarm")
        sf_train, sf_val = defaultdict(int), defaultdict(int)
    else:
        if not os.path.exists(SF_DRIVER_LIST):
            print(f"[2/4] 未找到 driver_imgs_list.csv，跳过 StateFarm")
            sf_train, sf_val = defaultdict(int), defaultdict(int)
        else:
            print(f"[2/4] 切分 StateFarm subjects（val_ratio={args.sf_val_ratio}）")
            train_subjects, val_subjects = split_subjects(args.sf_val_ratio, args.seed)
            if train_subjects is not None:
                print(
                    f"      train subjects = {len(train_subjects)}, val subjects = {len(val_subjects)}"
                )
                print(f"      val 包含的 subject: {sorted(val_subjects)}")
            sf_train, sf_val = copy_statefarm(
                train_subjects,
                val_subjects,
                max_per_class_per_split=args.max_sf_per_class or None,
            )

    if args.skip_auc:
        print("[3/4] 已跳过 AUC DD V2")
        auc_train, auc_val = defaultdict(int), defaultdict(int)
    else:
        print(f"[3/4] 处理 AUC DD V2（root={args.auc_root}，val_ratio={args.auc_val_ratio}）")
        auc_train, auc_val = copy_auc(
            args.auc_root,
            args.auc_val_ratio,
            args.seed,
            max_per_class_per_split=args.max_auc_per_class or None,
        )

    print(f"[4/4] 合并自录数据（desk val_ratio={args.desk_val_ratio}）")
    desk_train, desk_val = copy_desk_v2(args.desk_val_ratio, args.seed)

    print_stats(
        "dms_v2_cls 数据集统计",
        sf_train,
        sf_val,
        auc_train,
        auc_val,
        desk_train,
        desk_val,
    )
    print(f"\n输出目录：{OUT_ROOT}")
    print("下一步：python scripts/train_dms_v2.py")


if __name__ == "__main__":
    main()
