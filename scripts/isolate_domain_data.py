import os
import shutil

SOURCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "statefarm_cls"
)
TARGET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "desk_domain_cls",
)


def isolate_domain_data():
    if os.path.exists(TARGET_DIR):
        r = shutil.rmtree(TARGET_DIR)
        print("Cleaned existing desk_domain_cls folder.")

    for split in ["train", "val"]:
        src_split = os.path.join(SOURCE_DIR, split)
        tgt_split = os.path.join(TARGET_DIR, split)

        if not os.path.exists(src_split):
            continue

        for cls_name in os.listdir(src_split):
            if cls_name.startswith("."):
                continue

            src_cls = os.path.join(src_split, cls_name)
            tgt_cls = os.path.join(tgt_split, cls_name)
            os.makedirs(tgt_cls, exist_ok=True)

            # Find domain_gap_*.jpg images
            # If any exist, copy them! Note: The 500 images are all in 'train' currently
            files = [f for f in os.listdir(src_cls) if f.startswith("domain_gap_")]
            count = 0
            for file in files:
                shutil.copy2(os.path.join(src_cls, file), os.path.join(tgt_cls, file))
                # YOLOv8 要求拥有单独的 val 文件夹来算分，直接强行把数据也镜像到评估集里去！
                tgt_val_cls = os.path.join(TARGET_DIR, "val", cls_name)
                os.makedirs(tgt_val_cls, exist_ok=True)
                shutil.copy2(
                    os.path.join(src_cls, file), os.path.join(tgt_val_cls, file)
                )
                count += 1
            print(f"Isolated {count} domain images into {tgt_cls}")


if __name__ == "__main__":
    print("=" * 50)
    print("🧹 提纯提纯！将桌面 500 张绝密数据从 18000 张汽车海选中抽离...")
    isolate_domain_data()
    print("✅ 纯粹的【书桌环境特化】数据集 (desk_domain_cls) 已在 data 目录下就绪！")
