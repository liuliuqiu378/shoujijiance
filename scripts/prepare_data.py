"""数据清洗与划分脚本。

任务：考场手机检测（YOLO 格式，单类 phone，class_id=0）。
输入：
  data/train/train/images  + data/train/train/labels  (rar 解压结果)
输出：
  phone_detect/dataset/{images,labels}/{train,val}
  phone_detect/dataset.yaml

清洗策略：
  1. 剔除无法用 PIL 打开的损坏图片及其对应标签文件（rar 解压失败导致的空/损坏jpg）。
  2. 保留空标签文件作为负样本（真实考场多数无手机）。
  3. 校验标签格式（class_id=0，坐标在[0,1]），过滤越界框行。
  4. 按图片ID随机划分 train/val (8:2)，保证两张集合图片互不重叠。
"""
import os
import glob
import json
import random
from PIL import Image

SEED = 42
VAL_RATIO = 0.2
SRC_IMG = "data/train/train/images"
SRC_LAB = "data/train/train/labels"
OUT_ROOT = "phone_detect/dataset"
DATA_YAML = "phone_detect/dataset.yaml"


def is_valid_image(path):
    try:
        Image.open(path).verify()
        return True
    except Exception:
        return False


def clean_label_lines(path):
    """返回清洗后的标签行列表；若图片应被跳过(全部无效且非空集)返回 None 表示忽略。"""
    if not os.path.exists(path):
        return []  # 无标签 -> 空标签(负样本)
    with open(path) as f:
        raw = [l.strip() for l in f if l.strip()]
    cleaned = []
    for l in raw:
        p = l.split()
        if len(p) != 5:
            continue
        try:
            cid, x, y, w, h = map(float, p)
        except ValueError:
            continue
        if cid != 0:
            continue
        # 越界框丢弃；中心点允许极小误差，宽高必须正
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue
        cleaned.append(f"{int(cid)} {x} {y} {w} {h}")
    return cleaned


def main():
    random.seed(SEED)
    os.makedirs(f"{OUT_ROOT}/images/train", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/images/val", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/labels/train", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/labels/val", exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(SRC_IMG, "*.jpg")))
    valid = []
    corrupt = 0
    for p in img_files:
        if is_valid_image(p):
            valid.append(os.path.basename(p))
        else:
            corrupt += 1
    print(f"总图片 {len(img_files)}，损坏 {corrupt}，有效 {len(valid)}")

    # 划分
    random.shuffle(valid)
    n_val = int(len(valid) * VAL_RATIO)
    val_set = set(valid[:n_val])
    train_set = set(valid[n_val:])

    stats = {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0,
             "train_boxes": 0, "val_boxes": 0}

    def copy_subset(id_set, split):
        for name in id_set:
            sid = name[:-4]  # 去 .jpg
            lab_src = os.path.join(SRC_LAB, sid + ".txt")
            lines = clean_label_lines(lab_src)
            # 复制图片
            import shutil
            shutil.copy(os.path.join(SRC_IMG, name),
                        os.path.join(OUT_ROOT, "images", split, name))
            # 写标签（即使为空也写空文件，保留负样本）
            with open(os.path.join(OUT_ROOT, "labels", split, sid + ".txt"), "w") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")
            if lines:
                stats[f"{split}_pos"] += 1
                stats[f"{split}_boxes"] += len(lines)
            else:
                stats[f"{split}_neg"] += 1

    copy_subset(train_set, "train")
    copy_subset(val_set, "val")

    print("统计:", json.dumps(stats, indent=2, ensure_ascii=False))

    yaml_content = f"""# 考场手机检测数据集 (自动生成)
path: {os.path.abspath(OUT_ROOT)}
train: images/train
val: images/val
test: ../data/test_without_label/test/images

nc: 1
names: ['phone']
"""
    with open(DATA_YAML, "w") as f:
        f.write(yaml_content)
    print(f"已写出 {DATA_YAML}")


if __name__ == "__main__":
    main()
