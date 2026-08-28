"""B榜增强训练：train = 原train + breview(1955张人工核查), val = breview。

训练完成后，对保存的多个 epoch 权重在 breview 验证集上算官方综合分
(0.5*mAP + 0.3*P + 0.2*R)，挑最高的作为 best_breview.pt。

用法:
  python train_breview.py --model yolo11m.pt --name v20_b11m_breview --seed 11 --epochs 120
"""
import os
import json
import argparse
import subprocess
import glob
from ultralytics import YOLO

ROOT = "/home/hmn-cjy/liuliuqiu/fangzuobi/phone_detect"
BIMG = "/home/hmn-cjy/liuliuqiu/fangzuobi/data/Btest/test_b/images"


def build_yaml():
    """生成 dataset_breview.yaml: train=原train+breview, val=breview。"""
    yaml_path = f"{ROOT}/dataset_breview.yaml"
    content = f"""path: {ROOT}/dataset
train:
  - images/train_v12
  - images/breview
val: images/breview
nc: 1
names: ['phone']
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--save_period", type=int, default=10)
    args = ap.parse_args()

    yaml_path = build_yaml()
    model = YOLO(args.model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=f"{ROOT}/runs",
        device=args.device,
        workers=8,
        optimizer="auto",
        lr0=0.001,
        cos_lr=True,
        close_mosaic=15,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        label_smoothing=0.1,
        box=7.5, dfl=1.5, cls=0.5,
        patience=20,
        seed=args.seed,
        save_period=args.save_period,
        verbose=True,
    )
    print(f"[{args.name}] 训练完成，开始挑 breview 最佳权重...")


if __name__ == "__main__":
    main()
