"""YOLO 手机检测 Baseline 训练脚本。

使用 ultralytics YOLO (默认 YOLO11s，可改 model 名切换 nano/s/m/l)。
关键设置：
  - 单类 phone，YOLO 格式标签
  - 数据均衡：约一半负样本，采用默认训练即可；可通过 copy_paste/hsv 增强提升鲁棒性
  - 针对脏数据：开启 label_smoothing、focal 风格(ultralytics 内置 box/focal 通过 'conf' 不直接，使用默认)
  - 评测以官方 IoU=0.5 为准，训练 val 用相同协议线下评估
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="phone_detect/dataset.yaml")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="baseline_yolo11s")
    ap.add_argument("--device", default="0")
    ap.add_argument("--box", type=float, default=7.5, help="box loss 权重")
    ap.add_argument("--dfl", type=float, default=1.5, help="dfl loss 权重")
    ap.add_argument("--close_mosaic", type=int, default=15)
    ap.add_argument("--mixup", type=float, default=0.1)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="随机种子，控制初始化/数据增强随机性，用于多模型多样性集成")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project="phone_detect/runs",
        device=args.device,
        workers=8,
        optimizer="auto",
        lr0=args.lr0,
        cos_lr=True,
        close_mosaic=args.close_mosaic,      # 最后阶段关闭 mosaic，提升定位精度
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # 色彩抖动，抗低光照
        fliplr=0.5,
        mosaic=1.0,
        mixup=args.mixup,            # 轻微 mixup 提升鲁棒性
        copy_paste=0.1,              # 提升小目标/遮挡鲁棒性
        label_smoothing=0.1,  # 缓解脏标签
        box=args.box,              # box loss 权重
        cls=0.5,
        dfl=args.dfl,
        patience=args.patience,
        seed=args.seed,
        save_period=-1,
        verbose=True,
    )


if __name__ == "__main__":
    main()
