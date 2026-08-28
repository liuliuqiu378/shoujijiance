"""测试集推理并生成提交 JSON。

输出格式（与官方一致）：
[
  {"image_id": "xxx.jpg", "detections":[{"bbox":[x1,y1,x2,y2],"confidence":f,"class":"phone"}]},
  ...
]
注意：
  - 空图返回 "detections": []
  - bbox 为像素坐标 x_min,y_min,x_max,y_max
  - 置信度范围 [0,1]，评测脚本会过滤 <0.25 的框
"""
import os
import json
import argparse
from ultralytics import YOLO
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="训练产生的 best.pt")
    ap.add_argument("--test-dir", default="data/test_without_label/test/images")
    ap.add_argument("--out", default="phone_detect/submit.json")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="推理时保留低置信度框，官方再按0.25过滤；这里设低以便提交全量")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--min_area", type=float, default=0.0,
                    help="过滤占图面积比例低于该值的框(0~1)，用于砍黑边/小误检")
    args = ap.parse_args()

    model = YOLO(args.weights)
    imgs = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith(".jpg")])

    results = []
    for name in imgs:
        path = os.path.join(args.test_dir, name)
        # 校验图片可读取
        try:
            with Image.open(path) as im:
                W, H = im.size
        except Exception:
            results.append({"image_id": name, "detections": []})
            continue
        dets = []
        pred = model.predict(path, conf=args.conf, iou=args.iou,
                             imgsz=args.imgsz, device=args.device, verbose=False)
        for box in pred[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            # 裁剪到图像范围内
            x1 = max(0.0, min(x1, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W))
            y2 = max(y1 + 1, min(y2, H))
            # 面积过滤：占图比例低于阈值则丢弃（砍黑边/小误检）
            if args.min_area > 0:
                area_ratio = ((x2 - x1) * (y2 - y1)) / (W * H)
                if area_ratio < args.min_area:
                    continue
            dets.append({
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "confidence": round(conf, 4),
                "class": "phone",
            })
        results.append({"image_id": name, "detections": dets})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已写出 {args.out}，共 {len(results)} 张图")


if __name__ == "__main__":
    main()
