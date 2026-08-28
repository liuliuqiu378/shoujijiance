"""对验证集推理生成 pred.json（用于本地评估，复刻官方协议）。

输出格式同提交样例：[{"image_id":"x.jpg","detections":[{"bbox":[x1,y1,x2,y2],"confidence":f,"class":"phone"}]}]
这里 conf 阈值设很低(0.05)，保留全量框，由 evaluate.py 按官方 0.25 过滤。
"""
import os
import json
import argparse
from ultralytics import YOLO
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-root", default="phone_detect/dataset/images/val")
    ap.add_argument("--out", default="phone_detect/val_pred.json")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.weights)
    imgs = sorted([f for f in os.listdir(args.img_root) if f.lower().endswith(".jpg")])
    results = []
    for name in imgs:
        path = os.path.join(args.img_root, name)
        with Image.open(path) as im:
            W, H = im.size
        dets = []
        pred = model.predict(path, conf=args.conf, iou=args.iou,
                             imgsz=args.imgsz, device=args.device, verbose=False)
        for box in pred[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            x1 = max(0.0, min(x1, W - 1)); y1 = max(0.0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W)); y2 = max(y1 + 1, min(y2, H))
            dets.append({"bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                         "confidence": round(conf, 4), "class": "phone"})
        results.append({"image_id": name, "detections": dets})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"验证集推理完成：{args.out}，{len(results)} 张")


if __name__ == "__main__":
    main()
