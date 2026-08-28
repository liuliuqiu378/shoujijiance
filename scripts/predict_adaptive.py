"""自适应conf提交生成: 单目标图放低阈值(拉召回), 多目标图提高阈值(保精度)。
按模型原始检出数(低阈值估计)分桶:
  - 0个检出: 忽略(无检测)
  - 1个检出: conf>=single_conf (默认0.40)
  - >=2个检出: conf>=multi_conf (默认0.60)

用法:
  python predict_adaptive.py --weights ... --test-dir ... --out ... \
      --single-conf 0.40 --multi-conf 0.60 --imgsz 1280 --device 0
"""
import os
import json
import argparse
from ultralytics import YOLO
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--single-conf", type=float, default=0.40)
    ap.add_argument("--multi-conf", type=float, default=0.60)
    ap.add_argument("--est-conf", type=float, default=0.05,
                    help="估计'模型认为有几个目标'用的低阈值")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--min_area", type=float, default=0.0)
    args = ap.parse_args()

    model = YOLO(args.weights)
    imgs = sorted([f for f in os.listdir(args.test_dir) if f.lower().endswith(".jpg")])
    results = []
    for name in imgs:
        p = os.path.join(args.test_dir, name)
        try:
            with Image.open(p) as im:
                W, H = im.size
        except Exception:
            results.append({"image_id": name, "detections": []})
            continue
        pr = model.predict(p, conf=0.001, iou=0.5, imgsz=args.imgsz,
                           device=args.device, verbose=False)
        all_dets = []
        for box in pr[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c = float(box.conf[0])
            all_dets.append((c, [x1, y1, x2, y2]))
        # 估计检出数(低阈值)
        cnt = sum(1 for c, b in all_dets if c >= args.est_conf)
        # 分桶阈值
        if cnt == 0:
            keep_conf = 1.0  # 无检出 -> 全过滤
        elif cnt == 1:
            keep_conf = args.single_conf
        else:
            keep_conf = args.multi_conf
        dets = []
        for c, b in all_dets:
            if c < keep_conf:
                continue
            x1, y1, x2, y2 = b
            x1 = max(0.0, min(x1, W - 1))
            y1 = max(0.0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W))
            y2 = max(y1 + 1, min(y2, H))
            if args.min_area > 0:
                if ((x2 - x1) * (y2 - y1)) / (W * H) < args.min_area:
                    continue
            dets.append({"bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                         "confidence": round(c, 4), "class": "phone"})
        results.append({"image_id": name, "detections": dets})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    n = sum(1 for x in results if x["detections"])
    print(f"已写出 {args.out}，共 {len(results)} 张，有检测 {n} 张 ({100*n/len(results):.1f}%)")


if __name__ == "__main__":
    main()
