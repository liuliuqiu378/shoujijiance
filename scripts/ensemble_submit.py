"""集成提交：多权重预测融合（加权 NMS）。

对每张测试图，取多个模型的原始预测框，按 conf 加权融合重叠框，
再用统一 NMS 去重，最后按全局阈值过滤输出。

用法：
  python ensemble_submit.py \
    --weights runs/detect/phone_detect/runs/v3_yolo11m_1280/weights/best.pt \
             runs/detect/phone_detect/runs/baseline_yolo11s/weights/best.pt \
    --imgsz 1280 1280 \
    --out phone_detect/submit_ensemble.json --conf 0.35
"""
import os
import json
import argparse
from PIL import Image
import numpy as np
from ultralytics import YOLO


def infer_one(model, path, imgsz, conf=0.01):
    pred = model.predict(path, conf=conf, iou=0.5, imgsz=imgsz,
                         device="0", verbose=False)
    boxes = []
    for b in pred[0].boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        c = float(b.conf[0])
        boxes.append([x1, y1, x2, y2, c])
    return boxes


def iou_of(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def fuse_wbf(dets, iou_thr=0.5, conf_w=2.0):
    """加权框融合 (Weighted Boxes Fusion)。

    dets: [[x1,y1,x2,y2,conf], ...]
    - 所有 IoU>=iou_thr 的框归为一组，坐标按 conf^conf_w 加权平均；
    - 融合后 conf = 组内各框 conf 的加权(或算术)平均，保留"多模型一致"的高置信。
    - skip_box_thr: 单框 conf < 此值的先丢弃，避免噪声框污染融合。
    """
    if not dets:
        return []
    skip_box_thr = 0.0001
    dets = [d for d in dets if d[4] >= skip_box_thr]
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: -d[4])
    clusters = []  # 每个 cluster: list of det
    used = [False] * len(dets)
    for i, d in enumerate(dets):
        if used[i]:
            continue
        used[i] = True
        cluster = [d]
        for j in range(i + 1, len(dets)):
            if used[j]:
                continue
            if iou_of(d, dets[j]) >= iou_thr:
                used[j] = True
                cluster.append(dets[j])
        clusters.append(cluster)

    fused = []
    for cl in clusters:
        wsum = sum((c ** conf_w) for *_, c in cl)
        if wsum <= 0:
            wsum = len(cl)
        x1 = sum(b[0] * (b[4] ** conf_w) for b in cl) / wsum
        y1 = sum(b[1] * (b[4] ** conf_w) for b in cl) / wsum
        x2 = sum(b[2] * (b[4] ** conf_w) for b in cl) / wsum
        y2 = sum(b[3] * (b[4] ** conf_w) for b in cl) / wsum
        # 融合 conf：组内算术平均（WBF 标准做法），一致命中的框保留高置信
        fc = sum(b[4] for b in cl) / len(cl)
        fused.append([x1, y1, x2, y2, min(1.0, fc)])
    return fused


def fuse_vote(models_dets, iou_thr=0.5):
    """图级 vote 融合：仅保留被两个模型都命中的框（IoU>=iou_thr 配对）。

    models_dets: 每个模型的 dets 列表 [[x1,y1,x2,y2,conf], ...]，按模型分组。
    返回：投票一致的融合框（坐标取两模型均值，conf 取两模型均值）。
    专攻 Precision：丢弃仅单模型命中的框（多为 FP/噪声）。
    """
    if len(models_dets) < 2:
        # 单模型退化为原样返回
        return [d for d in models_dets[0]] if models_dets else []
    fused = []
    # 以第一个模型为基准，去匹配其余每个模型
    base = list(models_dets[0])
    others = models_dets[1:]
    for b in base:
        matched_confs = [b[4]]
        matched_boxes = [b[:4]]
        keep_b = True
        for od in others:
            hit = None
            best_iou = 0.0
            for o in od:
                v = iou_of(b, o)
                if v > best_iou:
                    best_iou, hit = v, o
            if hit is None or best_iou < iou_thr:
                keep_b = False
                break
            matched_confs.append(hit[4])
            matched_boxes.append(hit[:4])
        if keep_b:
            n = len(matched_boxes)
            x1 = sum(x[0] for x in matched_boxes) / n
            y1 = sum(x[1] for x in matched_boxes) / n
            x2 = sum(x[2] for x in matched_boxes) / n
            y2 = sum(x[3] for x in matched_boxes) / n
            fc = sum(matched_confs) / n
            fused.append([x1, y1, x2, y2, min(1.0, fc)])
    return fused
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--imgsz", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="phone_detect/submit_ensemble.json")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf_w", type=float, default=2.0)
    ap.add_argument("--method", choices=["wbf", "wnms", "vote"], default="wbf")
    ap.add_argument("--mode", choices=["val", "test"], default="test")
    ap.add_argument("--img_root", default=None, help="test 模式自定义图片根目录(覆盖默认)")
    ap.add_argument("--min_area", type=float, default=0.0, help="过滤框面积占图比例低于此值的框(治黑边误检), 如0.02")
    ap.add_argument("--scan", action="store_true", help="val 模式扫描多 conf")
    args = ap.parse_args()

    assert len(args.weights) == len(args.imgsz), "weights/imgsz 数量需一致"
    models = [YOLO(w) for w in args.weights]
    if args.mode == "val":
        img_root = "phone_detect/dataset/images/val"
        lab_root = "phone_detect/dataset/labels/val"
        from evaluate import load_gt, xywh_to_xyxy_norm, iou, compute_map_11pt_v2
        gt = load_gt(lab_root)
        img_sizes = {}
        for sid in gt:
            with Image.open(os.path.join(img_root, sid + ".jpg")) as im:
                img_sizes[sid] = im.size
        sids = sorted(gt.keys())
    else:
        img_root = args.img_root or "data/test_without_label/test/images"
        sids = sorted([f[:-4] for f in os.listdir(img_root) if f.lower().endswith(".jpg")])

    # 先收集每个 sid 的融合框（按 conf 升序过滤前的全量）
    raw = {}
    for sid in sids:
        path = os.path.join(img_root, sid + ".jpg")
        with Image.open(path) as im:
            W, H = im.size
        if args.method == "vote":
            per_model = [infer_one(m, path, sz) for m, sz in zip(models, args.imgsz)]
            merged = fuse_vote(per_model, args.iou)
        else:
            all_dets = []
            for m, sz in zip(models, args.imgsz):
                all_dets.extend(infer_one(m, path, sz))
            if args.method == "wbf":
                merged = fuse_wbf(all_dets, args.iou, args.conf_w)
            else:
                merged = weighted_nms(all_dets, args.iou, args.conf_w)
        dets = []
        for x1, y1, x2, y2, c in merged:
            x1 = max(0.0, min(x1, W - 1)); y1 = max(0.0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W)); y2 = max(y1 + 1, min(y2, H))
            dets.append((c, [x1, y1, x2, y2]))
        raw[sid] = dets

    if args.mode == "val":
        if args.scan:
            print("=== 集成 val 阈值扫描 (官方协议) ===")
            best = None
            for c in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
                image_dets = {s: [(cc, b) for cc, b in raw[s] if cc >= c] for s in sids}
                all_d = []
                for s in sids:
                    for cc, b in image_dets[s]:
                        all_d.append((cc, s, b))
                all_d.sort(key=lambda x: -x[0])
                matched = {}; tp = fp = 0
                for cc, s, b in all_d:
                    W, H = img_sizes[s]
                    px = (b[0] / W, b[1] / H, b[2] / W, b[3] / H)
                    gts = [xywh_to_xyxy_norm(x) for x in gt[s]]
                    if s not in matched:
                        matched[s] = set()
                    bi, bv = -1, 0.0
                    for i, gb in enumerate(gts):
                        if i in matched[s]:
                            continue
                        v = iou(px, gb)
                        if v > bv:
                            bv, bi = v, i
                    if bi >= 0 and bv >= 0.5:
                        matched[s].add(bi); tp += 1
                    else:
                        fp += 1
                fn = sum(len(gt[s]) - len(matched.get(s, set())) for s in sids)
                P = tp / (tp + fp) if tp + fp else 0
                R = tp / (tp + fn) if tp + fn else 0
                m = compute_map_11pt_v2(gt, img_sizes, image_dets, 0.5)
                sc = 100 * (0.5 * m + 0.3 * P + 0.2 * R)
                print(f"conf={c:.2f} mAP={m:.4f} P={P:.4f} R={R:.4f} 总分={sc:.2f}")
                if best is None or sc > best[1]:
                    best = (c, sc)
            print(f">>> 集成最优 conf={best[0]:.2f}, 总分={best[1]:.2f}")
        return

    out = []
    for sid in sids:
        W, H = Image.open(os.path.join(img_root, sid + ".jpg")).size
        dets = []
        for cc, b in raw[sid]:
            if cc < args.conf:
                continue
            x1, y1, x2, y2 = b
            area = (x2 - x1) * (y2 - y1) / (W * H)
            if area < args.min_area:   # 过滤黑边/背景小块误检
                continue
            dets.append({"bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                         "confidence": round(cc, 4), "class": "phone"})
        out.append({"image_id": sid + ".jpg", "detections": dets})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    wd = sum(1 for x in out if x["detections"])
    print(f"集成提交已生成 -> {args.out} (conf={args.conf}, 有检测 {wd}/{len(out)})")


if __name__ == "__main__":
    main()
