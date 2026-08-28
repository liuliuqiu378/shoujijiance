"""本地评估脚本 —— 复刻官方评测协议。

协议要点（见 data/说明.md）：
  - IoU 阈值固定 0.5
  - 推理置信度 < 0.25 的预测框直接忽略（不参与任何指标）
  - 重复框处理：按置信度降序；每个 GT 只匹配第一个 IoU>=0.5 的预测为 TP，其余为 FP
  - 负样本(空 GT)若提交非空检测，每个预测框计 FP
  - mAP@0.5 采用 VOC2012 11 点插值法（单类，等价于 AP@0.5）
  - 总分 = 100*(0.5*mAP + 0.3*Precision + 0.2*Recall)

用法:
  python evaluate.py --pred pred.json --gt-root dataset/labels/val --img-root dataset/images/val
pred.json 格式同提交样例：[{"image_id": "x.jpg", "detections":[{"bbox":[x1,y1,x2,y2],"confidence":f,"class":"phone"}]}]
"""
import os
import json
import argparse
import glob
from collections import defaultdict


def load_gt(label_root):
    gt = {}
    for lf in glob.glob(os.path.join(label_root, "*.txt")):
        sid = os.path.basename(lf)[:-4]
        boxes = []
        if os.path.getsize(lf) > 0:
            with open(lf) as f:
                for l in f:
                    p = l.strip().split()
                    if len(p) != 5:
                        continue
                    _, x, y, w, h = map(float, p)
                    boxes.append((x, y, w, h))
        gt[sid] = boxes
    return gt


def xywh_to_xyxy_norm(b):
    x, y, w, h = b
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    gt = load_gt(args.gt_root)
    # 图片尺寸（用于像素->归一化）
    img_sizes = {}
    for sid in gt:
        p = os.path.join(args.img_root, sid + ".jpg")
        from PIL import Image
        with Image.open(p) as im:
            img_sizes[sid] = im.size  # (W,H)

    with open(args.pred) as f:
        preds = json.load(f)

    # 收集所有 (image, conf, iou_max_with_unmatched) 用于 PR 计算单类
    # 按官方：对所有图，所有过滤后预测框按 conf 降序，依次匹配各自图的 GT
    image_dets = defaultdict(list)
    for item in preds:
        sid = item["image_id"][:-4] if item["image_id"].endswith(".jpg") else item["image_id"]
        for d in item.get("detections", []):
            if d.get("confidence", 0) < args.conf:
                continue
            image_dets[sid].append((d["confidence"], d["bbox"]))

    tp = fp = fn_total = 0
    # 用于 AP 计算：记录每图匹配情况
    # 按 conf 全局降序处理，维护每图 GT 是否已匹配
    matched = defaultdict(set)
    all_dets = []
    for sid, dets in image_dets.items():
        for conf, bbox in dets:
            all_dets.append((conf, sid, bbox))
    all_dets.sort(key=lambda x: -x[0])

    gt_boxes_cache = {}
    for conf, sid, bbox in all_dets:
        W, H = img_sizes.get(sid, (1, 1))
        # pred 像素 -> 归一化 xyxy
        x1, y1, x2, y2 = bbox
        px = (x1 / W, y1 / H, x2 / W, y2 / H)
        gts = gt_boxes_cache.get(sid)
        if gts is None:
            gts = [xywh_to_xyxy_norm(b) for b in gt.get(sid, [])]
            gt_boxes_cache[sid] = gts
        best_i, best_iou = -1, 0.0
        for i, gb in enumerate(gts):
            if i in matched[sid]:
                continue
            iouv = iou(px, gb)
            if iouv > best_iou:
                best_iou, best_i = iouv, i
        if best_i >= 0 and best_iou >= args.iou:
            matched[sid].add(best_i)
            tp += 1
        else:
            fp += 1

    # FN：每张图未匹配的 GT 数
    for sid, gts in gt.items():
        fn_total += len(gts) - len(matched.get(sid, set()))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0.0

    # 11-point mAP@0.5  (单类)
    mAP = compute_map_11pt(gt, img_sizes, gt_boxes_cache, image_dets, args)

    score = 100 * (0.5 * mAP + 0.3 * precision + 0.2 * recall)
    print(f"TP={tp} FP={fp} FN={fn_total}")
    print(f"mAP@0.5 = {mAP:.4f}")
    print(f"Precision= {precision:.4f}")
    print(f"Recall   = {recall:.4f}")
    print(f"总分     = {score:.2f}")


def compute_map_11pt(gt, img_sizes, gt_boxes_cache, image_dets, args):
    """VOC2012 11-point 插值 AP（单类，IoU=阈值）。"""
    rec_thr = args.iou
    # 全局排序预测
    all_dets = []
    for sid, dets in image_dets.items():
        for conf, bbox in dets:
            all_dets.append((conf, sid, bbox))
    all_dets.sort(key=lambda x: -x[0])
    total_gt = sum(len(v) for v in gt.values())
    if total_gt == 0:
        return 0.0
    matched = defaultdict(set)
    tp_curve = []
    fp_curve = []
    cur_tp = cur_fp = 0
    for conf, sid, bbox in all_dets:
        W, H = img_sizes.get(sid, (1, 1))
        x1, y1, x2, y2 = bbox
        px = (x1 / W, y1 / H, x2 / W, y2 / H)
        gts = gt_boxes_cache.get(sid)
        if gts is None:
            gts = [xywh_to_xyxy_norm(b) for b in gt.get(sid, [])]
            gt_boxes_cache[sid] = gts
        best_i, best_iou = -1, 0.0
        for i, gb in enumerate(gts):
            if i in matched[sid]:
                continue
            iouv = iou(px, gb)
            if iouv > best_iou:
                best_iou, best_i = iouv, i
        if best_i >= 0 and best_iou >= rec_thr:
            matched[sid].add(best_i)
            cur_tp += 1
        else:
            cur_fp += 1
        tp_curve.append(cur_tp)
        fp_curve.append(cur_fp)
    # 计算 PR 曲线
    prec = [t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp_curve, fp_curve)]
    rec = [t / total_gt for t in tp_curve]
    # 11-point
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        vals = [p for p, r in zip(prec, rec) if r >= t]
        ap += max(vals) if vals else 0.0
    ap /= 11.0
    return ap


def compute_map_11pt_v2(gt, img_sizes, image_dets, iou_thr=0.5):
    """可被外部 import 的 11-point AP（不依赖 args）。

    image_dets: {sid: [(conf, bbox), ...]}  已按 conf 阈值过滤后的预测
    """
    all_dets = []
    for sid, dets in image_dets.items():
        for conf, bbox in dets:
            all_dets.append((conf, sid, bbox))
    all_dets.sort(key=lambda x: -x[0])
    total_gt = sum(len(v) for v in gt.values())
    if total_gt == 0:
        return 0.0
    matched = defaultdict(set)
    gt_cache = {}
    tp_curve = []
    fp_curve = []
    cur_tp = cur_fp = 0
    for conf, sid, bbox in all_dets:
        W, H = img_sizes.get(sid, (1, 1))
        x1, y1, x2, y2 = bbox
        px = (x1 / W, y1 / H, x2 / W, y2 / H)
        gts = gt_cache.get(sid)
        if gts is None:
            gts = [xywh_to_xyxy_norm(b) for b in gt.get(sid, [])]
            gt_cache[sid] = gts
        if sid not in matched:
            matched[sid] = set()
        best_i, best_iou = -1, 0.0
        for i, gb in enumerate(gts):
            if i in matched[sid]:
                continue
            iouv = iou(px, gb)
            if iouv > best_iou:
                best_iou, best_i = iouv, i
        if best_i >= 0 and best_iou >= iou_thr:
            matched[sid].add(best_i)
            cur_tp += 1
        else:
            cur_fp += 1
        tp_curve.append(cur_tp)
        fp_curve.append(cur_fp)
    prec = [t / (t + f) if (t + f) > 0 else 0.0 for t, f in zip(tp_curve, fp_curve)]
    rec = [t / total_gt for t in tp_curve]
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        vals = [p for p, r in zip(prec, rec) if r >= t]
        ap += max(vals) if vals else 0.0
    ap /= 11.0
    return ap


if __name__ == "__main__":
    main()
