"""从 YOLO 验证集预测结果 + GT 生成效果/错例 demo 图。
- 正确检测(TP): 绿框=预测命中, 橙虚线=GT
- 误检(FP): 红框=预测, 橙虚线=GT(无)
- 漏检(FN): 橙虚线=GT, 红框=无
- 多目标: 一图多手机

用法:
  python make_demos.py --pred yolo_val_pred.json --gt-root .../labels/val \
      --img-root .../images/val --out assets --per-type 6
"""
import os
import json
import glob
import argparse
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"


def load_font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except Exception:
        return ImageFont.load_default()


def load_gt(label_root):
    gt = {}
    for lf in glob.glob(os.path.join(label_root, "*.txt")):
        sid = os.path.basename(lf)[:-4]
        boxes = []
        if os.path.getsize(lf) > 0:
            for l in open(lf):
                p = l.strip().split()
                if len(p) == 5:
                    _, x, y, w, h = map(float, p)
                    boxes.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        gt[sid] = boxes
    return gt


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-type", type=int, default=6)
    args = ap.parse_args()

    gt = load_gt(args.gt_root)
    preds = json.load(open(args.pred))
    fp_pred = {it["image_id"][:-4]: [(d["bbox"], d.get("confidence", 0.5))
                                     for d in it.get("detections", [])] for it in preds}

    correct, fp, fn, multi = [], [], [], []
    for sid, gts in gt.items():
        preds_s = fp_pred.get(sid, [])
        matched_gt = set()
        matched_pred = set()
        for pi, (pb, cf) in enumerate(preds_s):
            x1, y1, x2, y2 = pb
            pn = (x1/1, y1/1, x2/1, y2/1)  # 像素级与归一化gt比较需同空间，这里用归一化
            best, bi = 0, -1
            for gi, gb in enumerate(gts):
                if gi in matched_gt:
                    continue
                # pb 是像素，gb 是归一化 -> 需图片尺寸
                pass
        # 重新用尺寸正确匹配
        ip = os.path.join(args.img_root, sid + ".jpg")
        if not os.path.exists(ip):
            continue
        W, H = Image.open(ip).size
        # pred 像素 -> 归一化
        preds_n = [((b[0]/W, b[1]/H, b[2]/W, b[3]/H), cf) for b, cf in preds_s]
        mg, mp = set(), set()
        for pi, (pn, cf) in enumerate(preds_n):
            best, bi = 0, -1
            for gi, gb in enumerate(gts):
                if gi in mg:
                    continue
                v = iou(pn, gb)
                if v > best:
                    best, bi = v, gi
            if bi >= 0 and best >= 0.5:
                mg.add(bi); mp.add(pi)
        n_pred, n_gt = len(preds_s), len(gts)
        if n_gt >= 2:
            multi.append(sid)
        if n_pred > 0 and len(mp) > 0 and len(mg) == n_gt:
            correct.append(sid)
        if n_pred > len(mp):
            fp.append(sid)  # 有未匹配预测 = 误检
        if len(mg) < n_gt:
            fn.append(sid)  # 有未匹配GT = 漏检

    def draw(sid, kind):
        ip = os.path.join(args.img_root, sid + ".jpg")
        img = Image.open(ip).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)
        fnt = load_font(22)
        gts = gt.get(sid, [])
        preds_s = fp_pred.get(sid, [])
        # GT 橙虚线
        for (x1, y1, x2, y2) in gts:
            d.rectangle([x1*W, y1*H, x2*W, y2*H], outline=(255, 165, 0), width=3)
        # pred 绿(命中)/红(FP)
        preds_n = [((b[0]/W, b[1]/H, b[2]/W, b[3]/H), cf) for b, cf in preds_s]
        mg = set()
        for pi, (pn, cf) in enumerate(preds_n):
            best, bi = 0, -1
            for gi, gb in enumerate(gts):
                if gi in mg:
                    continue
                v = iou(pn, gb)
                if v > best:
                    best, bi = v, gi
            if bi >= 0 and best >= 0.5:
                mg.add(bi)
                col = (0, 200, 0)
            else:
                col = (220, 0, 0)
            d.rectangle([pn[0]*W, pn[1]*H, pn[2]*W, pn[3]*H], outline=col, width=3)
        tag = {"correct": "正确检出 TP", "fp": "误检 FP", "fn": "漏检 FN", "multi": "多目标"}[kind]
        d.text((8, 8), tag, fill=(255, 255, 0), font=fnt)
        return img

    os.makedirs(args.out, exist_ok=True)
    picks = {"correct": correct, "fp": fp, "fn": fn, "multi": multi}
    for kind, lst in picks.items():
        # 优先选框较大的（更清晰）
        lst_sorted = sorted(lst, key=lambda s: -max([(g[2]-g[0])*(g[3]-g[1]) for g in gt.get(s, [])] or [0]))
        for i, sid in enumerate(lst_sorted[:args.per_type]):
            img = draw(sid, kind)
            outp = os.path.join(args.out, f"demo_{kind}_{sid[:8]}.png")
            img.save(outp)
            print("saved", outp)


if __name__ == "__main__":
    main()
