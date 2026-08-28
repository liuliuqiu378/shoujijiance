"""扫描不同 conf 下在 breview 验证集上的官方综合分，找提交最优阈值。
直接 import evaluate 的 load_gt/compute_map_11pt_v2，避免 subprocess 环境错。

用法:
  python scan_conf_breview.py --weights runs/v20_b11m_breview/weights/best_breview.pt --device 0
"""
import os
import json
import argparse
import glob
from collections import defaultdict
from ultralytics import YOLO

ROOT = "/home/hmn-cjy/liuliuqiu/fangzuobi/phone_detect"
BIMG = "/home/hmn-cjy/liuliuqiu/fangzuobi/data/Btest/test_b/images"
LAB = f"{ROOT}/dataset/labels/breview"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("conf_list", nargs="*", type=float,
                    default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    args = ap.parse_args()

    # import evaluate 函数
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate", f"{ROOT}/scripts/evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    gt = ev.load_gt(LAB)
    # 图片尺寸
    img_sizes = {}
    for sid in gt:
        p = os.path.join(BIMG, sid + ".jpg")
        from PIL import Image
        with Image.open(p) as im:
            img_sizes[sid] = im.size

    # 一次性推理 breview 验证集的图（与 gt 一致），得到全量低conf预测
    model = YOLO(args.weights)
    all_preds = {}  # sid -> [(conf, [x1,y1,x2,y2])]
    for sid in sorted(gt.keys()):
        name = sid + ".jpg"
        p = os.path.join(BIMG, name)
        if not os.path.exists(p):
            continue
        dets = []
        pr = model.predict(p, conf=0.001, iou=0.5,
                           imgsz=args.imgsz, device=args.device, verbose=False)
        for box in pr[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append((float(box.conf[0]), [x1, y1, x2, y2]))
        all_preds[sid] = dets

    print(f"权重: {args.weights}")
    print(f"{'conf':>6} {'mAP':>7} {'P':>7} {'R':>7} {'总分':>7} {'检出框':>7}")
    for conf in args.conf_list:
        # 按 conf 过滤
        image_dets = defaultdict(list)
        total_box = 0
        for sid, dets in all_preds.items():
            for c, b in dets:
                if c >= conf:
                    image_dets[sid].append((c, b))
                    total_box += 1
        # 算 P/R
        matched = defaultdict(set)
        tp = fp = 0
        all_dets = []
        for sid, dl in image_dets.items():
            for c, b in dl:
                all_dets.append((c, sid, b))
        all_dets.sort(key=lambda x: -x[0])
        gt_cache = {}
        for c, sid, b in all_dets:
            W, H = img_sizes[sid]
            px = (b[0]/W, b[1]/H, b[2]/W, b[3]/H)
            gts = gt_cache.get(sid)
            if gts is None:
                gts = [ev.xywh_to_xyxy_norm(g) for g in gt.get(sid, [])]
                gt_cache[sid] = gts
            best_i, best_iou = -1, 0.0
            for i, gb in enumerate(gts):
                if i in matched[sid]:
                    continue
                iouv = ev.iou(px, gb)
                if iouv > best_iou:
                    best_iou, best_i = iouv, i
            if best_i >= 0 and best_iou >= 0.5:
                matched[sid].add(best_i)
                tp += 1
            else:
                fp += 1
        fn = 0
        for sid, gts in gt.items():
            fn += len(gts) - len(matched.get(sid, set()))
        P = tp/(tp+fp) if (tp+fp) > 0 else 0
        R = tp/(tp+fn) if (tp+fn) > 0 else 0
        mAP = ev.compute_map_11pt_v2(gt, img_sizes, image_dets, iou_thr=0.5)
        score = 100*(0.5*mAP + 0.3*P + 0.2*R)
        print(f"{conf:>6.2f} {mAP:>7.4f} {P:>7.4f} {R:>7.4f} {score:>7.2f} {total_box:>7}")


if __name__ == "__main__":
    main()
