"""在 breview 验证集上挑综合分最高的 epoch 权重。

对指定 run 目录下所有 epoch 权重，推理 breview 图 -> pred.json -> evaluate.py 算
官方综合分(0.5map+0.3P+0.2R)，选最高者复制为 best_breview.pt。

用法:
  python select_best_breview.py --run runs/v20_b11m_breview --device 0
"""
import os
import json
import argparse
import glob
from ultralytics import YOLO

ROOT = "/home/hmn-cjy/liuliuqiu/fangzuobi/phone_detect"
BIMG = "/home/hmn-cjy/liuliuqiu/fangzuobi/data/Btest/test_b/images"
LAB = f"{ROOT}/dataset/labels/breview"


def infer_run(run_dir, device):
    """对 run 下每个 epoch 权重推理 breview，返回 {wpath: pred_json}。"""
    wfiles = sorted(glob.glob(os.path.join(run_dir, "weights", "epoch*.pt")))
    if not wfiles:
        wfiles = sorted(glob.glob(os.path.join(run_dir, "weights", "*.pt")))
    results = {}
    for w in wfiles:
        model = YOLO(w)
        preds = []
        for name in sorted(os.listdir(BIMG)):
            if not name.endswith(".jpg"):
                continue
            p = os.path.join(BIMG, name)
            dets = []
            try:
                pr = model.predict(p, conf=0.001, iou=0.5, imgsz=1280,
                                   device=device, verbose=False)
                for box in pr[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append({"bbox": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)],
                                 "confidence": round(float(box.conf[0]), 4),
                                 "class": "phone"})
            except Exception:
                pass
            preds.append({"image_id": name, "detections": dets})
        results[w] = preds
    return results


def eval_pred(preds):
    """复刻 evaluate.py 核心，返回 (score, mAP, P, R)。"""
    from collections import defaultdict
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate", f"{ROOT}/scripts/evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    gt = ev.load_gt(LAB)
    # 转 preds 为 evaluate 需要的格式
    pred_map = {p["image_id"][:-4]: p["detections"] for p in preds}
    # 直接调用 evaluate 的逻辑
    import subprocess
    tmp = "/tmp/_breview_pred.json"
    json.dump(preds, open(tmp, "w"))
    out = subprocess.run(
        ["python3", f"{ROOT}/scripts/evaluate.py", "--pred", tmp,
         "--gt-root", LAB, "--img-root", BIMG, "--conf", "0.25"],
        capture_output=True, text=True)
    score = m = pr = rc = 0.0
    for line in out.stdout.strip().splitlines():
        if line.startswith("mAP@0.5"):
            m = float(line.split("=")[1].strip())
        elif line.startswith("Precision"):
            pr = float(line.split("=")[1].strip())
        elif line.startswith("Recall"):
            rc = float(line.split("=")[1].strip())
        elif line.startswith("总分"):
            score = float(line.split("=")[1].strip())
    return score, m, pr, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    print(f"推理 {args.run} 下所有 epoch 权重 ...")
    preds = infer_run(args.run, args.device)
    best = None
    for w, pj in preds.items():
        score, m, pr, rc = eval_pred(pj)
        print(f"  {os.path.basename(w)}: score={score:.4f} mAP={m:.4f} P={pr:.4f} R={rc:.4f}")
        if best is None or score > best[1]:
            best = (w, score, m, pr, rc)
    if best:
        dst = os.path.join(args.run, "weights", "best_breview.pt")
        import shutil
        shutil.copy(best[0], dst)
        print(f"最佳: {os.path.basename(best[0])} score={best[1]:.4f} -> 已复制为 {dst}")


if __name__ == "__main__":
    main()
