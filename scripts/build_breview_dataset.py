"""从人工核查结果构造 B榜 1955 张的训练/验证数据。

输入:
  phone_detect/review_results.json  (人工标注: boxes状态, missed_boxes像素框, missed数量)
  phone_detect/submit_B_5yolo_wbf_c0.25.json  (模型预测框, 像素坐标)
  data/Btest/test_b/images/  (B榜原图)

规则(用户确认):
  - 预测框里 "没问题"(ok) -> 作为正样本GT保留
  - 预测框里 "实际没有"(none) -> 删除(不写)
  - 漏标框 missed_boxes(像素) -> 新增为正样本GT
  - 空预测图(is_empty_pred)且 missed>0 -> 仅用 missed_boxes 作为GT(无预测可参考)
  - 空预测图且 missed==0 -> 空标签(负样本)
  - 有预测图但用户没点任何框(全默认ok) -> 所有预测框作GT

输出:
  phone_detect/dataset/images/breview/<sid>.jpg  (软链到原图)
  phone_detect/dataset/labels/breview/<sid>.txt  (YOLO格式 class_id cx cy w h, 归一化)
  phone_detect/dataset_breview.yaml  (train=原train+breview, val=breview)
"""
import os
import json
import glob
import argparse

ROOT = "/home/hmn-cjy/liuliuqiu/fangzuobi/phone_detect"
BIMG = "/home/hmn-cjy/liuliuqiu/fangzuobi/data/Btest/test_b/images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default=f"{ROOT}/review_results.json")
    ap.add_argument("--submit", default=f"{ROOT}/submit_B_5yolo_wbf_c0.25.json")
    ap.add_argument("--out_dir", default=f"{ROOT}/dataset")
    args = ap.parse_args()

    review = json.load(open(args.review, encoding="utf-8"))
    submit = json.load(open(args.submit, encoding="utf-8"))
    # 提交框像素坐标
    sub_map = {e["image_id"]: e.get("detections", []) for e in submit}

    img_out = os.path.join(args.out_dir, "images", "breview")
    lab_out = os.path.join(args.out_dir, "labels", "breview")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)

    done = {k: v for k, v in review.items() if v.get("done")}
    stats = {"pos": 0, "neg": 0, "boxes": 0, "skipped_noinfo": 0}

    for sid_jpg, v in done.items():
        sid = sid_jpg[:-4] if sid_jpg.endswith(".jpg") else sid_jpg
        # 软链图片
        src = os.path.join(BIMG, sid_jpg)
        dst = os.path.join(img_out, sid_jpg)
        if not os.path.exists(src):
            stats["skipped_noinfo"] += 1
            continue
        if not os.path.islink(dst):
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        lines = []
        W = H = None
        # 用提交图尺寸(或读图)
        try:
            from PIL import Image
            with Image.open(src) as im:
                W, H = im.size
        except Exception:
            W = H = None

        dets = sub_map.get(sid_jpg, [])
        if W is None and dets:
            # 退而用预测框估算? 不可靠, 跳过
            pass

        # 1) 预测框: ok 保留, none 删除
        boxes_status = v.get("boxes", [])
        for i, b in enumerate(boxes_status):
            if b.get("status") == "none":
                continue
            if i < len(dets):
                x1, y1, x2, y2 = dets[i]["bbox"]
                if W:
                    lines.append(px2yolo(x1, y1, x2, y2, W, H))
        # 2) 漏标框(像素) 新增
        for mb in v.get("missed_boxes", []):
            x1, y1, x2, y2 = mb
            if W:
                lines.append(px2yolo(x1, y1, x2, y2, W, H))
            else:
                # 漏标框无图尺寸则跳过(极少)
                pass

        # 写标签
        with open(os.path.join(lab_out, sid + ".txt"), "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
                stats["pos"] += 1
                stats["boxes"] += len(lines)
            else:
                # 空标签(负样本) 或 仅漏检但无尺寸
                stats["neg"] += 1

    print("B榜核查数据集构建完成:", json.dumps(stats, ensure_ascii=False))
    print(f"  正样本图 {stats['pos']} 负样本图 {stats['neg']} 框 {stats['boxes']}")


def px2yolo(x1, y1, x2, y2, W, H):
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    w = min(1.0, max(0.0, w))
    h = min(1.0, max(0.0, h))
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


if __name__ == "__main__":
    main()
