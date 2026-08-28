"""B榜检测结果人工核查网页服务。

展示测试集上的模型预测框，逐图让用户确认每个框是否正确（没问题 / 实际没有），
并填写漏检数量，结果保存到 JSON 供后续分析。

用法:
  python review_app.py --submit ../submit_B_5yolo_wbf_c0.25.json \
      --img_root ../../data/Btest/test_b/images --port 8899
"""
import os
import json
import argparse
from flask import Flask, request, jsonify, send_file, Response

app = Flask(__name__)

# 全局状态
IMG_ROOT = ""
SUBMIT = []          # 提交数据列表
IMG_ORDER = []       # 展示顺序（image_id 列表）
RESULTS = {}         # image_id -> {boxes:[{idx,status}], missed:int, note:str}
RESULT_PATH = ""


def build_order(submit):
    """构造展示顺序：优先有检测框的图，按可疑度排序（多框、低置信度优先）。"""
    items = []
    for entry in submit:
        sid = entry["image_id"]
        dets = entry.get("detections", [])
        if not dets:
            items.append((sid, 0, 0.0, 0))  # 空图排最后
            continue
        n = len(dets)
        minc = min(d["confidence"] for d in dets)
        # 可疑度：多框 + 低置信度 -> 越大越可疑
        susp = n * 1.0 + (1.0 - minc) * 2.0
        items.append((sid, n, minc, susp))
    # 非空图按可疑度降序，空图置后（可疑度=0）
    nonempty = [x for x in items if x[1] > 0]
    empty = [x for x in items if x[1] == 0]
    nonempty.sort(key=lambda x: -x[3])
    order = [x[0] for x in nonempty] + [x[0] for x in empty]
    return order


def load_results():
    global RESULTS
    if os.path.exists(RESULT_PATH):
        try:
            RESULTS = json.load(open(RESULT_PATH, encoding="utf-8"))
        except Exception:
            RESULTS = {}


def save_results():
    tmp = RESULT_PATH + ".tmp"
    json.dump(RESULTS, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    os.replace(tmp, RESULT_PATH)


@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(__file__), "review.html"))


@app.route("/api/meta")
def meta():
    total = len(IMG_ORDER)
    reviewed = sum(1 for v in RESULTS.values() if v.get("done"))
    return jsonify({
        "total": total,
        "reviewed": reviewed,
        "empty_total": sum(1 for s in IMG_ORDER if _is_empty(s)),
        "have_box_total": sum(1 for s in IMG_ORDER if not _is_empty(s)),
    })


def _is_empty(sid):
    for e in SUBMIT:
        if e["image_id"] == sid:
            return len(e.get("detections", [])) == 0
    return True


@app.route("/api/image")
def get_image():
    idx = int(request.args.get("idx", 0))
    if idx < 0 or idx >= len(IMG_ORDER):
        return jsonify({"error": "out of range"}), 400
    sid = IMG_ORDER[idx]
    path = os.path.join(IMG_ROOT, sid)
    if not os.path.exists(path):
        return jsonify({"error": "no image"}), 404
    return send_file(path)


@app.route("/api/item")
def get_item():
    idx = int(request.args.get("idx", 0))
    if idx < 0 or idx >= len(IMG_ORDER):
        return jsonify({"error": "out of range"}), 400
    sid = IMG_ORDER[idx]
    entry = next((e for e in SUBMIT if e["image_id"] == sid), None)
    dets = entry.get("detections", []) if entry else []
    # 图片尺寸
    from PIL import Image
    w, h = Image.open(os.path.join(IMG_ROOT, sid)).size
    boxes = [{"bbox": d["bbox"], "confidence": d["confidence"]} for d in dets]
    saved = RESULTS.get(sid, {})
    return jsonify({
        "idx": idx,
        "total": len(IMG_ORDER),
        "image_id": sid,
        "width": w,
        "height": h,
        "boxes": boxes,
        "is_empty_pred": len(dets) == 0,
        "saved": saved,
    })


@app.route("/api/save", methods=["POST"])
def save():
    data = request.get_json()
    sid = data["image_id"]
    RESULTS[sid] = {
        "boxes": data.get("boxes", []),   # [{idx, status: ok|none}]
        "missed": data.get("missed", 0),
        "missed_boxes": data.get("missed_boxes", []),  # 漏标框 [[x1,y1,x2,y2],...]
        "note": data.get("note", ""),
        "is_empty_pred": data.get("is_empty_pred", False),
        "done": True,
    }
    save_results()
    return jsonify({"ok": True})


@app.route("/api/summary")
def summary():
    """统计已标注结果，给出问题分布。"""
    n_review = 0
    fp_boxes = 0           # 标注为"实际没有"的框
    total_pred_boxes = 0
    missed_total = 0
    empty_pred_review = 0
    empty_pred_missed = 0  # 空预测但用户填了漏检
    for sid, v in RESULTS.items():
        if not v.get("done"):
            continue
        n_review += 1
        if v.get("is_empty_pred"):
            empty_pred_review += 1
            if v.get("missed", 0) > 0:
                empty_pred_missed += 1
                missed_total += v["missed"]
        else:
            for b in v.get("boxes", []):
                total_pred_boxes += 1
                if b.get("status") == "none":
                    fp_boxes += 1
            missed_total += v.get("missed", 0)
    return jsonify({
        "reviewed_images": n_review,
        "pred_boxes_reviewed": total_pred_boxes,
        "fp_boxes_labeled": fp_boxes,
        "missed_total": missed_total,
        "empty_pred_images": empty_pred_review,
        "empty_pred_with_missed": empty_pred_missed,
        "fp_rate_in_reviewed": round(fp_boxes / total_pred_boxes, 4) if total_pred_boxes else None,
    })


def main():
    global IMG_ROOT, SUBMIT, IMG_ORDER, RESULT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    IMG_ROOT = args.img_root
    SUBMIT = json.load(open(args.submit, encoding="utf-8"))
    RESULT_PATH = os.path.join(os.path.dirname(args.submit), "review_results.json")
    IMG_ORDER[:] = build_order(SUBMIT)
    load_results()
    print(f"载入 {len(SUBMIT)} 张图, 展示顺序 {len(IMG_ORDER)} 张")
    print(f"标注结果将保存到 {RESULT_PATH}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
