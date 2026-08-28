"""用 LocateAnything (NVIDIA 3B VLM) 在考场手机验证集上做零样本开放词汇检测，
并复用官方评测协议 (evaluate.py) 与 YOLO11m 公平对比。

关键说明：
- LocateAnything 是生成式 VLM，输出文本 token <box>x1,y1,x2,y2</box>，无显式置信度。
- 为匹配官方协议（conf<0.25 忽略），统一给预测框赋 conf=0.5（高于阈值，全部参与评测）。
- prompt 用官方 detect 模板："Locate all the instances that matches the following description: phone."

输出 JSON 格式同提交样例，可直接喂给 evaluate.py。
"""
import os
import re
import sys
import json
import time
import glob
import argparse

import torch
from PIL import Image

MP = "/home/hmn-cjy/liuliuqiu/modelscope-m/LocateAnything"
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0=全量")
    ap.add_argument("--conf", type=float, default=0.5, help="赋给LA预测框的置信度")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    print("loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MP, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    print("model loaded. mem=%.1fGB" % (torch.cuda.memory_allocated() / 1e9), flush=True)

    imgs = sorted(glob.glob(os.path.join(args.img_root, "*.jpg")))
    if args.limit:
        imgs = imgs[: args.limit]

    question = "Locate all the instances that matches the following description: phone."
    results = []
    t_total = 0.0
    n_boxes = 0
    for i, ip in enumerate(imgs):
        sid = os.path.basename(ip)
        img = Image.open(ip).convert("RGB")
        w, h = img.size
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question},
        ]}]
        text = proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = proc.process_vision_info(messages)
        inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        pv = inputs["pixel_values"].to(torch.bfloat16)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                pixel_values=pv, input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"), tokenizer=tok,
                max_new_tokens=args.max_new_tokens, use_cache=True,
                generation_mode="hybrid", do_sample=False, temperature=0.7,
                top_p=0.9, repetition_penalty=1.1, verbose=False,
            )
        ans = out[0] if isinstance(out, tuple) else out
        t_total += time.time() - t0
        dets = []
        for m in BOX_RE.finditer(ans):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            # 归一化 [0,1000] -> 像素
            bx1 = x1 / 1000 * w
            by1 = y1 / 1000 * h
            bx2 = x2 / 1000 * w
            by2 = y2 / 1000 * h
            # 防止越界/反向
            bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
            by1, by2 = min(by1, by2), max(by1, by2)
            if (bx2 - bx1) < 2 or (by2 - by1) < 2:
                continue
            dets.append({"bbox": [bx1, by1, bx2, by2], "confidence": args.conf, "class": "phone"})
        n_boxes += len(dets)
        results.append({"image_id": sid, "detections": dets})
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(imgs)}] avg={t_total/(i+1):.3f}s/img boxes={n_boxes}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"DONE: {len(results)} imgs, {n_boxes} boxes, avg {t_total/len(imgs):.3f}s/img -> {args.out}")


if __name__ == "__main__":
    main()
