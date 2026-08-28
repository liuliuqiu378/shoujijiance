"""用多个语义提示词测试 LocateAnything 对「打电话/手持/画报展示台」的区分能力。
每张图跑 4 个 prompt，记录每个 prompt 检出的 box 与 ref 文本。
"""
import os, re, json, time, argparse
from PIL import Image
import torch

MP = "/home/hmn-cjy/liuliuqiu/modelscope-m/LocateAnything"
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

PROMPTS = {
    "generic": "Locate all the instances that matches the following description: phone.",
    "calling": "Locate phones that a person is currently holding to their ear to make a call.",
    "holding": "Locate phones that are being held in someone's hand.",
    "poster":  "Locate phones that appear in a poster, advertisement, or on a display stand.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-list", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
    model = AutoModel.from_pretrained(MP, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    img_list = open(args.img_list).read().strip().splitlines()
    results = []
    for ip in img_list:
        sid = os.path.basename(ip)
        img = Image.open(ip).convert("RGB")
        w, h = img.size
        per_prompt = {}
        for name, question in PROMPTS.items():
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
                    max_new_tokens=2048, use_cache=True, generation_mode="hybrid",
                    do_sample=False, temperature=0.7, top_p=0.9, repetition_penalty=1.1, verbose=False,
                )
            ans = out[0] if isinstance(out, tuple) else out
            dt = time.time() - t0
            boxes = []
            for m in BOX_RE.finditer(ans):
                x1, y1, x2, y2 = [int(g) for g in m.groups()]
                boxes.append([x1/1000*w, y1/1000*h, x2/1000*w, y2/1000*h])
            per_prompt[name] = {"answer": ans, "time": dt, "n_boxes": len(boxes), "boxes": boxes}
        results.append({"image_id": sid, "prompts": per_prompt})
        print(f"done {sid}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
