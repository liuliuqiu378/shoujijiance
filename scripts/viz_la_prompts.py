"""把 eval_locateanything_prompts.py 的结果画成 1×4 对比图。"""
import os, json, math
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"

def font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except Exception:
        return ImageFont.load_default()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/tmp/la_prompt_results.json")
    ap.add_argument("--out", default="/tmp/la_prompt_viz")
    args = ap.parse_args()

    data = json.load(open(args.results))
    os.makedirs(args.out, exist_ok=True)
    f = font(22)
    for it in data:
        sid = it["image_id"]
        img_path = os.path.join("/tmp/la_prompt_imgs", sid)
        base = os.path.basename(sid)[:-4]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        # 2×2 网格
        cell_w, cell_h = w, h
        grid = Image.new("RGB", (cell_w*2, cell_h*2))
        positions = [(0,0),(cell_w,0),(0,cell_h),(cell_w,cell_h)]
        prompts = [("generic","通用 phone"), ("calling","打电话"), ("holding","手持"), ("poster","画报/展示台")]
        for (xoff,yoff), (key,title) in zip(positions, prompts):
            cell = img.copy()
            d = ImageDraw.Draw(cell)
            res = it["prompts"][key]
            for b in res["boxes"]:
                x1,y1,x2,y2 = b
                # 防止反向
                x1,x2 = min(x1,x2), max(x1,x2)
                y1,y2 = min(y1,y2), max(y1,y2)
                d.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=4)
            d.text((8,8), f"{title} ({res['n_boxes']}框)", fill=(255,255,0), font=f)
            grid.paste(cell, (xoff,yoff))
        out_path = os.path.join(args.out, f"la_prompt_{base}.png")
        grid.save(out_path)
        print("saved", out_path)

if __name__ == "__main__":
    main()
