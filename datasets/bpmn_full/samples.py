# 20 оверлеев с покрытием классов
import random
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(r"e:\Projects\MegaSchool\SuccessfulBPMN\datasets\bpmn_full")
classes = [c.strip() for c in (root/"classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
num_classes = len(classes)

img_dir = root/"images"/"val" if (root/"images"/"val").exists() else root/"images"/"train"
lbl_dir = root/"labels"/img_dir.name

images = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}])

img_classes = []
for img in images:
    lbl = lbl_dir / f"{img.stem}.txt"
    if not lbl.exists():
        continue
    cls_set = set()
    for ln in lbl.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        if 0 <= cls < num_classes:
            cls_set.add(cls)
    if cls_set:
        img_classes.append((img, cls_set))

remaining = set(range(num_classes))
selected = []
img_classes.sort(key=lambda x: len(x[1]), reverse=True)
for img, cls_set in img_classes:
    if remaining & cls_set:
        selected.append((img, cls_set))
        remaining -= cls_set
    if not remaining:
        break

if len(selected) < 20:
    pool = [x for x in img_classes if x not in selected]
    random.seed(42); random.shuffle(pool)
    selected += pool[: max(0, 20 - len(selected))]

selected = selected[:20]
out_dir = root / "overlay_samples_allclasses"
out_dir.mkdir(parents=True, exist_ok=True)

palette = [
    (231, 76, 60), (46, 204, 113), (52, 152, 219), (155, 89, 182),
    (241, 196, 15), (230, 126, 34), (26, 188, 156), (127, 140, 141),
    (192, 57, 43), (39, 174, 96), (41, 128, 185), (142, 68, 173),
    (0, 0, 0), (255, 0, 255), (0, 128, 255), (128, 0, 255)
]

covered = set()
for img, _ in selected:
    lbl = lbl_dir / f"{img.stem}.txt"
    im = Image.open(img).convert("RGB")
    draw = ImageDraw.Draw(im)
    w_img, h_img = im.size

    for ln in lbl.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        x0 = (cx - w/2) * w_img
        y0 = (cy - h/2) * h_img
        x1 = (cx + w/2) * w_img
        y1 = (cy + h/2) * h_img
        color = palette[cls % len(palette)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), classes[cls], fill=color)
        covered.add(cls)

    im.save(out_dir / f"{img.stem}_overlay.png")

print("Saved overlays to:", out_dir)
print("Missing classes:", [classes[i] for i in sorted(set(range(num_classes)) - covered)])
