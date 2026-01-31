import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_classes(root: Path) -> List[str]:
    classes_path = root / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_path}")
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_images(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def _read_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    rows: List[Tuple[int, float, float, float, float]] = []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            continue
        rows.append((cls, x, y, w, h))
    return rows


def _yolo_to_coco_bbox(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    x0 = (x - w / 2.0) * img_w
    y0 = (y - h / 2.0) * img_h
    bw = w * img_w
    bh = h * img_h

    x0 = max(0.0, min(x0, img_w - 1.0))
    y0 = max(0.0, min(y0, img_h - 1.0))
    x1 = min(float(img_w), x0 + bw)
    y1 = min(float(img_h), y0 + bh)
    bw = x1 - x0
    bh = y1 - y0
    return x0, y0, bw, bh


def convert_split(root: Path, split: str, classes: List[str], out_dir: Path) -> Path:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    for img_path in _iter_images(images_dir):
        with Image.open(img_path) as im:
            w, h = im.size

        file_name = str(img_path.relative_to(root)).replace("\\", "/")
        images.append({"id": img_id, "file_name": file_name, "width": w, "height": h})

        label_path = labels_dir / f"{img_path.stem}.txt"
        for cls, x, y, bw, bh in _read_labels(label_path):
            if cls < 0 or cls >= len(classes):
                continue
            x0, y0, bw, bh = _yolo_to_coco_bbox(x, y, bw, bh, w, h)
            if bw <= 0 or bh <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,
                    "bbox": [x0, y0, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        img_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]

    coco = {
        "info": {"description": f"bpmn_full {split}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"instances_{split}.json"
    out_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    classes = _load_classes(root)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "annotations")

    outputs: Dict[str, str] = {}
    for split in args.splits:
        out_path = convert_split(root, split, classes, out_dir)
        outputs[split] = str(out_path)

    print(json.dumps({"status": "ok", "outputs": outputs}, ensure_ascii=True))


if __name__ == "__main__":
    main()
