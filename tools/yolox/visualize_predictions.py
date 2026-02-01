#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict

import cv2


def _ensure_path(path, dataset_root):
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(dataset_root, path)
    return candidate


def _color_for_class(class_id):
    # Deterministic vivid-ish color per class id.
    r = (class_id * 37) % 255
    g = (class_id * 67) % 255
    b = (class_id * 97) % 255
    return (b, g, r)


def _draw_box(img, bbox, label, color, thickness, font_scale):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        text = label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = x1
        text_y = y1 - 4 if y1 - 4 - th > 0 else y1 + th + 4
        cv2.rectangle(
            img,
            (text_x, text_y - th - 2),
            (text_x + tw + 2, text_y + 2),
            color,
            -1,
        )
        cv2.putText(
            img,
            text,
            (text_x + 1, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser("Visualize COCO-format predictions on test images")
    parser.add_argument(
        "--dataset-root",
        default="datasets/bpmn_full",
        help="Root folder with images/ and annotations/",
    )
    parser.add_argument(
        "--ann",
        default="annotations/instances_test.json",
        help="COCO annotations json (relative to dataset root or absolute)",
    )
    parser.add_argument(
        "--pred",
        default="YOLOX/yolox_testdev_2017.json",
        help="COCO-format predictions json (list of {image_id, category_id, bbox, score})",
    )
    parser.add_argument(
        "--out",
        default="results/vis_test_pred",
        help="Output folder for images with drawn bboxes",
    )
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    parser.add_argument("--topk", type=int, default=200, help="Max predictions per image")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--box-thickness", type=int, default=2, help="BBox thickness")
    parser.add_argument("--font-scale", type=float, default=0.5, help="Label font scale")
    parser.add_argument("--no-labels", action="store_true", help="Disable labels")
    args = parser.parse_args()

    ann_path = _ensure_path(args.ann, args.dataset_root)
    pred_path = _ensure_path(args.pred, args.dataset_root)
    out_root = args.out

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotations not found: {ann_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in ann["images"]}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in ann["categories"]}

    preds_by_image = defaultdict(list)
    for p in preds:
        preds_by_image[p["image_id"]].append(p)

    os.makedirs(out_root, exist_ok=True)

    total_images = 0
    total_boxes = 0
    for image_id, rel_path in image_id_to_file.items():
        if args.limit and total_images >= args.limit:
            break
        image_path = rel_path
        if not os.path.isabs(image_path):
            image_path = os.path.join(args.dataset_root, rel_path)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        h, w = img.shape[:2]
        items = preds_by_image.get(image_id, [])
        items = [p for p in items if p.get("score", 0.0) >= args.score_thr]
        items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        if args.topk > 0:
            items = items[: args.topk]

        for p in items:
            x, y, bw, bh = p["bbox"]
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(w - 1, int(round(x + bw)))
            y2 = min(h - 1, int(round(y + bh)))
            cat_id = p.get("category_id", -1)
            name = cat_id_to_name.get(cat_id, str(cat_id))
            score = p.get("score", 0.0)
            label = "" if args.no_labels else f"{name} {score:.2f}"
            color = _color_for_class(cat_id if cat_id >= 0 else 0)
            _draw_box(img, (x1, y1, x2, y2), label, color, args.box_thickness, args.font_scale)
            total_boxes += 1

        out_path = os.path.join(out_root, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
        total_images += 1

    print(f"Saved {total_images} images with {total_boxes} boxes to: {out_root}")


if __name__ == "__main__":
    main()
