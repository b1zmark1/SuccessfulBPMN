from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch

from preprocess import PreprocessConfig, preprocess
from detect import DetectionConfig, detect_text_boxes, draw_text_boxes


def _add_yolox_to_path(repo_root: str) -> None:
    yolox_dir = os.path.join(repo_root, "YOLOX")
    if os.path.isdir(yolox_dir) and yolox_dir not in sys.path:
        sys.path.insert(0, yolox_dir)


def _get_images(path: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[str] = []
    for base, _, files in os.walk(path):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                out.append(os.path.join(base, name))
    out.sort()
    return out


class YoloXPredictor:
    def __init__(self, model, exp, cls_names, device, fp16, legacy):
        from yolox.data.data_augment import ValTransform
        from yolox.utils import postprocess

        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.postprocess = postprocess

    def inference(self, bgr):
        if bgr is None:
            return None, None
        h, w = bgr.shape[:2]
        ratio = min(self.test_size[0] / h, self.test_size[1] / w)
        img_info = {"height": h, "width": w, "ratio": ratio, "raw_img": bgr}

        img, _ = self.preproc(bgr, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            outputs = self.model(img)
            outputs = self.postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def draw(self, output, img_info):
        from yolox.utils import vis

        img = img_info["raw_img"].copy()
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= img_info["ratio"]
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        return vis(img, bboxes, scores, cls, self.confthre, self.cls_names)

    def to_detections(self, output, img_info):
        if output is None:
            return []
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= img_info["ratio"]
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        dets = []
        for i in range(bboxes.shape[0]):
            class_idx = int(cls[i])
            class_name = (
                self.cls_names[class_idx]
                if self.cls_names is not None and class_idx < len(self.cls_names)
                else str(class_idx)
            )
            if class_name == "text":
                # text will come from EasyOCR
                continue
            x1, y1, x2, y2 = [float(x) for x in bboxes[i].numpy().tolist()]
            dets.append(
                {
                    "class_id": class_idx,
                    "class_name": class_name,
                    "score": float(scores[i].numpy().item()),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "source": "yolox",
                }
            )
        return dets


def _load_class_names(dataset_root: str):
    classes_path = os.path.join(dataset_root, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names
    try:
        from yolox.data.datasets import COCO_CLASSES
        return COCO_CLASSES
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Image file or folder")
    ap.add_argument("--out", required=True, help="Output dir for overlays/json")
    ap.add_argument("--exp-file", default="results/yolox_tiny_bpmn.py")
    ap.add_argument("--ckpt", default="results/best_ckpt.pth")
    ap.add_argument("--dataset-root", default="datasets/bpmn_full")
    ap.add_argument("--device", default=None, choices=["cpu", "gpu"])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--nms", type=float, default=None)
    ap.add_argument("--tsize", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-text", action="store_true", help="Disable text detector")
    ap.add_argument("--lang", type=str, default="ru")
    ap.add_argument("--gpu-text", action="store_true")
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _add_yolox_to_path(repo_root)

    from yolox.exp import get_exp
    from yolox.utils import fuse_model

    if args.device is None:
        args.device = "gpu" if torch.cuda.is_available() else "cpu"

    exp = get_exp(args.exp_file, None)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()
    # Detect Git LFS pointer (text) instead of real checkpoint
    with open(args.ckpt, "rb") as f:
        head = f.read(200)
    if b"git-lfs.github.com/spec/v1" in head or head.startswith(b"version "):
        raise SystemExit(
            "Checkpoint looks like a Git LFS pointer, not a real .pth file. "
            "Please download the actual weights (git lfs pull) or replace the file."
        )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)

    cls_names = _load_class_names(args.dataset_root)
    predictor = YoloXPredictor(model, exp, cls_names, args.device, args.fp16, args.legacy)

    images_path = Path(args.images)
    if images_path.is_dir():
        image_list = _get_images(str(images_path))
    else:
        image_list = [str(images_path)]
    if args.limit and args.limit > 0:
        image_list = image_list[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    download_enabled = not bool(args.no_download)
    d_cfg = DetectionConfig(
        lang=args.lang,
        gpu=bool(args.gpu_text),
        download_enabled=bool(download_enabled),
        model_storage_directory=args.model_dir,
    )

    pre_cfg = PreprocessConfig()
    t0 = time.time()
    for img_path in image_list:
        pre = preprocess(img_path, pre_cfg)
        model_bgr = pre["model_bgr"]

        outputs, img_info = predictor.inference(model_bgr)
        yolox_output = outputs[0] if outputs else None
        yolox_overlay = predictor.draw(yolox_output, img_info)
        yolox_dets = predictor.to_detections(yolox_output, img_info)

        text_det = None
        text_overlay = None
        ensemble_overlay = None
        if not args.no_text:
            text_det = detect_text_boxes(model_bgr.copy(), d_cfg)
            text_overlay = draw_text_boxes(model_bgr.copy(), text_det)

            # overlay text on top of YOLOX overlay for a combined view
            ensemble_overlay = draw_text_boxes(yolox_overlay.copy(), text_det)

        rel = os.path.relpath(img_path, str(images_path)) if images_path.is_dir() else os.path.basename(img_path)
        out_img_dir = out_dir / os.path.dirname(rel)
        out_img_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_img_dir / (Path(rel).stem + "_yolox.png")), yolox_overlay)
        if text_overlay is not None:
            cv2.imwrite(str(out_img_dir / (Path(rel).stem + "_text.png")), text_overlay)
        if ensemble_overlay is not None:
            cv2.imwrite(str(out_img_dir / (Path(rel).stem + "_ensemble.png")), ensemble_overlay)

        text_dets = []
        if text_det is not None:
            for b in text_det.get("boxes", []):
                bbox = b.get("bbox", None)
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(x) for x in bbox]
                text_dets.append(
                    {
                        "class_id": -1,
                        "class_name": "text",
                        "score": float(b.get("score", 1.0)) if isinstance(b.get("score", 1.0), (int, float)) else 1.0,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_xywh": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "source": "easyocr",
                    }
                )

        result = {
            "image": {
                "file_name": rel,
                "width": int(model_bgr.shape[1]),
                "height": int(model_bgr.shape[0]),
                "resize_ratio": float(pre["resize_ratio"]),
                "geom_angle_deg": float(pre["geom_angle_deg"]),
            },
            "meta": {
                "yolox_conf": float(exp.test_conf),
                "yolox_nms": float(exp.nmsthre),
                "tsize": list(exp.test_size),
            },
            "detections": yolox_dets + text_dets,
        }
        with open(out_img_dir / (Path(rel).stem + "_ensemble.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    dt = time.time() - t0
    print(f"Saved {len(image_list)} images to {out_dir} in {dt:.1f}s")


if __name__ == "__main__":
    main()
