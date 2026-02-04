from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from preprocess import PreprocessConfig, preprocess
from detect import DetectionConfig, detect_text_boxes, draw_text_boxes


def _add_yolox_to_path(repo_root: str) -> None:
    yolox_dir = os.path.join(repo_root, "YOLOX")
    if os.path.isdir(yolox_dir) and yolox_dir not in sys.path:
        sys.path.insert(0, yolox_dir)


def _get_images(path: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    p = Path(path)
    if p.is_file():
        return [str(p.resolve())]
    out: list[str] = []
    for root, _, files in os.walk(path):
        for name in files:
            if Path(name).suffix.lower() in exts:
                out.append(str((Path(root) / name).resolve()))
    out.sort()
    return out


def _load_class_names(dataset_root: str):
    classes_path = os.path.join(dataset_root, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names
    return None


class OpenVinoBackend:
    def __init__(self, model_xml: str):
        from openvino.runtime import Core

        core = Core()
        model = core.read_model(model_xml)
        self.compiled = core.compile_model(model, "CPU")
        self.input_port = self.compiled.inputs[0]
        self.output_port = self.compiled.outputs[0]
        self.req = self.compiled.create_infer_request()

    def infer(self, inp: np.ndarray) -> np.ndarray:
        out_map = self.req.infer({self.input_port: inp})
        out = out_map[self.output_port]
        return np.asarray(out)


def _decode_and_postprocess(
    raw: np.ndarray,
    decoder,
    num_classes: int,
    conf: float,
    nms: float,
):
    from yolox.utils import postprocess

    pred = torch.from_numpy(raw)
    pred = decoder(pred, dtype=pred.type())
    outputs = postprocess(pred, num_classes, conf, nms, class_agnostic=True)
    return outputs[0] if outputs and len(outputs) > 0 else None


def _yolox_detections_from_output(
    output,
    ratio: float,
    class_names: list[str] | None,
) -> list[dict[str, Any]]:
    if output is None:
        return []
    output = output.cpu()
    bboxes = output[:, 0:4]
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    detections: list[dict[str, Any]] = []
    for i in range(bboxes.shape[0]):
        class_id = int(cls[i])
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        if class_name == "text":
            continue
        x1, y1, x2, y2 = [float(x) for x in bboxes[i].numpy().tolist()]
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "score": float(scores[i].numpy().item()),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "source": "yolox",
            }
        )
    return detections


def _text_detections(text_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not text_json:
        return []
    out: list[dict[str, Any]] = []
    for b in text_json.get("boxes", []):
        bbox = b.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        out.append(
            {
                "class_id": -1,
                "class_name": "text",
                "score": 1.0,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "source": "easyocr_detect",
            }
        )
    return out


def _draw_yolox_boxes(bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    out = bgr.copy()
    for d in detections:
        if d.get("source") != "yolox":
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in d["bbox_xyxy"]]
        cls_name = str(d.get("class_name", "obj"))
        score = float(d.get("score", 0.0))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            out,
            f"{cls_name}:{score:.1%}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser("OpenVINO-only ensemble infer")
    ap.add_argument("--images", required=True, help="Image file or folder")
    ap.add_argument("--out", required=True, help="Output dir for overlays/json")
    ap.add_argument("--exp-file", required=True)
    ap.add_argument("--openvino-xml", required=True)
    ap.add_argument("--dataset-root", default="datasets/bpmn_full")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--tsize", type=int, default=None)
    ap.add_argument("--lang", type=str, default="ru")
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    repo_root = str(Path(__file__).resolve().parents[1])
    _add_yolox_to_path(repo_root)

    from yolox.exp import get_exp
    from yolox.data.data_augment import ValTransform

    images = _get_images(args.images)
    if not images:
        raise SystemExit(f"No images found in: {args.images}")

    exp = get_exp(args.exp_file, None)
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    exp.test_conf = float(args.conf)
    exp.nmsthre = float(args.nms)

    model = exp.get_model()
    model.head.hw = [(exp.test_size[0] // s, exp.test_size[1] // s) for s in model.head.strides]
    decoder = model.head.decode_outputs

    val_preproc = ValTransform(legacy=False)
    class_names = _load_class_names(args.dataset_root)
    pre_cfg = PreprocessConfig()
    det_cfg = DetectionConfig(
        lang=args.lang,
        gpu=False,
        download_enabled=not bool(args.no_download),
        model_storage_directory=args.model_dir,
    )
    backend = OpenVinoBackend(args.openvino_xml)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        pre = preprocess(img_path, pre_cfg)
        model_bgr = pre["model_bgr"]
        h, w = model_bgr.shape[:2]
        ratio = min(exp.test_size[0] / h, exp.test_size[1] / w)
        img, _ = val_preproc(model_bgr, None, exp.test_size)
        inp = np.expand_dims(img, 0).astype(np.float32, copy=False)

        raw = backend.infer(inp)
        yolo_out = _decode_and_postprocess(raw, decoder, exp.num_classes, exp.test_conf, exp.nmsthre)
        yolo_dets = _yolox_detections_from_output(yolo_out, ratio, class_names)

        text_json = detect_text_boxes(model_bgr.copy(), det_cfg)
        text_dets = _text_detections(text_json)
        detections = yolo_dets + text_dets

        yolox_overlay = _draw_yolox_boxes(model_bgr, yolo_dets)
        text_overlay = draw_text_boxes(model_bgr.copy(), text_json)
        ensemble_overlay = draw_text_boxes(yolox_overlay.copy(), text_json)

        stem = Path(img_path).stem
        out_json = {
            "coord_space": "model",
            "image": {
                "file_name": Path(img_path).name,
                "width": int(model_bgr.shape[1]),
                "height": int(model_bgr.shape[0]),
                "orig_width": int(pre["orig_bgr"].shape[1]),
                "orig_height": int(pre["orig_bgr"].shape[0]),
                "resize_ratio": float(pre["resize_ratio"]),
                "geom_angle_deg": float(pre["geom_angle_deg"]),
                "deskew_matrix": pre["deskew_matrix"].tolist(),
                "deskew_matrix_inv": pre["deskew_matrix_inv"].tolist(),
            },
            "meta": {
                "backend": "openvino",
                "yolox_conf": float(exp.test_conf),
                "yolox_nms": float(exp.nmsthre),
                "test_size": list(exp.test_size),
            },
            "detections": detections,
        }

        with open(out_dir / f"{stem}_ensemble.json", "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        cv2.imwrite(str(out_dir / f"{stem}_yolox.png"), yolox_overlay)
        cv2.imwrite(str(out_dir / f"{stem}_text.png"), text_overlay)
        cv2.imwrite(str(out_dir / f"{stem}_ensemble.png"), ensemble_overlay)

    print(f"Saved {len(images)} images to {out_dir}")


if __name__ == "__main__":
    main()
