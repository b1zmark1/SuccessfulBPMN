from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import cv2

from preprocess import PreprocessConfig, preprocess
from detect import DetectionConfig, detect_text_boxes, draw_text_boxes


def _add_yolox_to_path(repo_root: str) -> None:
    yolox_dir = os.path.join(repo_root, "YOLOX")
    if os.path.isdir(yolox_dir) and yolox_dir not in sys.path:
        sys.path.insert(0, yolox_dir)


def _get_images(path: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    p = Path(path)
    if p.is_file():
        return [str(p.resolve())]
    out = []
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


class OnnxBackend:
    name = "onnxruntime"

    def __init__(self, model_path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def infer(self, inp: np.ndarray) -> np.ndarray:
        out = self.sess.run(None, {self.input_name: inp})[0]
        return np.asarray(out)


class OpenVinoBackend:
    name = "openvino"

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
    decode_output: bool,
    num_classes: int,
    conf: float,
    nms: float,
):
    pred = torch.from_numpy(raw)
    if decode_output:
        pred = decoder(pred, dtype=pred.type())
    from yolox.utils import postprocess

    outputs = postprocess(pred, num_classes, conf, nms, class_agnostic=True)
    return outputs[0] if outputs and len(outputs) > 0 else None


def _yolox_detections_from_output(
    output,
    ratio: float,
    class_names: List[str] | None,
) -> List[Dict[str, Any]]:
    if output is None:
        return []
    output = output.cpu()
    bboxes = output[:, 0:4]
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    detections: List[Dict[str, Any]] = []
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


def _text_detections(text_json: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not text_json:
        return []
    out: List[Dict[str, Any]] = []
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
                "source": "easyocr",
            }
        )
    return out


def _draw_yolox_boxes(bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
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


def _run_backend(
    backend,
    images: List[str],
    out_root: Path,
    val_preproc,
    test_size,
    decoder,
    decode_output: bool,
    num_classes: int,
    conf: float,
    nms: float,
    class_names,
    pre_cfg: PreprocessConfig,
    det_cfg: DetectionConfig,
    warmup: int,
) -> Dict[str, Any]:
    backend_dir = out_root / backend.name
    det_out_dir = backend_dir / "detections"
    backend_dir.mkdir(parents=True, exist_ok=True)
    det_out_dir.mkdir(parents=True, exist_ok=True)

    per_file: List[Dict[str, Any]] = []
    timed_images = images[warmup:] if warmup > 0 and len(images) > warmup else images

    for idx, img_path in enumerate(images):
        t0 = time.perf_counter()
        pre = preprocess(img_path, pre_cfg)
        model_bgr = pre["model_bgr"]
        h, w = model_bgr.shape[:2]
        ratio = min(test_size[0] / h, test_size[1] / w)
        img, _ = val_preproc(model_bgr, None, test_size)
        inp = np.expand_dims(img, 0).astype(np.float32, copy=False)
        t1 = time.perf_counter()

        raw = backend.infer(inp)
        yolo_out = _decode_and_postprocess(raw, decoder, decode_output, num_classes, conf, nms)
        yolo_dets = _yolox_detections_from_output(yolo_out, ratio, class_names)
        t2 = time.perf_counter()

        text_json = detect_text_boxes(model_bgr.copy(), det_cfg)
        text_dets = _text_detections(text_json)
        t3 = time.perf_counter()

        dets = yolo_dets + text_dets
        t4 = time.perf_counter()

        # Save overlays like ensemble_infer.py
        yolox_overlay = _draw_yolox_boxes(model_bgr, yolo_dets)
        text_overlay = draw_text_boxes(model_bgr.copy(), text_json)
        ensemble_overlay = draw_text_boxes(yolox_overlay.copy(), text_json)

        is_warmup = idx < warmup
        rec = {
            "file_name": os.path.basename(img_path),
            "path": img_path,
            "warmup": is_warmup,
            "num_detections": len(dets),
            "time_ms": {
                "preprocess": (t1 - t0) * 1000.0,
                "object_detector": (t2 - t1) * 1000.0,
                "text_detector": (t3 - t2) * 1000.0,
                "merge": (t4 - t3) * 1000.0,
                "total": (t4 - t0) * 1000.0,
            },
        }
        per_file.append(rec)

        out_json = {
            "image": {
                "file_name": os.path.basename(img_path),
                "path": img_path,
                "width": int(w),
                "height": int(h),
                "resize_ratio": float(pre["resize_ratio"]),
                "geom_angle_deg": float(pre["geom_angle_deg"]),
            },
            "meta": {
                "backend": backend.name,
                "yolox_conf": conf,
                "yolox_nms": nms,
                "test_size": list(test_size),
            },
            "detections": dets,
        }
        with open(det_out_dir / f"{Path(img_path).stem}_ensemble.json", "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        cv2.imwrite(str(det_out_dir / f"{Path(img_path).stem}_yolox.png"), yolox_overlay)
        cv2.imwrite(str(det_out_dir / f"{Path(img_path).stem}_text.png"), text_overlay)
        cv2.imwrite(str(det_out_dir / f"{Path(img_path).stem}_ensemble.png"), ensemble_overlay)

    used = [x for x in per_file if not x["warmup"]]
    mean_total = float(np.mean([x["time_ms"]["total"] for x in used])) if used else 0.0
    mean_pre = float(np.mean([x["time_ms"]["preprocess"] for x in used])) if used else 0.0
    mean_obj = float(np.mean([x["time_ms"]["object_detector"] for x in used])) if used else 0.0
    mean_text = float(np.mean([x["time_ms"]["text_detector"] for x in used])) if used else 0.0
    mean_merge = float(np.mean([x["time_ms"]["merge"] for x in used])) if used else 0.0

    summary = {
        "backend": backend.name,
        "num_files_total": len(images),
        "num_files_timed": len(used),
        "warmup_files": warmup,
        "avg_ms_per_file": {
            "preprocess": mean_pre,
            "object_detector": mean_obj,
            "text_detector": mean_text,
            "merge": mean_merge,
            "total": mean_total,
        },
        "per_file": per_file,
    }
    with open(backend_dir / "times.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser("Benchmark ONNX vs OpenVINO with preprocessing + ensemble")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--exp-file", required=True)
    ap.add_argument("--dataset-root", default="datasets/bpmn_full")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--openvino-xml", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--nms", type=float, default=0.65)
    ap.add_argument("--tsize", type=int, default=None)
    ap.add_argument("--no-decode-output", action="store_true")
    ap.add_argument("--warmup", type=int, default=1)
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

    # decoder from YOLOX head for exported models with decode_in_inference=False
    model = exp.get_model()
    # YOLOXHead.decode_outputs expects feature map shapes in head.hw.
    # For fixed test_size we can derive them directly from strides.
    model.head.hw = [
        (exp.test_size[0] // s, exp.test_size[1] // s) for s in model.head.strides
    ]
    decoder = model.head.decode_outputs
    decode_output = not bool(args.no_decode_output)

    val_preproc = ValTransform(legacy=False)
    class_names = _load_class_names(args.dataset_root)

    pre_cfg = PreprocessConfig()
    det_cfg = DetectionConfig(
        lang=args.lang,
        gpu=False,
        download_enabled=not bool(args.no_download),
        model_storage_directory=args.model_dir,
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    onnx_backend = OnnxBackend(args.onnx)
    ov_backend = OpenVinoBackend(args.openvino_xml)

    onnx_summary = _run_backend(
        onnx_backend,
        images,
        out_root,
        val_preproc,
        exp.test_size,
        decoder,
        decode_output,
        exp.num_classes,
        exp.test_conf,
        exp.nmsthre,
        class_names,
        pre_cfg,
        det_cfg,
        args.warmup,
    )

    ov_summary = _run_backend(
        ov_backend,
        images,
        out_root,
        val_preproc,
        exp.test_size,
        decoder,
        decode_output,
        exp.num_classes,
        exp.test_conf,
        exp.nmsthre,
        class_names,
        pre_cfg,
        det_cfg,
        args.warmup,
    )

    summary = {
        "images_root": str(Path(args.images).resolve()),
        "num_images": len(images),
        "preprocess_config": asdict(pre_cfg),
        "text_detector_config": asdict(det_cfg),
        "yolox": {
            "exp_file": str(Path(args.exp_file).resolve()),
            "test_size": list(exp.test_size),
            "conf": exp.test_conf,
            "nms": exp.nmsthre,
            "decode_output": decode_output,
        },
        "results": {
            "onnxruntime": onnx_summary["avg_ms_per_file"],
            "openvino": ov_summary["avg_ms_per_file"],
        },
    }
    with open(out_root / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Benchmark done.")
    print(f"ONNX avg total ms/file: {onnx_summary['avg_ms_per_file']['total']:.2f}")
    print(f"OpenVINO avg total ms/file: {ov_summary['avg_ms_per_file']['total']:.2f}")
    print(f"Artifacts: {out_root.resolve()}")


if __name__ == "__main__":
    main()
