#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Must be set before importing paddle/paddleocr.
if "--disable-mkldnn" in sys.argv:
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_mkldnn"] = "0"
    os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
    os.environ["FLAGS_new_executor"] = "0"
    os.environ["FLAGS_use_pir_api"] = "0"


@dataclass(frozen=True)
class OcrLine:
    bbox_xyxy: Tuple[float, float, float, float]  # original image coord space
    text: str
    score: float
    rotation: str  # "rot0" | "rot90_cw"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_dict_lines(path: str) -> List[str]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except Exception:
        try:
            raw = Path(path).read_text(encoding="utf-8-sig")
        except Exception:
            return []
    lines = [ln.strip() for ln in raw.splitlines()]
    return [ln for ln in lines if ln]


def _ensure_out_path(out_arg: str) -> Path:
    p = Path(out_arg)
    if p.exists() and p.is_dir():
        return p / "ocr.json"
    if p.suffix.lower() == ".json":
        return p
    return p / "ocr.json"


def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)


def _cap_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img
    scale = max_side / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _rotate90_cw(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def _map_point_from_rot90_cw_to_orig(xr: float, yr: float, orig_h: float) -> Tuple[float, float]:
    # new_x = orig_h - 1 - y
    # new_y = x
    # => x = new_y, y = orig_h - 1 - new_x
    x = yr
    y = (orig_h - 1.0) - xr
    return x, y


def _parse_poly(poly: Any) -> Optional[np.ndarray]:
    if poly is None:
        return None
    if isinstance(poly, np.ndarray):
        arr = poly.astype(np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        return None
    if isinstance(poly, (list, tuple)):
        try:
            arr = np.array(poly, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        return None
    if isinstance(poly, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", poly)
        if len(nums) < 8:
            return None
        vals = [float(x) for x in nums]
        pts = []
        for i in range(0, len(vals) - 1, 2):
            pts.append([vals[i], vals[i + 1]])
        arr = np.array(pts, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
        return None
    return None


def _poly_to_bbox_xyxy(poly: np.ndarray) -> Tuple[float, float, float, float]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _extract_lines_from_predict(
    pages: Any,
    rotation: str,
    scaled_orig_h: int,
    scale_back: float,
    map_from_rotated: bool,
    min_line_score: float,
) -> List[OcrLine]:
    if pages is None:
        return []
    if isinstance(pages, dict):
        pages_list = [pages]
    elif isinstance(pages, list):
        pages_list = pages
    else:
        return []

    out: List[OcrLine] = []
    for page in pages_list:
        if not isinstance(page, dict):
            continue
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("rec_polys") or page.get("dt_polys") or []

        n = min(len(texts), len(scores), len(polys))
        for i in range(n):
            txt = str(texts[i]) if texts[i] is not None else ""
            txt = txt.strip()
            sc = float(scores[i]) if scores[i] is not None else 0.0
            if txt == "" or sc < min_line_score:
                continue

            poly = _parse_poly(polys[i])
            if poly is None:
                continue

            if map_from_rotated:
                mapped = []
                for (xr, yr) in poly.tolist():
                    x_s, y_s = _map_point_from_rot90_cw_to_orig(float(xr), float(yr), float(scaled_orig_h))
                    mapped.append([x_s, y_s])
                poly_s = np.array(mapped, dtype=np.float32)
            else:
                poly_s = poly.astype(np.float32)

            x1, y1, x2, y2 = _poly_to_bbox_xyxy(poly_s)

            # back to original coords
            x1 /= scale_back
            y1 /= scale_back
            x2 /= scale_back
            y2 /= scale_back

            out.append(OcrLine(bbox_xyxy=(x1, y1, x2, y2), text=txt, score=sc, rotation=rotation))
    return out


def _extract_lines_from_ocr27(
    result: Any,
    rotation: str,
    scaled_orig_h: int,
    scale_back: float,
    map_from_rotated: bool,
    min_line_score: float,
) -> List[OcrLine]:
    if not result:
        return []
    # PaddleOCR 2.x returns list per image; each item: [poly, (text, score)]
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        lines_in = result[0]
    elif isinstance(result, list):
        lines_in = result
    else:
        return []

    out: List[OcrLine] = []
    for item in lines_in:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        poly = _parse_poly(item[0])
        if poly is None:
            continue
        # PaddleOCR 2.x: [poly, (text, score)]
        # RapidOCR: [poly, text, score]
        txt = ""
        sc = 0.0
        rec = item[1]
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            txt = str(rec[0]).strip() if rec[0] is not None else ""
            try:
                sc = float(rec[1])
            except Exception:
                sc = 0.0
        elif len(item) >= 3:
            txt = str(item[1]).strip() if item[1] is not None else ""
            try:
                sc = float(item[2])
            except Exception:
                sc = 0.0
        if txt == "" or sc < min_line_score:
            continue

        if map_from_rotated:
            mapped = []
            for (xr, yr) in poly.tolist():
                x_s, y_s = _map_point_from_rot90_cw_to_orig(float(xr), float(yr), float(scaled_orig_h))
                mapped.append([x_s, y_s])
            poly_s = np.array(mapped, dtype=np.float32)
        else:
            poly_s = poly.astype(np.float32)

        x1, y1, x2, y2 = _poly_to_bbox_xyxy(poly_s)

        # back to original coords
        x1 /= scale_back
        y1 /= scale_back
        x2 /= scale_back
        y2 /= scale_back

        out.append(OcrLine(bbox_xyxy=(x1, y1, x2, y2), text=txt, score=sc, rotation=rotation))
    return out


def _dedupe_lines(lines: List[OcrLine]) -> List[OcrLine]:
    if not lines:
        return []

    def norm_text(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "").strip()).lower()

    def iou(a: Sequence[float], b: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # sort by score desc to keep best line among near-duplicates
    sorted_lines = sorted(lines, key=lambda ln: -ln.score)
    kept: List[OcrLine] = []

    for ln in sorted_lines:
        t = norm_text(ln.text)
        if not t:
            continue
        dup = False
        for prev in kept:
            if t != norm_text(prev.text):
                continue
            if iou(ln.bbox_xyxy, prev.bbox_xyxy) >= 0.35:
                dup = True
                break
        if not dup:
            kept.append(ln)

    return kept


def _clamp_bbox(b: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(int(round(x1)), w))
    y1 = max(0, min(int(round(y1)), h))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _sort_lines_reading_order(lines: List[OcrLine]) -> List[OcrLine]:
    def key(ln: OcrLine) -> Tuple[float, float]:
        x1, y1, _, _ = ln.bbox_xyxy
        return (y1, x1)
    return sorted(lines, key=key)


_LAT_TO_CYR = {
    "A": "А",
    "B": "В",
    "C": "С",
    "D": "Д",
    "E": "Е",
    "F": "Ф",
    "G": "Г",
    "H": "Н",
    "I": "И",
    "J": "Й",
    "K": "К",
    "L": "Л",
    "M": "М",
    "N": "П",
    "O": "О",
    "P": "Р",
    "Q": "Я",
    "R": "Г",
    "S": "С",
    "T": "Т",
    "U": "И",
    "V": "В",
    "W": "Ш",
    "X": "Х",
    "Y": "У",
    "Z": "З",
    "a": "а",
    "b": "в",
    "c": "с",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "н",
    "i": "и",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "п",
    "o": "о",
    "p": "р",
    "q": "я",
    "r": "г",
    "s": "с",
    "t": "т",
    "u": "и",
    "v": "в",
    "w": "ш",
    "x": "х",
    "y": "у",
    "z": "з",
    "3": "З",
    "4": "Ч",
    "6": "б",
    "0": "О",
    "1": "І",
}


_ROMAN_TO_ASCII = {
    "\u2160": "I",
    "\u2161": "II",
    "\u2162": "III",
    "\u2163": "IV",
    "\u2164": "V",
    "\u2165": "VI",
    "\u2166": "VII",
    "\u2167": "VIII",
    "\u2168": "IX",
    "\u2169": "X",
}


def _fix_cyrillic_confusables(text: str, dict_cyr_chars: int) -> str:
    if not text:
        return text
    if dict_cyr_chars < 60:
        return text
    # normalize roman numerals first
    out = "".join(_ROMAN_TO_ASCII.get(ch, ch) for ch in text)
    out = "".join(_LAT_TO_CYR.get(ch, ch) for ch in out)
    return out


def _load_blocks_optional(blocks_path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not blocks_path:
        return None
    raw = _read_json(Path(blocks_path))
    if isinstance(raw, list):
        blocks = raw
    elif isinstance(raw, dict):
        if "blocks" in raw and isinstance(raw["blocks"], list):
            blocks = raw["blocks"]
        elif "text_blocks" in raw and isinstance(raw["text_blocks"], list):
            blocks = raw["text_blocks"]
        else:
            raise ValueError(f"Unsupported blocks json format: keys={list(raw.keys())[:20]}")
    else:
        raise ValueError("Unsupported blocks json format (expected list or dict).")

    out: List[Dict[str, Any]] = []
    for i, b in enumerate(blocks, start=1):
        if not isinstance(b, dict):
            continue
        bbox = b.get("bbox_xyxy") or b.get("bbox") or b.get("xyxy")
        if not bbox or len(bbox) != 4:
            continue
        block_id = b.get("block_id") or b.get("id") or i
        out.append({"block_id": int(block_id), "bbox_xyxy": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]})
    return out


def _area_xyxy(b: Sequence[float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _assign_lines_to_blocks(
    lines: List[OcrLine],
    blocks: List[Dict[str, Any]],
    assign_coverage_thresh: float,
) -> List[Dict[str, Any]]:
    lines_sorted = _sort_lines_reading_order(lines)
    out: List[Dict[str, Any]] = []

    for b in blocks:
        bb = b["bbox_xyxy"]
        block_bbox = tuple(map(float, bb))
        assigned: List[OcrLine] = []
        for ln in lines_sorted:
            inter = _intersection_area(block_bbox, ln.bbox_xyxy)
            if inter <= 0:
                continue
            cov = inter / max(1e-6, _area_xyxy(ln.bbox_xyxy))
            if cov >= assign_coverage_thresh:
                assigned.append(ln)

        assigned = _sort_lines_reading_order(assigned)
        if assigned:
            text = "\n".join([x.text for x in assigned]).strip()
            conf = float(np.mean([x.score for x in assigned])) if assigned else 0.0
            chosen = [{
                "engine": "paddleocr_fullimage_assign",
                "rotation": "rot0/rot90",
                "conf": conf,
                "score": conf,
                "paddle_debug": {"items": len(assigned), "lines": len(assigned)},
            }]
        else:
            text = ""
            conf = 0.0
            chosen = [{
                "engine": "paddleocr_fullimage_assign",
                "rotation": "rot0/rot90",
                "conf": 0.0,
                "score": 0.0,
                "paddle_debug": {"items": 0, "lines": 0},
            }]

        out.append({
            "block_id": int(b["block_id"]),
            "bbox_xyxy": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
            "text": text,
            "confidence": conf,
            "confidence_available": True,
            "chosen": chosen,
        })

    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to image (png/jpg)")
    p.add_argument("--out", required=True, help="Output: .json file OR directory")
    p.add_argument("--blocks", default=None, help="Optional: text_blocks.json to assign lines into given blocks")
    p.add_argument("--lang", default="ru")
    p.add_argument("--device", default="cpu", help="PaddleOCR device, e.g. cpu/gpu")
    p.add_argument("--backend", choices=["paddle", "rapidocr"], default="paddle")
    p.add_argument("--rapidocr-det", default="detection/v5/det.onnx", help="HF path or local path")
    p.add_argument("--rapidocr-rec", default="languages/eslav/rec.onnx", help="HF path or local path")
    p.add_argument("--rapidocr-dict", default="languages/eslav/dict.txt", help="HF path or local path")
    p.add_argument("--max-side-limit", type=int, default=4000, help="Max image side for OCR to avoid OOM")
    p.add_argument("--upscale-factor", type=float, default=3.0)
    p.add_argument("--try-rotate-90", action="store_true")
    p.add_argument("--min-line-score", type=float, default=0.30)
    p.add_argument("--assign-coverage-thresh", type=float, default=0.70)
    p.add_argument("--use-doc-orientation-classify", action="store_true", default=False)
    p.add_argument("--use-doc-unwarping", action="store_true", default=False)
    p.add_argument("--use-textline-orientation", action="store_true", default=False)
    p.add_argument("--text-rec-score-thresh", type=float, default=0.0)
    p.add_argument("--no-source-check", action="store_true")
    p.add_argument("--disable-mkldnn", action="store_true", help="Disable OneDNN/MKLDNN to avoid CPU runtime errors")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.no_source_check:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    if args.disable_mkldnn:
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_mkldnn"] = "0"
        os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
        os.environ["FLAGS_new_executor"] = "0"
        os.environ["FLAGS_use_pir_api"] = "0"
        try:
            import paddle

            paddle.set_flags(
                {
                    "FLAGS_use_mkldnn": False,
                    "FLAGS_enable_mkldnn": False,
                    "FLAGS_use_pir_api": False,
                    "FLAGS_new_executor": False,
                }
            )
        except Exception:
            pass

    img_path = Path(args.input)
    out_path = _ensure_out_path(args.out)

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    h0, w0 = img_bgr.shape[:2]
    scale = float(args.upscale_factor)
    img_scaled = _resize(img_bgr, scale)
    img_scaled = _cap_max_side(img_scaled, int(args.max_side_limit))
    hs, ws = img_scaled.shape[:2]

    ocr = None
    if args.backend == "paddle":
        from paddleocr import PaddleOCR

        use_gpu = str(args.device).lower() in {"gpu", "cuda"}
        ocr = PaddleOCR(
            lang=str(args.lang),
            device=str(args.device),
            use_gpu=use_gpu,
            use_angle_cls=bool(args.use_textline_orientation),
            use_doc_orientation_classify=bool(args.use_doc_orientation_classify),
            use_doc_unwarping=bool(args.use_doc_unwarping),
            use_textline_orientation=bool(args.use_textline_orientation),
            text_rec_score_thresh=float(args.text_rec_score_thresh),
        )
    elif args.backend == "rapidocr":
        from huggingface_hub import hf_hub_download
        from rapidocr_onnxruntime import RapidOCR
        try:
            from rapidocr_onnxruntime.utils import UpdateParameters
        except Exception:
            UpdateParameters = None
        try:
            from rapidocr_onnxruntime.utils.parse_parameters import UpdateParameters as UpdateParametersV2
        except Exception:
            UpdateParametersV2 = None

        def _patch_update_params(cls):
            if cls is None:
                return
            if hasattr(cls, "update_rec_params"):
                _orig = cls.update_rec_params

                def _patched(self, config, rec_dict):
                    if rec_dict:
                        keys_path = rec_dict.get("rec_keys_path") or rec_dict.get("rec_character_path")
                        rec_char = rec_dict.get("rec_character")
                        if keys_path and "keys_path" not in rec_dict:
                            rec_dict["keys_path"] = keys_path
                        if keys_path and "character_path" not in rec_dict:
                            rec_dict["character_path"] = keys_path
                        if keys_path and "character_dict_path" not in rec_dict:
                            rec_dict["character_dict_path"] = keys_path
                        if rec_char and "character" not in rec_dict:
                            rec_dict["character"] = rec_char
                        if keys_path and "character" not in rec_dict:
                            chars = _read_dict_lines(str(keys_path))
                            if chars:
                                rec_dict["character"] = chars
                        if "rec_keys_path" in rec_dict:
                            del rec_dict["rec_keys_path"]
                        if "rec_character_path" in rec_dict:
                            del rec_dict["rec_character_path"]
                        if "rec_character_dict_path" in rec_dict:
                            del rec_dict["rec_character_dict_path"]
                        if "rec_character" in rec_dict:
                            del rec_dict["rec_character"]
                    return _orig(self, config, rec_dict)

                cls.update_rec_params = _patched
                return
            if hasattr(cls, "update_params"):
                _orig = cls.update_params

                def _patched(self, *args, **kwargs):
                    params = None
                    if "params" in kwargs:
                        params = kwargs.get("params")
                    elif len(args) >= 2 and isinstance(args[1], dict):
                        params = args[1]
                    if params and isinstance(params, dict):
                        if "rec_keys_path" in params or "rec_character" in params or "rec_character_path" in params:
                            new_params = dict(params)
                            keys_path = new_params.get("rec_keys_path") or new_params.get("rec_character_path")
                            rec_char = new_params.get("rec_character")
                            if keys_path and "keys_path" not in new_params:
                                new_params["keys_path"] = keys_path
                            if keys_path and "character_path" not in new_params:
                                new_params["character_path"] = keys_path
                            if keys_path and "character_dict_path" not in new_params:
                                new_params["character_dict_path"] = keys_path
                            if rec_char and "character" not in new_params:
                                new_params["character"] = rec_char
                            if keys_path and "character" not in new_params:
                                chars = _read_dict_lines(str(keys_path))
                                if chars:
                                    new_params["character"] = chars
                            if "rec_keys_path" in new_params:
                                del new_params["rec_keys_path"]
                            if "rec_character_path" in new_params:
                                del new_params["rec_character_path"]
                            if "rec_character_dict_path" in new_params:
                                del new_params["rec_character_dict_path"]
                            if "rec_character" in new_params:
                                del new_params["rec_character"]
                            if "params" in kwargs:
                                kwargs["params"] = new_params
                            else:
                                args = (args[0], new_params, *args[2:])
                    return _orig(self, *args, **kwargs)

                cls.update_params = _patched

        _patch_update_params(UpdateParameters)
        _patch_update_params(UpdateParametersV2)

        def _resolve(path_or_hf: str) -> str:
            if os.path.exists(path_or_hf):
                return path_or_hf
            return hf_hub_download("monkt/paddleocr-onnx", path_or_hf)

        det_path = _resolve(str(args.rapidocr_det))
        rec_path = _resolve(str(args.rapidocr_rec))
        dict_path = _resolve(str(args.rapidocr_dict))

        chars = _read_dict_lines(dict_path)
        dict_cyr = sum(1 for ch in "".join(chars) if ("А" <= ch <= "я") or (ch in "Ёё"))
        rapid_debug = {
            "det_path": str(det_path),
            "rec_path": str(rec_path),
            "dict_path": str(dict_path),
            "dict_cyrillic_chars": int(dict_cyr),
        }

        # Direct config injection for versions that don't pass rec_* via UpdateParameters
        try:
            import rapidocr_onnxruntime.main as rapid_main
            from pathlib import Path as _P

            if hasattr(rapid_main, "root_dir"):
                config_path = str(_P(rapid_main.root_dir) / "config.yaml")
                if _P(config_path).exists():
                    config = rapid_main.read_yaml(config_path)
                    if isinstance(config, dict) and "Rec" in config:
                        rec_cfg = config["Rec"]
                        if isinstance(rec_cfg, dict):
                            rec_cfg["model_path"] = rec_path
                            rec_cfg["keys_path"] = dict_path
                            rec_cfg["character_path"] = dict_path
                            rec_cfg["character_dict_path"] = dict_path
                            if chars:
                                rec_cfg["character"] = chars
                    ocr = rapid_main.RapidOCR(**config)
                else:
                    ocr = None
            else:
                ocr = None
        except Exception:
            ocr = None

        if ocr is None:
            ocr = RapidOCR(
                det_model_path=det_path,
                rec_model_path=rec_path,
                rec_keys_path=dict_path,
                rec_character_path=dict_path,
                rec_character_dict_path=dict_path,
                rec_character=chars if chars else None,
            )

    lines: List[OcrLine] = []

    # rot0
    if args.backend == "rapidocr":
        res0, _ = ocr(_bgr_to_rgb(img_scaled))
        lines.extend(_extract_lines_from_ocr27(
            result=res0,
            rotation="rot0",
            scaled_orig_h=hs,
            scale_back=scale,
            map_from_rotated=False,
            min_line_score=float(args.min_line_score),
        ))
    else:
        if hasattr(ocr, "predict"):
            res0 = ocr.predict(_bgr_to_rgb(img_scaled))
            lines.extend(_extract_lines_from_predict(
                pages=res0,
                rotation="rot0",
                scaled_orig_h=hs,
                scale_back=scale,
                map_from_rotated=False,
                min_line_score=float(args.min_line_score),
            ))
        else:
            res0 = ocr.ocr(_bgr_to_rgb(img_scaled), cls=bool(args.use_textline_orientation))
            lines.extend(_extract_lines_from_ocr27(
                result=res0,
                rotation="rot0",
                scaled_orig_h=hs,
                scale_back=scale,
                map_from_rotated=False,
                min_line_score=float(args.min_line_score),
            ))

    # rot90
    if args.try_rotate_90:
        img_rot = _rotate90_cw(img_scaled)
        if args.backend == "rapidocr":
            res90, _ = ocr(_bgr_to_rgb(img_rot))
            lines.extend(_extract_lines_from_ocr27(
                result=res90,
                rotation="rot90_cw",
                scaled_orig_h=hs,
                scale_back=scale,
                map_from_rotated=True,
                min_line_score=float(args.min_line_score),
            ))
        else:
            if hasattr(ocr, "predict"):
                res90 = ocr.predict(_bgr_to_rgb(img_rot))
                lines.extend(_extract_lines_from_predict(
                    pages=res90,
                    rotation="rot90_cw",
                    scaled_orig_h=hs,
                    scale_back=scale,
                    map_from_rotated=True,
                    min_line_score=float(args.min_line_score),
                ))
            else:
                res90 = ocr.ocr(_bgr_to_rgb(img_rot), cls=bool(args.use_textline_orientation))
                lines.extend(_extract_lines_from_ocr27(
                    result=res90,
                    rotation="rot90_cw",
                    scaled_orig_h=hs,
                    scale_back=scale,
                    map_from_rotated=True,
                    min_line_score=float(args.min_line_score),
                ))

    lines = _dedupe_lines(lines)
    lines = _sort_lines_reading_order(lines)
    if args.backend == "rapidocr" and "rapid_debug" in locals():
        fixed = []
        for ln in lines:
            fixed.append(OcrLine(
                bbox_xyxy=ln.bbox_xyxy,
                text=_fix_cyrillic_confusables(ln.text, rapid_debug.get("dict_cyrillic_chars", 0)),
                score=ln.score,
                rotation=ln.rotation,
            ))
        lines = fixed

    blocks_in = _load_blocks_optional(args.blocks)

    if blocks_in is None:
        # DETECT MODE: each OCR line becomes a block
        out_blocks: List[Dict[str, Any]] = []
        for i, ln in enumerate(lines, start=1):
            x1, y1, x2, y2 = _clamp_bbox(ln.bbox_xyxy, w0, h0)
            out_blocks.append({
                "block_id": i,
                "bbox_xyxy": [x1, y1, x2, y2],
                "text": ln.text,
                "confidence": float(ln.score),
                "confidence_available": True,
                "chosen": [{
                    "engine": "paddleocr_fullimage_detect",
                    "rotation": ln.rotation,
                    "conf": float(ln.score),
                    "score": float(ln.score),
                    "paddle_debug": {"items": 1, "lines": 1},
                }],
            })

        out_json = {
            "coord_space": "auto",
            "engine": f"{args.backend}_fullimage_detect_v1",
            "ocr_config": {
                "lang": str(args.lang),
                "device": str(args.device),
                "backend": str(args.backend),
                "upscale_factor": scale,
                "try_rotate_90": bool(args.try_rotate_90),
                "min_line_score": float(args.min_line_score),
                "use_doc_orientation_classify": bool(args.use_doc_orientation_classify),
                "use_doc_unwarping": bool(args.use_doc_unwarping),
                "use_textline_orientation": bool(args.use_textline_orientation),
                "text_rec_score_thresh": float(args.text_rec_score_thresh),
            },
            "blocks": out_blocks,
        }
        if args.backend == "rapidocr" and "rapid_debug" in locals():
            out_json["rapidocr_debug"] = rapid_debug
        _write_json(out_path, out_json)
        return

    # ASSIGN MODE (optional)
    out_blocks = _assign_lines_to_blocks(
        lines=lines,
        blocks=blocks_in,
        assign_coverage_thresh=float(args.assign_coverage_thresh),
    )

    out_json = {
        "coord_space": "auto",
        "engine": f"{args.backend}_fullimage_assign_v1",
        "ocr_config": {
            "lang": str(args.lang),
            "device": str(args.device),
            "backend": str(args.backend),
            "upscale_factor": scale,
            "try_rotate_90": bool(args.try_rotate_90),
            "min_line_score": float(args.min_line_score),
            "assign_coverage_thresh": float(args.assign_coverage_thresh),
            "use_doc_orientation_classify": bool(args.use_doc_orientation_classify),
            "use_doc_unwarping": bool(args.use_doc_unwarping),
            "use_textline_orientation": bool(args.use_textline_orientation),
            "text_rec_score_thresh": float(args.text_rec_score_thresh),
        },
        "blocks": out_blocks,
    }
    if args.backend == "rapidocr" and "rapid_debug" in locals():
        out_json["rapidocr_debug"] = rapid_debug
    _write_json(out_path, out_json)


if __name__ == "__main__":
    main()
