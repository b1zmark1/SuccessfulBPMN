"""
assign_text.py

Что исправлено относительно типичного "cv2 debug + overlap к любым классам":
- Дебаг-отрисовка текста переведена на Pillow (кириллица нормальная, не "?????").
- По умолчанию OCR НЕ привязывается к sequence_flow (иначе ловишь мусор из-за больших bbox).
- Добавлена опциональная привязка OCR к событиям (start/end/intermediate) по близости.
- Привязка к sequence_flow (если включить) с фильтрами: max thickness / max area ratio / min conf / radius px.
- В output добавлены unassigned_text_blocks для контроля.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# ------------------------- geometry -------------------------


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def w(self) -> int:
        return max(0, self.x2 - self.x1)

    def h(self) -> int:
        return max(0, self.y2 - self.y1)

    def area(self) -> int:
        return self.w() * self.h()

    def clip(self, W: int, H: int) -> "Box":
        return Box(
            x1=max(0, min(self.x1, W)),
            y1=max(0, min(self.y1, H)),
            x2=max(0, min(self.x2, W)),
            y2=max(0, min(self.y2, H)),
        )

    def expand(self, px: int) -> "Box":
        return Box(self.x1 - px, self.y1 - px, self.x2 + px, self.y2 + px)

    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def intersect_area(self, other: "Box") -> int:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        return iw * ih

    def iou(self, other: "Box") -> float:
        inter = self.intersect_area(other)
        if inter <= 0:
            return 0.0
        union = self.area() + other.area() - inter
        if union <= 0:
            return 0.0
        return inter / union

    def dist_point(self, px: float, py: float) -> float:
        """Distance from point to rectangle (0 if inside)."""
        dx = 0.0
        if px < self.x1:
            dx = float(self.x1) - px
        elif px > self.x2:
            dx = px - float(self.x2)

        dy = 0.0
        if py < self.y1:
            dy = float(self.y1) - py
        elif py > self.y2:
            dy = py - float(self.y2)

        return math.hypot(dx, dy)


# ------------------------- parsing -------------------------


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_int_box(bbox_xyxy: List[float | int]) -> Box:
    x1, y1, x2, y2 = bbox_xyxy
    return Box(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def _clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _looks_like_garbage(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    # много не-букв/цифр
    good = sum(ch.isalnum() for ch in t)
    if good == 0:
        return True
    if len(t) >= 4:
        # повтор одного символа (часто мусор)
        uniq = set(t)
        if len(uniq) == 1:
            return True
    return False


# ------------------------- font / drawing -------------------------


def _find_font_path(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)

    env = os.getenv("PREPROC_FONT_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return str(p)

    # macOS / Linux / Windows candidates
    candidates = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        # Windows
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\times.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def _load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    fp = _find_font_path(font_path)
    if fp:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    # fallback (может не иметь кириллицы)
    return ImageFont.load_default()


def _draw_debug(
    image_path: str,
    detections: List[Dict[str, Any]],
    out_path: str,
    font_path: Optional[str],
    draw_seqflow: bool,
    draw_all: bool,
) -> None:
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = _load_font(font_path, size=14)

    for det in detections:
        cls = det["class"]
        if (not draw_all) and (cls == "sequence_flow") and (not draw_seqflow):
            continue

        box = _as_int_box(det["bbox_xyxy"]).clip(W, H)
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(0, 255, 0), width=2)

        label = cls
        text = _clean_text(det.get("text", ""))
        if text:
            # показываем первую строку, чтобы не заспамить
            first = text.split("\n", 1)[0][:60]
            label = f"{cls}: {first}"

        # фон под текстом для читабельности
        tx, ty = box.x1 + 2, max(0, box.y1 - 18)
        tw, th = draw.textbbox((tx, ty), label, font=font)[2:]
        draw.rectangle([tx - 2, ty - 1, tx + tw + 2, ty + th + 1], fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    img.save(out_path)


# ------------------------- assignment logic -------------------------


def _combine_text(blocks: List[Dict[str, Any]]) -> Tuple[str, float, List[int]]:
    if not blocks:
        return "", 0.0, []

    # сортируем в порядке чтения
    blocks_sorted = sorted(blocks, key=lambda b: (b["bbox"].y1, b["bbox"].x1))

    texts: List[str] = []
    conf_sum = 0.0
    w_sum = 0.0
    ids: List[int] = []

    for b in blocks_sorted:
        t = _clean_text(b["text"])
        if not t:
            continue
        texts.append(t)
        ids.append(int(b["block_id"]))
        # вес по площади блока (стабильнее среднего)
        w = float(max(1, b["bbox"].area()))
        w_sum += w
        conf_sum += float(b["confidence"]) * w

    out_text = "\n".join(texts).strip()
    out_conf = (conf_sum / w_sum) if w_sum > 0 else 0.0
    return out_text, out_conf, ids


def _parse_ocr_blocks(ocr_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in ocr_json.get("blocks", []):
        bbox_xyxy = b.get("bbox_xyxy")
        if not bbox_xyxy:
            continue
        text = b.get("text", "")
        conf = float(b.get("confidence", 0.0))
        out.append(
            {
                "block_id": int(b.get("block_id")),
                "bbox": _as_int_box(bbox_xyxy),
                "text": text,
                "confidence": conf,
            }
        )
    return out


def _parse_yolox_detections(det_json: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    meta = det_json.get("meta", {})
    img = det_json.get("image", {})
    dets_in = det_json.get("detections", [])

    dets_out: List[Dict[str, Any]] = []
    for i, d in enumerate(dets_in, start=1):
        cls = d.get("class") or d.get("class_name")
        if not cls:
            continue
        bbox_xyxy = d.get("bbox_xyxy")
        if not bbox_xyxy:
            continue
        dets_out.append(
            {
                "det_id": f"det_{i}",
                "class": str(cls),
                "score": float(d.get("score", 0.0)),
                "bbox_xyxy": [int(round(x)) for x in bbox_xyxy],
                "source": d.get("source", "yolox"),
            }
        )

    return {"image": img, "meta": meta}, dets_out


def assign_text(
    image_path: str,
    yolox_path: str,
    ocr_path: str,
    out_path: str,
    debug_image_path: Optional[str],
    overlap_thr: float,
    min_ocr_conf: float,
    shape_pad_px: int,
    attach_events: bool,
    event_radius_px: int,
    event_radius_k: float,
    attach_seqflow: bool,
    min_ocr_conf_seqflow: float,
    seqflow_radius_px: int,
    seqflow_max_thickness: int,
    seqflow_max_area_ratio: float,
    font_path: Optional[str],
    debug_draw_seqflow: bool,
    debug_draw_all: bool,
) -> None:
    yolox_json = _load_json(yolox_path)
    ocr_json = _load_json(ocr_path)

    header, dets = _parse_yolox_detections(yolox_json)
    blocks = _parse_ocr_blocks(ocr_json)

    img_for_size = Image.open(image_path)
    W, H = img_for_size.size
    img_for_size.close()
    img_area = float(W * H)

    # 1) привязка блоков по overlap к "shape" классам (без sequence_flow)
    attach_classes = {
        "task",
        "subprocess",
        "gateway_exclusive",
        "gateway_parallel",
        "gateway_inclusive",
        "text_annotation",
        "data_object",
    }

    det_boxes: List[Tuple[int, Dict[str, Any], Box]] = []
    for idx, det in enumerate(dets):
        b = _as_int_box(det["bbox_xyxy"]).clip(W, H).expand(shape_pad_px).clip(W, H)
        det_boxes.append((idx, det, b))

    # индексируем только det-ы нужных классов
    candidate_dets = [(idx, det, box) for (idx, det, box) in det_boxes if det["class"] in attach_classes]

    # назначение блок->дет (один блок к одному дету)
    assigned_blocks: Dict[str, List[Dict[str, Any]]] = {d["det_id"]: [] for d in dets}
    used_block_ids: set[int] = set()

    for b in blocks:
        t = _clean_text(b["text"])
        if not t:
            continue
        if b["confidence"] < min_ocr_conf:
            continue
        if _looks_like_garbage(t):
            continue

        best_idx: Optional[int] = None
        best_det_id: Optional[str] = None
        best_score = 0.0

        b_area = float(max(1, b["bbox"].area()))
        for _, det, box in candidate_dets:
            inter = b["bbox"].intersect_area(box)
            if inter <= 0:
                continue
            overlap = float(inter) / b_area  # насколько блок "внутри" фигуры
            if overlap < overlap_thr:
                continue
            # если несколько — берём максимальный overlap
            if overlap > best_score:
                best_score = overlap
                best_det_id = det["det_id"]

        if best_det_id:
            assigned_blocks[best_det_id].append(b)
            used_block_ids.add(int(b["block_id"]))

    # 2) события (start/end/intermediate) по близости — только НЕиспользованные блоки
    if attach_events:
        event_classes = {"start_event", "end_event", "intermediate_event"}
        event_dets = [(idx, det, _as_int_box(det["bbox_xyxy"]).clip(W, H)) for idx, det in enumerate(dets) if det["class"] in event_classes]

        # Чтобы не перехватывать текст внутри task — соберём task боксы
        task_boxes = [_as_int_box(det["bbox_xyxy"]).clip(W, H).expand(shape_pad_px) for det in dets if det["class"] == "task"]

        for b in blocks:
            bid = int(b["block_id"])
            if bid in used_block_ids:
                continue
            t = _clean_text(b["text"])
            if not t:
                continue
            if b["confidence"] < min_ocr_conf:
                continue
            if _looks_like_garbage(t):
                continue

            # если блок заметно внутри какой-то task — не считаем событием
            inside_task = False
            for tb in task_boxes:
                inter = b["bbox"].intersect_area(tb)
                if inter > 0 and (float(inter) / float(max(1, b["bbox"].area()))) > 0.25:
                    inside_task = True
                    break
            if inside_task:
                continue

            bx, by = b["bbox"].center()
            best_det_id = None
            best_dist = 1e18
            for _, det, ebox in event_dets:
                ex, ey = ebox.center()
                dist = math.hypot(bx - ex, by - ey)
                radius = float(event_radius_px) + float(event_radius_k) * float(max(ebox.w(), ebox.h()))
                if dist <= radius and dist < best_dist:
                    best_dist = dist
                    best_det_id = det["det_id"]

            if best_det_id:
                assigned_blocks[best_det_id].append(b)
                used_block_ids.add(bid)

    # 3) sequence_flow (опционально) — строго фильтруем, иначе будет мусор
    if attach_seqflow:
        seq_dets = [(idx, det, _as_int_box(det["bbox_xyxy"]).clip(W, H)) for idx, det in enumerate(dets) if det["class"] == "sequence_flow"]

        for b in blocks:
            bid = int(b["block_id"])
            if bid in used_block_ids:
                continue
            t = _clean_text(b["text"])
            if not t:
                continue
            if b["confidence"] < min_ocr_conf_seqflow:
                continue
            if len(t.replace("\n", " ").strip()) < 3:
                continue
            if _looks_like_garbage(t):
                continue

            bx, by = b["bbox"].center()

            best_det_id = None
            best_dist = 1e18
            for _, det, sbox in seq_dets:
                # фильтр по "тонкости" и площади (отбрасываем жирные ложные flow bbox)
                thickness = min(sbox.w(), sbox.h())
                if thickness > seqflow_max_thickness:
                    continue
                if (float(sbox.area()) / img_area) > seqflow_max_area_ratio:
                    continue

                dist = sbox.dist_point(bx, by)
                if dist <= float(seqflow_radius_px) and dist < best_dist:
                    best_dist = dist
                    best_det_id = det["det_id"]

            if best_det_id:
                assigned_blocks[best_det_id].append(b)
                used_block_ids.add(bid)

    # собрать выход
    out_dets: List[Dict[str, Any]] = []
    for det in dets:
        det_id = det["det_id"]
        blocks_for_det = assigned_blocks.get(det_id, [])
        text, conf, block_ids = _combine_text(blocks_for_det)

        out_dets.append(
            {
                "det_id": det_id,
                "class": det["class"],
                "score": det["score"],
                "bbox_xyxy": det["bbox_xyxy"],
                "text": text,
                "text_block_ids": block_ids,
                "text_conf": conf,
                "source": det.get("source", "yolox"),
            }
        )

    unassigned = [
        {
            "block_id": int(b["block_id"]),
            "bbox_xyxy": [b["bbox"].x1, b["bbox"].y1, b["bbox"].x2, b["bbox"].y2],
            "text": _clean_text(b["text"]),
            "confidence": float(b["confidence"]),
        }
        for b in blocks
        if int(b["block_id"]) not in used_block_ids and _clean_text(b["text"])
    ]

    out_obj = {
        "image": {"path": image_path},
        "yolox_meta": header.get("meta", {}),
        "detections": out_dets,
        "unassigned_text_blocks": unassigned,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    if debug_image_path:
        _draw_debug(
            image_path=image_path,
            detections=out_dets,
            out_path=debug_image_path,
            font_path=font_path,
            draw_seqflow=debug_draw_seqflow,
            draw_all=debug_draw_all,
        )


# ------------------------- CLI -------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to source image")
    p.add_argument("--det", required=True, help="Path to YOLOX json (diagram1_ensemble.json)")
    p.add_argument("--ocr", required=True, help="Path to OCR json (run3_ocr_tuned/ocr.json)")
    p.add_argument("--out", required=True, help="Output json path")
    p.add_argument("--debug-image", default="", help="Optional path to debug overlay png")

    p.add_argument("--overlap-thr", type=float, default=0.55, help="Block area overlap threshold for shape assignment")
    p.add_argument("--min-ocr-conf", type=float, default=0.2, help="Min OCR confidence for shape/event assignment")
    p.add_argument("--shape-pad-px", type=int, default=6, help="Expand det bbox by px when matching blocks")

    p.add_argument("--attach-events", action="store_true", help="Attach remaining text blocks to events by proximity")
    p.add_argument("--event-radius-px", type=int, default=45, help="Absolute radius in px for event proximity")
    p.add_argument("--event-radius-k", type=float, default=0.8, help="Radius multiplier of event size")

    p.add_argument("--attach-seqflow", action="store_true", help="Attach remaining blocks to sequence_flow (strict filters)")
    p.add_argument("--min-ocr-conf-seqflow", type=float, default=0.6, help="Min OCR confidence for seqflow assignment")
    p.add_argument("--seqflow-radius-px", type=int, default=35, help="Max distance from block center to seqflow bbox")
    p.add_argument("--seqflow-max-thickness", type=int, default=22, help="Reject seqflow bbox if thickness > this")
    p.add_argument("--seqflow-max-area-ratio", type=float, default=0.02, help="Reject seqflow bbox if area ratio > this")

    p.add_argument("--font", default="", help="TTF/TTC font path for Cyrillic in debug (optional)")

    p.add_argument("--debug-draw-seqflow", action="store_true", help="Draw sequence_flow boxes in debug image")
    p.add_argument("--debug-draw-all", action="store_true", help="Draw all detections in debug image (incl. events/flows)")

    return p


def main() -> None:
    args = build_parser().parse_args()

    assign_text(
        image_path=args.image,
        yolox_path=args.det,
        ocr_path=args.ocr,
        out_path=args.out,
        debug_image_path=(args.debug_image if args.debug_image else None),
        overlap_thr=args.overlap_thr,
        min_ocr_conf=args.min_ocr_conf,
        shape_pad_px=args.shape_pad_px,
        attach_events=bool(args.attach_events),
        event_radius_px=args.event_radius_px,
        event_radius_k=args.event_radius_k,
        attach_seqflow=bool(args.attach_seqflow),
        min_ocr_conf_seqflow=args.min_ocr_conf_seqflow,
        seqflow_radius_px=args.seqflow_radius_px,
        seqflow_max_thickness=args.seqflow_max_thickness,
        seqflow_max_area_ratio=args.seqflow_max_area_ratio,
        font_path=(args.font if args.font else None),
        debug_draw_seqflow=bool(args.debug_draw_seqflow),
        debug_draw_all=bool(args.debug_draw_all),
    )


if __name__ == "__main__":
    main()
