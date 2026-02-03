from __future__ import annotations

# CHANGED:
# - Вместо "from label_assign import ..." используется "import label_assign as la"
# - Скрипт не падает, если la.assign_blocks отсутствует: есть fallback assignment
# - Добавлен вывод overlay + report.json + labeled.json

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ВАЖНО: чтобы `import label_assign` работал при запуске:
#   python preprocanddetect/label_res.py ...
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import label_assign as la  # noqa: E402  (import after sys.path tweak)


BBox = Tuple[float, float, float, float]


NODE_CLASSES = {
    "start_event",
    "intermediate_event",
    "end_event",
    "task",
    "gateway_exclusive",
    "gateway_parallel",
    "gateway_inclusive",
    "subprocess",
    "pool",
    "lane",
    "data_object",
    "text_annotation",
}

EDGE_CLASSES = {
    "sequence_flow",
}


@dataclass(frozen=True)
class OcrBlock:
    block_id: int
    bbox: BBox
    text: str
    confidence: float
    confidence_available: bool


@dataclass
class Det:
    det_id: int
    class_name: str
    score: float
    bbox: BBox
    source: str


@dataclass
class Assigned:
    det_id: int
    class_name: str
    det_score: float
    bbox: BBox
    text: str
    block_id: Optional[int]
    ocr_confidence: Optional[float]
    match_score: Optional[float]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_bbox_xyxy(v: Any) -> Optional[BBox]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in v]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox_area(b: BBox) -> float:
    return max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))


def _bbox_center(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _iou(a: BBox, b: BBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = _bbox_area(a) + _bbox_area(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _inside_ratio(inner: BBox, outer: BBox) -> float:
    # какая доля inner лежит внутри outer (по площади пересечения / площади inner)
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a = _bbox_area(inner)
    return float(inter / a) if a > 0 else 0.0


def _dist_norm(a: BBox, b: BBox) -> float:
    # нормированная дистанция между центрами (делим на диагональ объединённого bbox)
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    dx = ax - bx
    dy = ay - by
    d = float((dx * dx + dy * dy) ** 0.5)
    ux1 = min(a[0], b[0])
    uy1 = min(a[1], b[1])
    ux2 = max(a[2], b[2])
    uy2 = max(a[3], b[3])
    diag = float(((ux2 - ux1) ** 2 + (uy2 - uy1) ** 2) ** 0.5)
    if diag <= 1e-6:
        return 0.0
    return d / diag


def _clean_text(t: str) -> str:
    if t is None:
        return ""
    t = t.replace("\r", "").replace("\f", "")
    lines = [ln.strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _load_ocr_blocks(ocr_json: Dict[str, Any]) -> List[OcrBlock]:
    blocks_raw = ocr_json.get("blocks", [])
    if not isinstance(blocks_raw, list):
        return []

    out: List[OcrBlock] = []
    for b in blocks_raw:
        if not isinstance(b, dict):
            continue
        bid = b.get("block_id")
        try:
            bid_i = int(bid)
        except Exception:
            continue

        bb = _as_bbox_xyxy(b.get("bbox_xyxy") or b.get("bbox"))
        if bb is None:
            continue

        text = _clean_text(str(b.get("text") or ""))
        conf = b.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        conf_av = bool(b.get("confidence_available", True))

        out.append(
            OcrBlock(
                block_id=bid_i,
                bbox=bb,
                text=text,
                confidence=conf_f,
                confidence_available=conf_av,
            )
        )
    return out


def _load_detections(ensemble_json: Dict[str, Any]) -> List[Det]:
    dets_raw = ensemble_json.get("detections", [])
    if not isinstance(dets_raw, list):
        return []

    out: List[Det] = []
    did = 1
    for d in dets_raw:
        if not isinstance(d, dict):
            continue
        cls = str(d.get("class_name") or "")
        if not cls:
            continue
        bb = _as_bbox_xyxy(d.get("bbox_xyxy") or d.get("bbox"))
        if bb is None:
            continue
        sc = d.get("score", 0.0)
        try:
            sc_f = float(sc)
        except Exception:
            sc_f = 0.0
        src = str(d.get("source") or "")
        out.append(Det(det_id=did, class_name=cls, score=sc_f, bbox=bb, source=src))
        did += 1
    return out


def _pair_score(det: Det, blk: OcrBlock) -> float:
    """
    Скоринг кандидата соответствия текста и элемента.
    Идея: текст обычно внутри shape (inside_ratio высокий) и близко к центру.
    """
    iou = _iou(det.bbox, blk.bbox)
    inside = _inside_ratio(blk.bbox, det.bbox)
    dist = _dist_norm(det.bbox, blk.bbox)

    # доверие OCR — вторично, но пусть влияет
    ocr_conf = float(max(0.0, min(1.0, blk.confidence)))

    # penalty за далеко
    dist_term = max(0.0, 1.0 - dist)

    # текст-подписи рядом с задачей иногда частично вне bbox => разрешаем не только inside
    score = 2.2 * inside + 1.0 * iou + 0.6 * dist_term + 0.35 * ocr_conf

    # пустой/мусорный текст резко вниз
    if not blk.text.strip():
        score -= 2.0

    # короткие "Да/Нет" полезны, но не для task (часто label потока у gateway)
    # поэтому лёгкий штраф для task, чтобы не перехватывали "Да"
    if det.class_name == "task" and len(blk.text.strip()) <= 3:
        score -= 0.5

    return float(score)


def _assign_fallback(
    dets: List[Det],
    blocks: List[OcrBlock],
    min_match_score: float = 0.85,
) -> Tuple[List[Assigned], Dict[str, Any]]:
    """
    Greedy matching:
    1) считаем все пары (det, block) с score
    2) сортируем по score desc
    3) назначаем 1 block -> 1 det
    """
    nodes = [d for d in dets if d.class_name in NODE_CLASSES]
    edges = [d for d in dets if d.class_name in EDGE_CLASSES]

    pairs: List[Tuple[float, int, int]] = []
    for di, d in enumerate(nodes):
        for bi, b in enumerate(blocks):
            # грубый фильтр кандидатов:
            # - либо IoU хоть какой-то
            # - либо центр блока попадает в bbox детекта
            iou = _iou(d.bbox, b.bbox)
            bx, by = _bbox_center(b.bbox)
            in_bbox = (d.bbox[0] <= bx <= d.bbox[2]) and (d.bbox[1] <= by <= d.bbox[3])
            if not in_bbox and iou < 0.01:
                continue

            s = _pair_score(d, b)
            pairs.append((s, di, bi))

    pairs.sort(key=lambda x: x[0], reverse=True)

    det_taken = set()
    blk_taken = set()

    assigned_by_det: Dict[int, Tuple[int, float]] = {}

    for s, di, bi in pairs:
        if s < float(min_match_score):
            break
        if di in det_taken or bi in blk_taken:
            continue
        det_taken.add(di)
        blk_taken.add(bi)
        assigned_by_det[di] = (bi, s)

    out: List[Assigned] = []
    for di, d in enumerate(nodes):
        if di in assigned_by_det:
            bi, s = assigned_by_det[di]
            b = blocks[bi]
            out.append(
                Assigned(
                    det_id=d.det_id,
                    class_name=d.class_name,
                    det_score=float(d.score),
                    bbox=d.bbox,
                    text=b.text,
                    block_id=int(b.block_id),
                    ocr_confidence=float(b.confidence),
                    match_score=float(s),
                )
            )
        else:
            out.append(
                Assigned(
                    det_id=d.det_id,
                    class_name=d.class_name,
                    det_score=float(d.score),
                    bbox=d.bbox,
                    text="",
                    block_id=None,
                    ocr_confidence=None,
                    match_score=None,
                )
            )

    # "unused" blocks могут быть labels для sequence_flow или for gateway branches — оставим в отчёте
    unused_blocks = [asdict(b) for i, b in enumerate(blocks) if i not in blk_taken]

    report = {
        "mode": "fallback",
        "min_match_score": float(min_match_score),
        "nodes_total": int(len(nodes)),
        "edges_total": int(len(edges)),
        "ocr_blocks_total": int(len(blocks)),
        "nodes_labeled": int(sum(1 for a in out if a.block_id is not None and a.text.strip())),
        "ocr_blocks_unused": int(len(unused_blocks)),
        "unused_blocks": unused_blocks[:50],  # чтобы не раздувать
    }
    return out, report


def _assign_via_label_assign_if_possible(
    ensemble_json: Dict[str, Any],
    text_blocks_json: Dict[str, Any],
    ocr_json: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Если в label_assign.py реально есть assign_blocks(), пробуем использовать.
    Иначе возвращаем None.
    """
    fn = getattr(la, "assign_blocks", None)
    if fn is None or not callable(fn):
        return None, {"mode": "label_assign_missing", "available_symbols": [x for x in dir(la) if "assign" in x]}

    # Не навязываю формат: передаём сырой json как есть.
    # Если твой assign_blocks ожидает другое — он сам выбросит понятную ошибку.
    try:
        res = fn(
            ensemble_json=ensemble_json,
            text_blocks_json=text_blocks_json,
            ocr_json=ocr_json,
        )
        return res, {"mode": "label_assign", "ok": True}
    except TypeError:
        # если у тебя сигнатура другая, попробуем common варианты
        try:
            res = fn(ensemble_json, text_blocks_json, ocr_json)
            return res, {"mode": "label_assign", "ok": True, "called": "positional"}
        except Exception as e:
            return None, {"mode": "label_assign_failed", "error": repr(e)}
    except Exception as e:
        return None, {"mode": "label_assign_failed", "error": repr(e)}


def _draw_overlay(
    img_bgr: np.ndarray,
    dets: List[Det],
    ocr_blocks: List[OcrBlock],
    assigned: List[Assigned],
    out_path: Path,
) -> None:
    vis = img_bgr.copy()

    # 1) рисуем элементы YOLOX
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
        if d.class_name in EDGE_CLASSES:
            color = (180, 180, 180)
            th = 1
        elif d.class_name in NODE_CLASSES:
            color = (0, 140, 255)  # nodes
            th = 2
        else:
            color = (120, 120, 255)
            th = 1

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, th)
        cv2.putText(
            vis,
            f"{d.class_name}:{d.det_id}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # 2) рисуем OCR blocks
    for b in ocr_blocks:
        x1, y1, x2, y2 = [int(round(v)) for v in b.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 1)
        cv2.putText(
            vis,
            f"blk:{b.block_id}",
            (x1, min(vis.shape[0] - 2, y2 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )

    # 3) линии соответствия
    block_by_id = {b.block_id: b for b in ocr_blocks}
    det_by_id = {d.det_id: d for d in dets}

    for a in assigned:
        if a.block_id is None:
            continue
        d = det_by_id.get(a.det_id)
        b = block_by_id.get(a.block_id)
        if d is None or b is None:
            continue
        dx, dy = _bbox_center(d.bbox)
        bx, by = _bbox_center(b.bbox)
        cv2.line(vis, (int(dx), int(dy)), (int(bx), int(by)), (0, 220, 220), 1)

        # маленькая подпись match_score
        if a.match_score is not None:
            mx = int((dx + bx) / 2.0)
            my = int((dy + by) / 2.0)
            cv2.putText(
                vis,
                f"{a.match_score:.2f}",
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 220),
                1,
                cv2.LINE_AA,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), vis)
    if not ok:
        raise RuntimeError(f"Failed to write overlay: {out_path}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--text-blocks", required=True)
    ap.add_argument("--ocr", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--min-match-score", type=float, default=0.85)
    ap.add_argument(
        "--out-ensemble",
        type=str,
        default="",
        help="Output path for auto-generated merged ensemble JSON. "
        "Default: <outdir>/<ensemble_stem>_merged_labeled.json",
    )

    return ap.parse_args()


def _merge_labeled_into_ensemble(
    ensemble_json: Dict[str, Any],
    assigned: List[Assigned],
) -> Dict[str, Any]:
    """
    Auto-generate ensemble with OCR text labels:
    - keeps all detections from YOLOX (including sequence_flow)
    - injects node text fields from labeled assignment
    """
    merged = json.loads(json.dumps(ensemble_json, ensure_ascii=False))
    dets = merged.get("detections", [])
    if not isinstance(dets, list):
        raise ValueError("ensemble_json['detections'] must be a list")

    assigned_by_det_id: Dict[int, Assigned] = {int(a.det_id): a for a in assigned}
    used_assigned: set[int] = set()
    labeled_nodes = 0
    flows_kept = 0

    # Primary mapping by det_id (1-based order from detections list).
    for i, d in enumerate(dets, start=1):
        if not isinstance(d, dict):
            continue
        cls = str(d.get("class_name") or "")
        if cls in EDGE_CLASSES:
            flows_kept += 1

        a = assigned_by_det_id.get(i)
        if a is None:
            continue
        if str(a.class_name) != cls:
            continue

        txt = _clean_text(a.text or "")
        d["text"] = txt if txt else None
        d["text_block_ids"] = [int(a.block_id)] if a.block_id is not None else []
        d["text_conf"] = float(a.ocr_confidence) if a.ocr_confidence is not None else 0.0
        d["match_score"] = float(a.match_score) if a.match_score is not None else None

        if txt:
            labeled_nodes += 1
        used_assigned.add(i)

    # Fallback for unmatched labels: class + IoU.
    for a in assigned:
        if int(a.det_id) in used_assigned:
            continue
        txt = _clean_text(a.text or "")
        if not txt:
            continue
        best_j = None
        best_iou = 0.0
        for j, d in enumerate(dets):
            if not isinstance(d, dict):
                continue
            if str(d.get("class_name") or "") != str(a.class_name):
                continue
            bb = _as_bbox_xyxy(d.get("bbox_xyxy") or d.get("bbox"))
            if bb is None:
                continue
            iou = _iou(a.bbox, bb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None and best_iou >= 0.5:
            d = dets[best_j]
            d["text"] = txt
            d["text_block_ids"] = [int(a.block_id)] if a.block_id is not None else []
            d["text_conf"] = float(a.ocr_confidence) if a.ocr_confidence is not None else 0.0
            d["match_score"] = float(a.match_score) if a.match_score is not None else None
            labeled_nodes += 1
            used_assigned.add(int(a.det_id))

    meta = merged.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["text_merge_source"] = "label_res:labeled+ensemble"
    meta["text_nodes_labeled"] = int(labeled_nodes)
    meta["sequence_flows_kept"] = int(flows_kept)
    meta["assigned_nodes_total"] = int(len(assigned))
    meta["assigned_nodes_matched"] = int(len(used_assigned))
    merged["meta"] = meta
    return merged


def main() -> None:
    args = _parse_args()

    ensemble_p = Path(args.ensemble).expanduser().resolve()
    text_blocks_p = Path(args.text_blocks).expanduser().resolve()
    ocr_p = Path(args.ocr).expanduser().resolve()
    image_p = Path(args.image).expanduser().resolve()
    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble_json = _read_json(ensemble_p)
    text_blocks_json = _read_json(text_blocks_p)
    ocr_json = _read_json(ocr_p)

    img = cv2.imread(str(image_p), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {image_p}")

    dets = _load_detections(ensemble_json)
    ocr_blocks = _load_ocr_blocks(ocr_json)

    # 1) пробуем la.assign_blocks (если есть)
    la_res, la_report = _assign_via_label_assign_if_possible(ensemble_json, text_blocks_json, ocr_json)

    assigned: List[Assigned]
    report: Dict[str, Any]

    if la_res is not None:
        # Если твой label_assign возвращает уже готовую структуру, сохраняем как есть.
        # Но для overlay нужен "assigned" список. Попробуем вытащить common формат.
        # Если формат другой — overlay будет только по dets/blocks, без линий match.
        assigned = []
        report = {"mode": "label_assign", "detail": la_report or {}}

        if isinstance(la_res, dict):
            # Ожидаем что-то вроде:
            # { "elements": [ { "det_id":..., "class_name":..., "bbox":..., "text":..., "block_id":... }, ... ] }
            elems = la_res.get("elements") or la_res.get("assigned") or la_res.get("nodes")
            if isinstance(elems, list):
                for e in elems:
                    if not isinstance(e, dict):
                        continue
                    det_id = e.get("det_id") or e.get("id")
                    try:
                        det_id_i = int(det_id)
                    except Exception:
                        continue
                    cls = str(e.get("class_name") or "")
                    bb = _as_bbox_xyxy(e.get("bbox_xyxy") or e.get("bbox"))
                    if bb is None:
                        # попробуем bbox детекта по det_id
                        d0 = next((d for d in dets if d.det_id == det_id_i), None)
                        if d0 is not None:
                            bb = d0.bbox
                        else:
                            continue
                    text = _clean_text(str(e.get("text") or ""))
                    blk_id = e.get("block_id")
                    blk_id_i = None
                    if blk_id is not None:
                        try:
                            blk_id_i = int(blk_id)
                        except Exception:
                            blk_id_i = None

                    assigned.append(
                        Assigned(
                            det_id=det_id_i,
                            class_name=cls,
                            det_score=float(e.get("det_score", 0.0) or 0.0),
                            bbox=bb,
                            text=text,
                            block_id=blk_id_i,
                            ocr_confidence=None,
                            match_score=None,
                        )
                    )

        _write_json(out_dir / "labeled.json", {"result": la_res, "report": report})
        print("[label_res] used label_assign.assign_blocks()")

    else:
        # 2) fallback
        assigned, report = _assign_fallback(
            dets=dets,
            blocks=ocr_blocks,
            min_match_score=float(args.min_match_score),
        )
        labeled = {
            "coord_space": str(ensemble_json.get("coord_space") or "model"),
            "image": ensemble_json.get("image", {}),
            "assigned_nodes": [asdict(a) for a in assigned],
            "report": report,
        }
        _write_json(out_dir / "labeled.json", labeled)
        print("[label_res] used fallback assignment (label_assign.assign_blocks missing or failed)")

    # overlay всегда полезен
    overlay_path = out_dir / "overlay_labeled.png"
    _draw_overlay(
        img_bgr=img,
        dets=dets,
        ocr_blocks=ocr_blocks,
        assigned=assigned if isinstance(assigned, list) else [],
        out_path=overlay_path,
    )

    _write_json(out_dir / "report.json", report)

    merged_ensemble = _merge_labeled_into_ensemble(
        ensemble_json=ensemble_json,
        assigned=(assigned if isinstance(assigned, list) else []),
    )
    if args.out_ensemble:
        merged_out_path = Path(args.out_ensemble).expanduser().resolve()
    else:
        merged_out_path = out_dir / f"{ensemble_p.stem}_merged_labeled.json"
    _write_json(merged_out_path, merged_ensemble)

    # короткая статистика
    nodes_total = sum(1 for d in dets if d.class_name in NODE_CLASSES)
    edges_total = sum(1 for d in dets if d.class_name in EDGE_CLASSES)
    labeled_nodes = 0
    if isinstance(assigned, list):
        labeled_nodes = sum(1 for a in assigned if (a.text or "").strip())

    print(
        f"[label_res] dets_total={len(dets)} nodes={nodes_total} edges={edges_total} "
        f"ocr_blocks={len(ocr_blocks)} labeled_nodes={labeled_nodes}"
    )
    print(f"[label_res] saved: {out_dir / 'labeled.json'}")
    print(f"[label_res] saved: {overlay_path}")
    print(f"[label_res] saved: {out_dir / 'report.json'}")
    print(f"[label_res] saved: {merged_out_path}")


if __name__ == "__main__":
    main()
