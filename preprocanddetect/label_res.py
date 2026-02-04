from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import label_assign as la  # noqa: E402


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


def _assign_via_label_assign(
    ensemble_json: Dict[str, Any],
    text_blocks_json: Dict[str, Any],
    ocr_json: Dict[str, Any],
) -> Dict[str, Any]:
    fn = getattr(la, "assign_blocks", None)
    if fn is None or not callable(fn):
        raise RuntimeError("label_assign.assign_blocks() is missing")
    return fn(ensemble_json=ensemble_json, text_blocks_json=text_blocks_json, ocr_json=ocr_json)


def _draw_overlay(
    img_bgr: np.ndarray,
    dets: List[Det],
    ocr_blocks: List[OcrBlock],
    assigned: List[Assigned],
    out_path: Path,
) -> None:
    vis = img_bgr.copy()

    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
        if d.class_name in EDGE_CLASSES:
            color = (180, 180, 180)
            th = 1
        elif d.class_name in NODE_CLASSES:
            color = (0, 140, 255)
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
    ap.add_argument("--out-ensemble", type=str, default="")
    ap.add_argument("--min-text-conf", type=float, default=0.4)
    return ap.parse_args()


def _merge_labeled_into_ensemble(
    ensemble_json: Dict[str, Any],
    assigned: List[Assigned],
    unused_blocks: List[Dict[str, Any]],
    min_text_conf: float,
) -> Dict[str, Any]:
    merged = json.loads(json.dumps(ensemble_json, ensure_ascii=False))
    dets = merged.get("detections", [])
    if not isinstance(dets, list):
        raise ValueError("ensemble_json['detections'] must be a list")

    assigned_by_det_id: Dict[int, Assigned] = {int(a.det_id): a for a in assigned}

    labeled_nodes = 0
    flows_kept = 0
    assigned_with_text_or_block = 0

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

        if a.block_id is not None or txt:
            assigned_with_text_or_block += 1
        if txt and cls in NODE_CLASSES:
            labeled_nodes += 1

    appended_text = 0
    for ub in unused_blocks:
        if not isinstance(ub, dict):
            continue
        txt = _clean_text(str(ub.get("text") or ""))
        if not txt:
            continue
        try:
            conf = float(ub.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf < float(min_text_conf):
            continue

        bb = ub.get("bbox_xyxy") or ub.get("bbox")
        if not isinstance(bb, list) or len(bb) != 4:
            continue

        bid = ub.get("block_id")
        try:
            bid_i = int(bid) if bid is not None else None
        except Exception:
            bid_i = None

        dets.append(
            {
                "class_name": "text",
                "score": float(conf),
                "bbox_xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                "source": "ocr",
                "text": txt,
                "text_block_ids": ([int(bid_i)] if bid_i is not None else []),
                "text_conf": float(conf),
            }
        )
        appended_text += 1

    meta = merged.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta["text_merge_source"] = "label_res:labeled+ensemble"
    meta["text_nodes_labeled"] = int(labeled_nodes)
    meta["sequence_flows_kept"] = int(flows_kept)
    meta["assigned_nodes_total"] = int(len(assigned))
    meta["assigned_nodes_matched"] = int(assigned_with_text_or_block)
    meta["appended_text_detections"] = int(appended_text)
    merged["meta"] = meta
    merged["detections"] = dets
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

    la_res = _assign_via_label_assign(ensemble_json, text_blocks_json, ocr_json)

    elems = la_res.get("elements")
    unused_blocks = la_res.get("unused_blocks") if isinstance(la_res.get("unused_blocks"), list) else []
    report_obj = la_res.get("report") if isinstance(la_res.get("report"), dict) else {}

    assigned: List[Assigned] = []
    if isinstance(elems, list):
        for e in elems:
            if not isinstance(e, dict):
                continue
            det_id = e.get("det_id")
            try:
                det_id_i = int(det_id)
            except Exception:
                continue
            cls = str(e.get("class_name") or "")
            bb = _as_bbox_xyxy(e.get("bbox_xyxy") or e.get("bbox"))
            if bb is None:
                continue

            text = _clean_text(str(e.get("text") or ""))
            blk_id = e.get("block_id")
            blk_id_i = None
            if blk_id is not None:
                try:
                    blk_id_i = int(blk_id)
                except Exception:
                    blk_id_i = None

            ocr_conf = e.get("ocr_confidence")
            try:
                ocr_conf_f = float(ocr_conf) if ocr_conf is not None else None
            except Exception:
                ocr_conf_f = None

            match_score = e.get("match_score")
            try:
                match_score_f = float(match_score) if match_score is not None else None
            except Exception:
                match_score_f = None

            assigned.append(
                Assigned(
                    det_id=det_id_i,
                    class_name=cls,
                    det_score=float(e.get("det_score", 0.0) or 0.0),
                    bbox=bb,
                    text=text,
                    block_id=blk_id_i,
                    ocr_confidence=ocr_conf_f,
                    match_score=match_score_f,
                )
            )

    labeled = {
        "coord_space": str(ensemble_json.get("coord_space") or "model"),
        "image": ensemble_json.get("image", {}),
        "assigned_nodes": [asdict(a) for a in assigned],
        "unused_blocks": unused_blocks,
        "report": {"mode": "label_assign", "detail": report_obj},
        "raw_label_assign": la_res,
    }
    _write_json(out_dir / "labeled.json", labeled)

    overlay_path = out_dir / "overlay_labeled.png"
    _draw_overlay(img_bgr=img, dets=dets, ocr_blocks=ocr_blocks, assigned=assigned, out_path=overlay_path)

    merged_ensemble = _merge_labeled_into_ensemble(
        ensemble_json=ensemble_json,
        assigned=assigned,
        unused_blocks=unused_blocks,
        min_text_conf=float(args.min_text_conf),
    )
    if args.out_ensemble:
        merged_out_path = Path(args.out_ensemble).expanduser().resolve()
    else:
        merged_out_path = out_dir / f"{ensemble_p.stem}_merged_labeled.json"
    _write_json(merged_out_path, merged_ensemble)

    _write_json(out_dir / "report.json", labeled["report"])

    print(f"[label_res] saved: {out_dir / 'labeled.json'}")
    print(f"[label_res] saved: {overlay_path}")
    print(f"[label_res] saved: {out_dir / 'report.json'}")
    print(f"[label_res] saved: {merged_out_path}")


if __name__ == "__main__":
    main()
