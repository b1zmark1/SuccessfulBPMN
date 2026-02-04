from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.w() * self.h()

    def cx(self) -> float:
        return self.x1 + self.w() * 0.5

    def cy(self) -> float:
        return self.y1 + self.h() * 0.5

    def diag(self) -> float:
        return math.hypot(self.w(), self.h())


@dataclass(frozen=True)
class ElementDet:
    det_id: int
    class_name: str
    bbox: BBox
    score: float
    source: str


@dataclass(frozen=True)
class TextBlock:
    block_id: int
    bbox: BBox
    text: str
    confidence: float


@dataclass(frozen=True)
class AssignConfig:
    # detections filtering
    min_node_det_score: float = 0.40
    min_edge_det_score: float = 0.40  # CHANGED: was 0.45 to allow low-score long/diagonal flows

    # node accept thresholds
    strict_node_accept_score: float = 0.85
    loose_node_accept_score: float = 0.60

    # strict: require "text inside shape"
    strict_min_block_in_elem: float = 0.40

    # loose: allow "shape inside text" (gateway label bigger than diamond)
    loose_min_overlap: float = 0.22

    # distance gating for loose classes (relative to element diag)
    loose_max_dist_k: float = 1.40

    # weights
    w_overlap: float = 1.25
    w_iou: float = 0.45
    w_center_inside: float = 0.35
    w_dist: float = 0.75
    w_conf: float = 0.20

    # edge text assignment (sequence_flow labels)
    enable_edge_text: bool = True
    min_edge_ocr_conf: float = 0.60
    min_edge_text_len: int = 2

    # CHANGED: hard distance threshold (px), no "radius_k"
    edge_max_dist_px: int = 45
    edge_min_aspect: float = 3.0

    # merge OCR blocks (same line, close gap)
    merge_blocks: bool = True
    merge_max_gap_px: int = 26
    merge_min_y_overlap: float = 0.55
    merge_max_center_dy_px: int = 8

    edge_classes: Tuple[str, ...] = ("sequence_flow",)


_EDGE_TRIM_RE = re.compile(r"^[\s\|\)\(\]\[>\-–—_`\\]+|[\s\|\)\(\]\[<\-–—_`\\]+$")
_ONLY_SMALL_GARBAGE_RE = re.compile(r"^[\s\|\)\(\]\[>\-–—_`\\<>]+$")
_CYR_ALNUM_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")


STRICT_NODE_CLASSES = {
    "task",
    "subprocess",
    "data_object",
    "text_annotation",
}

LOOSE_NODE_CLASSES = {
    "gateway_exclusive",
    "gateway_parallel",
    "gateway_inclusive",
    "start_event",
    "end_event",
    "intermediate_event",
    "pool",
    "lane",
}


def _bbox_from_xyxy(xyxy: Any) -> Optional[BBox]:
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _intersection_area(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _iou(a: BBox, b: BBox) -> float:
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = a.area() + b.area() - inter
    return (inter / union) if union > 0 else 0.0


def _center_inside(inner: BBox, outer: BBox) -> bool:
    cx = inner.cx()
    cy = inner.cy()
    return (outer.x1 <= cx <= outer.x2) and (outer.y1 <= cy <= outer.y2)


def _dist_norm(a: BBox, b: BBox) -> float:
    d = math.hypot(a.cx() - b.cx(), a.cy() - b.cy())
    denom = max(1.0, a.diag())
    return float(d / denom)


def _rect_dist_point(box: BBox, px: float, py: float) -> float:
    dx = 0.0
    if px < box.x1:
        dx = box.x1 - px
    elif px > box.x2:
        dx = px - box.x2

    dy = 0.0
    if py < box.y1:
        dy = box.y1 - py
    elif py > box.y2:
        dy = py - box.y2

    return math.hypot(dx, dy)


def clean_ocr_text(text: str) -> str:
    t = (text or "").replace("\r", "").replace("\f", "").strip()
    if not t:
        return ""

    lines = [ln.strip() for ln in t.split("\n")]
    lines2: List[str] = []
    for ln in lines:
        ln = _EDGE_TRIM_RE.sub("", ln).strip()
        if not ln:
            continue
        if _ONLY_SMALL_GARBAGE_RE.fullmatch(ln):
            continue
        alnum = len(_CYR_ALNUM_RE.findall(ln))
        if alnum <= 1 and len(ln) <= 3:
            continue
        lines2.append(ln)

    out = "\n".join(lines2).strip()
    if not out:
        return ""
    if len(_CYR_ALNUM_RE.findall(out)) <= 1:
        return ""

    out_lines = []
    for ln in out.split("\n"):
        ln = re.sub(r"[ \t]{2,}", " ", ln).strip()
        out_lines.append(ln)
    return "\n".join(out_lines).strip()


def _merge_text_inline(left: str, right: str) -> str:
    """
    Merge two texts that are on the same visual line.
    Fixes duplicates like: "Заявка принята в" + "в работу" => "Заявка принята в работу"
    """
    l = clean_ocr_text(left).replace("\n", " ").strip()
    r = clean_ocr_text(right).replace("\n", " ").strip()
    if not l:
        return r
    if not r:
        return l

    lt = l.split()
    rt = r.split()
    if lt and rt:
        if lt[-1].lower() == rt[0].lower():
            rt = rt[1:]
    if not rt:
        return " ".join(lt).strip()

    return (" ".join(lt) + " " + " ".join(rt)).strip()


def load_elements_from_ensemble(ensemble_json: Dict[str, Any], cfg: AssignConfig) -> List[ElementDet]:
    dets = ensemble_json.get("detections", [])
    if not isinstance(dets, list):
        raise ValueError("ensemble_json['detections'] must be a list")

    out: List[ElementDet] = []
    next_id = 1
    for d in dets:
        if not isinstance(d, dict):
            next_id += 1
            continue

        class_name = str(d.get("class_name", "")).strip()
        if not class_name:
            next_id += 1
            continue

        if class_name == "text":
            next_id += 1
            continue

        bbox = _bbox_from_xyxy(d.get("bbox_xyxy") or d.get("bbox"))
        if bbox is None:
            next_id += 1
            continue

        try:
            score = float(d.get("score", 0.0))
        except Exception:
            score = 0.0

        out.append(
            ElementDet(
                det_id=next_id,
                class_name=class_name,
                bbox=bbox,
                score=score,
                source=str(d.get("source", "unknown")),
            )
        )
        next_id += 1

    return out


def load_text_blocks_with_ocr(text_blocks_json: Dict[str, Any], ocr_json: Dict[str, Any], cfg: AssignConfig) -> List[TextBlock]:
    blocks = text_blocks_json.get("blocks", [])
    if not isinstance(blocks, list):
        raise ValueError("text_blocks_json['blocks'] must be a list")

    ocr_blocks = ocr_json.get("blocks", [])
    if not isinstance(ocr_blocks, list):
        raise ValueError("ocr_json['blocks'] must be a list")

    ocr_by_id: Dict[int, Dict[str, Any]] = {}
    for ob in ocr_blocks:
        if not isinstance(ob, dict):
            continue
        try:
            bid = int(ob.get("block_id"))
        except Exception:
            continue
        ocr_by_id[bid] = ob

    def bbox_any(d: Dict[str, Any]) -> Optional[BBox]:
        bb = d.get("bbox_xyxy") or d.get("bbox")
        return _bbox_from_xyxy(bb) if bb is not None else None

    out: List[TextBlock] = []

    for b in blocks:
        if not isinstance(b, dict):
            continue
        try:
            bid = int(b.get("block_id"))
        except Exception:
            continue
        bb = bbox_any(b)
        if bb is None:
            continue

        o = ocr_by_id.get(bid, {})
        raw_text = str(o.get("text") or b.get("text") or "")
        text = clean_ocr_text(raw_text)
        if not text:
            continue

        try:
            conf = float(o.get("confidence", b.get("confidence", 0.0)) or 0.0)
        except Exception:
            conf = 0.0

        out.append(TextBlock(block_id=bid, bbox=bb, text=text, confidence=float(conf)))

    used_ids = {t.block_id for t in out}

    for ob in ocr_blocks:
        if not isinstance(ob, dict):
            continue
        try:
            bid = int(ob.get("block_id"))
        except Exception:
            continue
        if bid in used_ids:
            continue

        bb = bbox_any(ob)
        if bb is None:
            continue

        text = clean_ocr_text(str(ob.get("text") or ""))
        if not text:
            continue

        try:
            conf = float(ob.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0

        out.append(TextBlock(block_id=bid, bbox=bb, text=text, confidence=float(conf)))

    if cfg.merge_blocks:
        out = _merge_neighbor_blocks(out, cfg)

    return out


def _y_overlap_ratio(a: BBox, b: BBox) -> float:
    inter = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    denom = max(1.0, min(a.h(), b.h()))
    return float(inter / denom)


def _merge_neighbor_blocks(blocks: List[TextBlock], cfg: AssignConfig) -> List[TextBlock]:
    if not blocks:
        return []

    blocks_sorted = sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1))
    used = [False] * len(blocks_sorted)
    merged: List[TextBlock] = []

    for i, b in enumerate(blocks_sorted):
        if used[i]:
            continue

        cur = b
        used[i] = True

        while True:
            best_j = None
            best_gap = None

            cy = cur.bbox.cy()
            for j in range(len(blocks_sorted)):
                if used[j]:
                    continue
                bj = blocks_sorted[j]

                if bj.bbox.x1 < cur.bbox.x2:
                    continue

                yov = _y_overlap_ratio(cur.bbox, bj.bbox)
                if yov < float(cfg.merge_min_y_overlap):
                    continue

                dy = abs(bj.bbox.cy() - cy)
                if dy > float(cfg.merge_max_center_dy_px):
                    continue

                gap = bj.bbox.x1 - cur.bbox.x2
                if gap < 0 or gap > float(cfg.merge_max_gap_px):
                    continue

                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_j = j

            if best_j is None:
                break

            bj = blocks_sorted[best_j]
            used[best_j] = True

            nb = BBox(
                x1=min(cur.bbox.x1, bj.bbox.x1),
                y1=min(cur.bbox.y1, bj.bbox.y1),
                x2=max(cur.bbox.x2, bj.bbox.x2),
                y2=max(cur.bbox.y2, bj.bbox.y2),
            )

            nt = _merge_text_inline(cur.text, bj.text)

            a1 = max(1.0, cur.bbox.area())
            a2 = max(1.0, bj.bbox.area())
            nc = float((cur.confidence * a1 + bj.confidence * a2) / (a1 + a2))

            cur = TextBlock(block_id=cur.block_id, bbox=nb, text=nt, confidence=nc)

        merged.append(cur)

    return merged


def _combine_blocks_text(blocks: List[TextBlock]) -> Tuple[str, float, List[int]]:
    if not blocks:
        return "", 0.0, []
    blocks_sorted = sorted(blocks, key=lambda b: (b.bbox.y1, b.bbox.x1))

    texts: List[str] = []
    w_sum = 0.0
    c_sum = 0.0
    ids: List[int] = []

    for b in blocks_sorted:
        t = clean_ocr_text(b.text)
        if not t:
            continue
        texts.append(t)
        ids.append(int(b.block_id))
        w = float(max(1.0, b.bbox.area()))
        w_sum += w
        c_sum += float(b.confidence) * w

    out_text = "\n".join(texts).strip()
    out_conf = (c_sum / w_sum) if w_sum > 0 else 0.0
    return out_text, float(out_conf), ids


def _score_pair(elem: ElementDet, blk: TextBlock, cfg: AssignConfig) -> Tuple[float, Dict[str, float]]:
    inter = _intersection_area(blk.bbox, elem.bbox)
    if inter <= 0.0:
        block_in_elem = 0.0
        elem_in_block = 0.0
    else:
        block_in_elem = float(inter / max(1e-9, blk.bbox.area()))
        elem_in_block = float(inter / max(1e-9, elem.bbox.area()))

    overlap = max(block_in_elem, elem_in_block)
    iou = _iou(blk.bbox, elem.bbox)
    center_inside = 1.0 if (_center_inside(blk.bbox, elem.bbox) or _center_inside(elem.bbox, blk.bbox)) else 0.0
    dist = _dist_norm(elem.bbox, blk.bbox)
    dist_term = 1.0 / (1.0 + dist)
    conf = float(max(0.0, min(1.0, blk.confidence)))

    score = (
        cfg.w_overlap * overlap
        + cfg.w_iou * iou
        + cfg.w_center_inside * center_inside
        + cfg.w_dist * dist_term
        + cfg.w_conf * conf
    )

    dbg = {
        "overlap": float(overlap),
        "block_in_elem": float(block_in_elem),
        "elem_in_block": float(elem_in_block),
        "iou": float(iou),
        "center_inside": float(center_inside),
        "dist": float(dist),
        "dist_term": float(dist_term),
        "conf": float(conf),
    }
    return float(score), dbg


def _eligible_for_node(elem: ElementDet, cfg: AssignConfig) -> bool:
    if elem.class_name in set(cfg.edge_classes):
        return False
    return elem.score >= float(cfg.min_node_det_score)


def _eligible_for_edge(elem: ElementDet, cfg: AssignConfig) -> bool:
    if elem.class_name not in set(cfg.edge_classes):
        return False
    return elem.score >= float(cfg.min_edge_det_score)


def _assign_nodes(
    elements: List[ElementDet],
    blocks: List[TextBlock],
    cfg: AssignConfig,
) -> Tuple[Dict[int, List[TextBlock]], Dict[int, float], Dict[int, int], set[int]]:
    node_elems = [e for e in elements if _eligible_for_node(e, cfg)]

    assigned_blocks_by_det: Dict[int, List[TextBlock]] = {e.det_id: [] for e in elements}
    best_score_by_block: Dict[int, float] = {}
    best_det_by_block: Dict[int, int] = {}

    for b in blocks:
        if not clean_ocr_text(b.text):
            continue

        best_det: Optional[int] = None
        best_score = -1.0
        best_dbg: Optional[Dict[str, float]] = None

        for e in node_elems:
            score, dbg = _score_pair(e, b, cfg)

            if e.class_name in STRICT_NODE_CLASSES:
                if dbg["block_in_elem"] < float(cfg.strict_min_block_in_elem):
                    continue
                accept_thr = float(cfg.strict_node_accept_score)
            elif e.class_name in LOOSE_NODE_CLASSES:
                if dbg["overlap"] < float(cfg.loose_min_overlap) and dbg["dist"] > float(cfg.loose_max_dist_k):
                    continue
                accept_thr = float(cfg.loose_node_accept_score)
            else:
                if dbg["overlap"] < float(cfg.loose_min_overlap) and dbg["dist"] > float(cfg.loose_max_dist_k):
                    continue
                accept_thr = float(cfg.loose_node_accept_score)

            if score > best_score:
                best_score = float(score)
                best_det = int(e.det_id)
                best_dbg = dbg

        if best_det is None or best_dbg is None:
            continue

        chosen = next((x for x in node_elems if x.det_id == best_det), None)
        if chosen is None:
            continue

        if chosen.class_name in STRICT_NODE_CLASSES:
            accept_thr = float(cfg.strict_node_accept_score)
        else:
            accept_thr = float(cfg.loose_node_accept_score)

        if best_score < accept_thr:
            continue

        prev = best_score_by_block.get(int(b.block_id))
        if prev is None or best_score > prev:
            best_score_by_block[int(b.block_id)] = float(best_score)
            best_det_by_block[int(b.block_id)] = int(best_det)

    used_block_ids: set[int] = set()
    for b in blocks:
        det_id = best_det_by_block.get(int(b.block_id))
        if det_id is None:
            continue
        assigned_blocks_by_det[det_id].append(b)
        used_block_ids.add(int(b.block_id))

    return assigned_blocks_by_det, best_score_by_block, best_det_by_block, used_block_ids


def _edge_is_line_like(b: BBox, cfg: AssignConfig) -> bool:
    w = max(1.0, b.w())
    h = max(1.0, b.h())
    aspect = max(w / h, h / w)
    return aspect >= float(cfg.edge_min_aspect)


def _assign_edges(
    elements: List[ElementDet],
    blocks: List[TextBlock],
    used_block_ids: set[int],
    cfg: AssignConfig,
) -> Tuple[Dict[int, List[TextBlock]], set[int]]:
    if not cfg.enable_edge_text:
        return {e.det_id: [] for e in elements}, set()

    edge_elems = [e for e in elements if _eligible_for_edge(e, cfg)]

    assigned: Dict[int, List[TextBlock]] = {e.det_id: [] for e in elements}
    used_by_edges: set[int] = set()

    for b in blocks:
        bid = int(b.block_id)
        if bid in used_block_ids:
            continue

        text = clean_ocr_text(b.text).replace("\n", " ").strip()
        if not text:
            continue
        if len(text) < int(cfg.min_edge_text_len):
            continue
        if float(b.confidence) < float(cfg.min_edge_ocr_conf):
            continue

        bx, by = b.bbox.cx(), b.bbox.cy()

        best_det = None
        best_dist = None

        for e in edge_elems:
            if not _edge_is_line_like(e.bbox, cfg):
                continue

            dist = _rect_dist_point(e.bbox, bx, by)
            if dist > float(cfg.edge_max_dist_px):
                continue

            if best_dist is None or dist < best_dist:
                best_dist = float(dist)
                best_det = int(e.det_id)

        if best_det is None:
            continue

        assigned[best_det].append(b)
        used_by_edges.add(bid)

    return assigned, used_by_edges


def assign_blocks(
    ensemble_json: Dict[str, Any],
    text_blocks_json: Dict[str, Any],
    ocr_json: Dict[str, Any],
    cfg: Optional[AssignConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = AssignConfig()

    elements = load_elements_from_ensemble(ensemble_json, cfg=cfg)
    blocks = load_text_blocks_with_ocr(text_blocks_json=text_blocks_json, ocr_json=ocr_json, cfg=cfg)

    assigned_nodes_by_det, best_score_by_block, _, used_block_ids = _assign_nodes(elements, blocks, cfg)
    assigned_edges_by_det, used_edge_block_ids = _assign_edges(elements, blocks, used_block_ids, cfg)

    assigned_blocks_by_det: Dict[int, List[TextBlock]] = {e.det_id: [] for e in elements}
    for e in elements:
        assigned_blocks_by_det[e.det_id].extend(assigned_nodes_by_det.get(e.det_id, []))
        assigned_blocks_by_det[e.det_id].extend(assigned_edges_by_det.get(e.det_id, []))

    used_all = set(used_block_ids) | set(used_edge_block_ids)

    out_elements: List[Dict[str, Any]] = []
    labeled_count = 0

    for e in elements:
        blks = assigned_blocks_by_det.get(e.det_id, [])
        text, conf, block_ids = _combine_blocks_text(blks)

        match_score = None
        if block_ids:
            labeled_count += 1
            match_score = float(max(best_score_by_block.get(i, 0.0) for i in block_ids))

        best_block_id = int(block_ids[0]) if block_ids else None

        out_elements.append(
            {
                "det_id": int(e.det_id),
                "class_name": str(e.class_name),
                "bbox_xyxy": [float(e.bbox.x1), float(e.bbox.y1), float(e.bbox.x2), float(e.bbox.y2)],
                "det_score": float(e.score),
                "source": str(e.source),
                "text": (text if text else None),
                "block_id": (best_block_id if best_block_id is not None else None),
                "block_ids": [int(x) for x in block_ids],
                "ocr_confidence": float(conf),
                "match_score": (match_score if match_score is not None else None),
            }
        )

    unused_blocks = [
        {
            "block_id": int(b.block_id),
            "bbox_xyxy": [float(b.bbox.x1), float(b.bbox.y1), float(b.bbox.x2), float(b.bbox.y2)],
            "text": clean_ocr_text(b.text),
            "confidence": float(b.confidence),
        }
        for b in blocks
        if int(b.block_id) not in used_all and clean_ocr_text(b.text)
    ]

    report = {
        "mode": "label_assign",
        "elements_total": int(len(elements)),
        "nodes_considered": int(sum(1 for e in elements if _eligible_for_node(e, cfg))),
        "edges_considered": int(sum(1 for e in elements if _eligible_for_edge(e, cfg))),
        "blocks_total": int(len(blocks)),
        "blocks_used_nodes": int(len(used_block_ids)),
        "blocks_used_edges": int(len(used_edge_block_ids)),
        "elements_labeled": int(labeled_count),
        "unused_blocks": int(len(unused_blocks)),
        "cfg": asdict(cfg),
    }

    return {
        "elements": out_elements,
        "unused_blocks": unused_blocks,
        "report": report,
    }
