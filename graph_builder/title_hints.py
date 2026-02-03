from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class TitleHintError(RuntimeError):
    pass


@dataclass(frozen=True)
class TitleHintConfig:
    min_horizontal_ratio: float = 1.2
    container_left_band_ratio: float = 0.22
    container_min_inside_ratio: float = 0.7
    ambiguity_margin: float = 0.05
    diagram_top_band_ratio: float = 0.20
    diagram_left_band_ratio: float = 0.40


def assign_title_hints(
    payload: Dict[str, Any],
    cfg: Optional[TitleHintConfig] = None,
) -> Dict[str, Any]:
    """
    Step 9.2:
    Detect pool/lane/diagram title candidates from geometry and store hints on text nodes.
    """
    if cfg is None:
        cfg = TitleHintConfig()

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise TitleHintError("payload must contain 'nodes' list")

    image = payload.get("image", {})
    if not isinstance(image, dict):
        image = {}
    img_w = _to_float(image.get("width"), 0.0)
    img_h = _to_float(image.get("height"), 0.0)

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    warnings: List[str] = []
    old_w = meta.get("warnings", [])
    if isinstance(old_w, list):
        warnings.extend([str(x) for x in old_w])

    parsed_nodes = [n for n in nodes if isinstance(n, dict)]
    text_nodes = [n for n in parsed_nodes if n.get("type") == "text"]
    container_nodes = [n for n in parsed_nodes if n.get("type") == "container"]

    # prepare container geometry
    containers = []
    for c in container_nodes:
        cb = _bbox(c.get("bbox"))
        cid = c.get("id")
        cname = str(c.get("class_name", "")).strip().lower()
        if cb and isinstance(cid, str):
            containers.append((cid, cname, cb))

    text_hint_map: Dict[str, Dict[str, Any]] = {}

    # 1) pool/lane title hints (left-edge horizontal text inside container)
    for t in text_nodes:
        tid = t.get("id")
        tb = _bbox(t.get("bbox"))
        if not (isinstance(tid, str) and tb):
            continue
        if not _is_horizontal(tb, cfg.min_horizontal_ratio):
            continue

        cands: List[Tuple[float, str, str]] = []
        for cid, cname, cb in containers:
            inside = _inside_ratio(tb, cb)
            if inside < cfg.container_min_inside_ratio:
                continue
            if not _in_left_band(tb, cb, cfg.container_left_band_ratio):
                continue
            # score: prefer more inside + more left-aligned
            left_dist = abs(tb[0] - cb[0]) / max(1.0, cb[2] - cb[0])
            score = inside + (1.0 - left_dist)
            cands.append((score, cid, cname))

        if not cands:
            continue
        cands.sort(reverse=True)
        best_score, best_cid, best_cname = cands[0]

        hint_type = "container_title"
        if best_cname == "pool":
            hint_type = "pool_title"
        elif best_cname == "lane":
            hint_type = "lane_title"

        text_hint_map[tid] = {
            "title_hint": hint_type,
            "title_target_container_id": best_cid,
            "title_hint_score": round(best_score, 4),
        }

        if len(cands) > 1 and (best_score - cands[1][0]) <= cfg.ambiguity_margin:
            warnings.append(
                f"title_hint_ambiguous: text {tid} has close container candidates {best_cid} and {cands[1][1]}"
            )

    # 2) diagram title candidate (top-left horizontal text, not already used as container title)
    if img_w > 0 and img_h > 0:
        diagram_cands: List[Tuple[float, str]] = []
        for t in text_nodes:
            tid = t.get("id")
            tb = _bbox(t.get("bbox"))
            if not (isinstance(tid, str) and tb):
                continue
            if tid in text_hint_map:
                continue
            if not _is_horizontal(tb, cfg.min_horizontal_ratio):
                continue
            if tb[1] > img_h * cfg.diagram_top_band_ratio:
                continue
            if tb[0] > img_w * cfg.diagram_left_band_ratio:
                continue
            # prefer larger, more top-left text
            area = max(1.0, (tb[2] - tb[0]) * (tb[3] - tb[1]))
            pos_score = (1.0 - tb[1] / max(1.0, img_h)) + (1.0 - tb[0] / max(1.0, img_w))
            score = area * 0.001 + pos_score
            diagram_cands.append((score, tid))

        if diagram_cands:
            diagram_cands.sort(reverse=True)
            best_score, best_tid = diagram_cands[0]
            text_hint_map[best_tid] = {
                "title_hint": "diagram_title_candidate",
                "title_target_container_id": None,
                "title_hint_score": round(best_score, 4),
            }
            if len(diagram_cands) > 1 and (best_score - diagram_cands[1][0]) <= cfg.ambiguity_margin:
                warnings.append(
                    f"title_hint_ambiguous: multiple diagram title candidates near top-left ({best_tid}, {diagram_cands[1][1]})"
                )

    updated_nodes: List[Dict[str, Any]] = []
    hint_count = 0
    for n in parsed_nodes:
        out_n = dict(n)
        nid = out_n.get("id")
        if isinstance(nid, str) and nid in text_hint_map:
            out_n.update(text_hint_map[nid])
            hint_count += 1
        updated_nodes.append(out_n)

    updated_nodes.sort(key=lambda n: (int(n.get("original_index", 10**12)), str(n.get("id", ""))))

    out_meta = dict(meta)
    out_meta["warnings"] = warnings

    out = dict(payload)
    out["meta"] = out_meta
    out["nodes"] = updated_nodes
    out["title_hint_stats"] = {
        "text_nodes": len(text_nodes),
        "container_nodes": len(container_nodes),
        "title_hints_assigned": hint_count,
    }
    return out


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _bbox(v: Any) -> Optional[Tuple[float, float, float, float]]:
    if not (isinstance(v, list) and len(v) == 4):
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in v]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _is_horizontal(b: Tuple[float, float, float, float], min_ratio: float) -> bool:
    w = b[2] - b[0]
    h = b[3] - b[1]
    if h <= 1e-9:
        return True
    return (w / h) >= min_ratio


def _inside_ratio(
    inner: Tuple[float, float, float, float],
    outer: Tuple[float, float, float, float],
) -> float:
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_inner = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area_inner


def _in_left_band(
    text_box: Tuple[float, float, float, float],
    container_box: Tuple[float, float, float, float],
    left_band_ratio: float,
) -> bool:
    c_w = container_box[2] - container_box[0]
    left_limit = container_box[0] + left_band_ratio * c_w
    # center-based criterion: text belongs to left area
    cx = (text_box[0] + text_box[2]) / 2.0
    return cx <= left_limit
