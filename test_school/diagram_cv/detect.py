import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectConfig:
    min_node_area: int = 250
    max_node_area_ratio: float = 0.60
    container_min_area_ratio: float = 0.12
    container_max_area_ratio: float = 0.95
    node_bbox_pad_for_remove: int = 6
    container_bbox_pad_for_remove: int = 4
    touch_dilate_px: int = 14
    min_path_len_px: int = 25
    sample_every_px: int = 8
    skeleton_max_iters: int = 80
    write_debug_images: bool = True


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _imwrite(path: str, img: np.ndarray) -> None:
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _to_binary_black_on_white(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    black_ratio = float((b < 128).mean())
    if black_ratio > 0.5:
        b = 255 - b
    return b


def _bbox_pad_xyxy(
    bbox_xywh: Tuple[int, int, int, int], pad: int, w: int, h: int
) -> Tuple[int, int, int, int]:
    x, y, bw, bh = bbox_xywh
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w - 1, x + bw + pad)
    y1 = min(h - 1, y + bh + pad)
    return x0, y0, x1, y1


def _contour_bbox_xywh(cnt: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(cnt)
    return int(x), int(y), int(w), int(h)


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def _approx_vertices(cnt: np.ndarray, eps_ratio: float = 0.02) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    eps = eps_ratio * peri
    return cv2.approxPolyDP(cnt, eps, True)


def _classify_shape(cnt: np.ndarray) -> str:
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if area < 1.0 or peri < 1.0:
        return "unknown"

    x, y, w, h = _contour_bbox_xywh(cnt)
    aspect = float(w) / float(h) if h > 0 else 0.0
    circ = _circularity(area, peri)

    approx = _approx_vertices(cnt, eps_ratio=0.02)
    v = int(len(approx))

    if circ > 0.78 and 0.80 <= aspect <= 1.25:
        return "circle"

    if v == 4 and 0.75 <= aspect <= 1.33:
        rect = cv2.minAreaRect(cnt)
        angle = float(rect[2])
        a = abs(angle)
        if 15.0 < a < 75.0:
            return "diamond"

    if v == 4:
        return "rectangle"

    if v >= 5:
        return "rounded_rect"

    return "unknown"


def detect_nodes(geom_gray: np.ndarray, cfg: DetectConfig) -> Tuple[List[Dict], List[Dict]]:
    geom = _to_binary_black_on_white(geom_gray)
    h, w = geom.shape[:2]
    img_area = float(w * h)

    inv = 255 - geom

    contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return [], []

    hierarchy = hierarchy[0]

    candidates: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]] = []
    for i, cnt in enumerate(contours):
        child = int(hierarchy[i][2])
        parent = int(hierarchy[i][3])
        if parent != -1 or child == -1:
            continue

        area = float(cv2.contourArea(cnt))
        if area < float(cfg.min_node_area):
            continue
        if area / img_area > cfg.max_node_area_ratio:
            continue

        bbox = _contour_bbox_xywh(cnt)
        candidates.append((cnt, bbox, area))

    containers: List[Dict] = []
    elements: List[Dict] = []

    candidates.sort(key=lambda t: (t[1][1], t[1][0]))

    cid = 0
    eid = 0
    for cnt, bbox, area in candidates:
        x, y, bw, bh = bbox
        bbox_area = float(bw * bh)
        bbox_area_ratio = bbox_area / img_area

        width_ratio = float(bw) / float(w) if w > 0 else 0.0
        height_ratio = float(bh) / float(h) if h > 0 else 0.0

        kind = _classify_shape(cnt)

        is_container = (
            cfg.container_min_area_ratio <= bbox_area_ratio <= cfg.container_max_area_ratio
            and (width_ratio > 0.60 or height_ratio > 0.45)
            and kind in ("rectangle", "rounded_rect", "unknown")
        )

        cx = int(x + bw / 2)
        cy = int(y + bh / 2)

        if is_container:
            containers.append(
                {
                    "id": f"c{cid}",
                    "kind": "container",
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "center": [cx, cy],
                    "area": float(area),
                }
            )
            cid += 1
        else:
            elements.append(
                {
                    "id": f"n{eid}",
                    "kind": kind,
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "center": [cx, cy],
                    "area": float(area),
                }
            )
            eid += 1

    return elements, containers


def _draw_nodes_overlay(
    original_bgr: np.ndarray,
    elements: List[Dict],
    containers: List[Dict],
    out_path: str,
) -> None:
    img = original_bgr.copy()

    for c in containers:
        x, y, w, h = c["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            img,
            f'{c["id"]}:{c["kind"]}',
            (x + 4, max(12, y + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 180),
            1,
            cv2.LINE_AA,
        )

    for n in elements:
        x, y, w, h = n["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            f'{n["id"]}:{n["kind"]}',
            (x + 4, max(12, y + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 150, 0),
            1,
            cv2.LINE_AA,
        )

    _imwrite(out_path, img)


def _build_remove_mask(
    shape_hw: Tuple[int, int],
    elements: List[Dict],
    containers: List[Dict],
    cfg: DetectConfig,
) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    for n in elements:
        x0, y0, x1, y1 = _bbox_pad_xyxy(tuple(n["bbox"]), cfg.node_bbox_pad_for_remove, w, h)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    for c in containers:
        x0, y0, x1, y1 = _bbox_pad_xyxy(tuple(c["bbox"]), cfg.container_bbox_pad_for_remove, w, h)
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    return mask


def _zhang_suen_thinning(bin_img_01: np.ndarray, max_iters: int) -> np.ndarray:
    img = bin_img_01.copy().astype(np.uint8)
    h, w = img.shape[:2]
    if h < 3 or w < 3:
        return img

    def neighbours(y: int, x: int) -> List[int]:
        return [
            img[y - 1, x],
            img[y - 1, x + 1],
            img[y, x + 1],
            img[y + 1, x + 1],
            img[y + 1, x],
            img[y + 1, x - 1],
            img[y, x - 1],
            img[y - 1, x - 1],
        ]

    def transitions(ns: List[int]) -> int:
        n = ns + [ns[0]]
        t = 0
        for i in range(8):
            if n[i] == 0 and n[i + 1] == 1:
                t += 1
        return t

    it = 0
    while it < max_iters:
        it += 1
        changed = False

        to_remove: List[Tuple[int, int]] = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                ns = neighbours(y, x)
                n_sum = sum(ns)
                if n_sum < 2 or n_sum > 6:
                    continue
                if transitions(ns) != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = ns
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_remove.append((y, x))

        if to_remove:
            for y, x in to_remove:
                img[y, x] = 0
            changed = True

        to_remove = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] != 1:
                    continue
                ns = neighbours(y, x)
                n_sum = sum(ns)
                if n_sum < 2 or n_sum > 6:
                    continue
                if transitions(ns) != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = ns
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_remove.append((y, x))

        if to_remove:
            for y, x in to_remove:
                img[y, x] = 0
            changed = True

        if not changed:
            break

    return img


def _skeletonize(bin_img_01: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    try:
        thinning = cv2.ximgproc.thinning  # type: ignore[attr-defined]
        src = (bin_img_01 * 255).astype(np.uint8)
        out = thinning(src, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)  # type: ignore[attr-defined]
        return (out > 0).astype(np.uint8)
    except Exception:
        return _zhang_suen_thinning(bin_img_01, max_iters=cfg.skeleton_max_iters)


def _build_pixel_graph(
    skel_01: np.ndarray,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], List[List[int]], np.ndarray]:
    coords = np.argwhere(skel_01 > 0)
    idx_map: Dict[Tuple[int, int], int] = {(int(y), int(x)): i for i, (y, x) in enumerate(coords)}
    n = int(coords.shape[0])
    adj: List[List[int]] = [[] for _ in range(n)]

    for i, (y, x) in enumerate(coords):
        yi = int(y)
        xi = int(x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                key = (yi + dy, xi + dx)
                j = idx_map.get(key)
                if j is not None:
                    adj[i].append(j)

    deg = np.array([len(a) for a in adj], dtype=np.int32)
    return coords, idx_map, adj, deg


def _trace_paths(coords: np.ndarray, adj: List[List[int]], deg: np.ndarray) -> List[List[int]]:
    terminals = set(int(i) for i in np.where(deg != 2)[0].tolist())
    visited_edges: set[Tuple[int, int]] = set()
    paths: List[List[int]] = []

    def edge_key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    for t in list(terminals):
        for nb in adj[t]:
            ek = edge_key(t, nb)
            if ek in visited_edges:
                continue

            path = [t]
            prev = t
            cur = nb
            visited_edges.add(ek)

            while True:
                path.append(cur)
                if cur in terminals:
                    break

                neigh = adj[cur]
                if len(neigh) < 1:
                    break
                if len(neigh) == 1:
                    nxt = neigh[0]
                else:
                    nxt = neigh[0] if neigh[0] != prev else neigh[1]

                visited_edges.add(edge_key(cur, nxt))
                prev, cur = cur, nxt

            paths.append(path)

    filtered: List[List[int]] = []
    for p in paths:
        if len(p) >= 2:
            filtered.append(p)
    return filtered


def _downsample_path_xy(coords: np.ndarray, path_idx: List[int], every: int) -> List[List[int]]:
    if not path_idx:
        return []
    out: List[List[int]] = []
    out.append([int(coords[path_idx[0], 1]), int(coords[path_idx[0], 0])])
    for k in range(every, len(path_idx) - 1, every):
        out.append([int(coords[path_idx[k], 1]), int(coords[path_idx[k], 0])])
    if len(path_idx) > 1:
        out.append([int(coords[path_idx[-1], 1]), int(coords[path_idx[-1], 0])])
    return out


def _point_hits_nodes(
    pt_xy: Tuple[int, int],
    padded_bboxes_xyxy: List[Tuple[str, Tuple[int, int, int, int]]],
) -> List[str]:
    x, y = pt_xy
    hits: List[str] = []
    for node_id, (x0, y0, x1, y1) in padded_bboxes_xyxy:
        if x0 <= x <= x1 and y0 <= y <= y1:
            hits.append(node_id)
    return hits


def detect_edges(
    cvbin_gray: np.ndarray,
    elements: List[Dict],
    containers: List[Dict],
    cfg: DetectConfig,
    out_dir: str,
) -> Tuple[List[Dict], int]:
    cvbin = _to_binary_black_on_white(cvbin_gray)
    h, w = cvbin.shape[:2]

    linework_01 = (cvbin < 128).astype(np.uint8)
    linework_255 = (linework_01 * 255).astype(np.uint8)

    remove_mask = _build_remove_mask((h, w), elements, containers, cfg)
    connectors_01 = linework_01.copy()
    connectors_01[remove_mask > 0] = 0

    connectors_255 = (connectors_01 * 255).astype(np.uint8)
    connectors_255 = cv2.morphologyEx(
        connectors_255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    connectors_01 = (connectors_255 > 0).astype(np.uint8)

    skel_01 = _skeletonize(connectors_01, cfg)

    coords, _, adj, deg = _build_pixel_graph(skel_01)
    if coords.size == 0:
        return [], 0

    paths_idx = _trace_paths(coords, adj, deg)

    padded: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for n in elements:
        x0, y0, x1, y1 = _bbox_pad_xyxy(tuple(n["bbox"]), cfg.touch_dilate_px, w, h)
        padded.append((n["id"], (x0, y0, x1, y1)))

    edges: List[Dict] = []
    ambiguous = 0
    eid = 0

    for p in paths_idx:
        if len(p) < cfg.min_path_len_px:
            continue

        p0 = (int(coords[p[0], 1]), int(coords[p[0], 0]))
        p1 = (int(coords[p[-1], 1]), int(coords[p[-1], 0]))

        hits0 = _point_hits_nodes(p0, padded)
        hits1 = _point_hits_nodes(p1, padded)

        if len(hits0) == 0 and len(hits1) == 0:
            continue

        is_amb = False
        src: Optional[str] = None
        dst: Optional[str] = None

        if len(hits0) == 1 and len(hits1) == 1 and hits0[0] != hits1[0]:
            src = hits0[0]
            dst = hits1[0]
        else:
            is_amb = True
            ambiguous += 1
            src = hits0[0] if len(hits0) == 1 else (hits0[0] if hits0 else None)
            dst = hits1[0] if len(hits1) == 1 else (hits1[0] if hits1 else None)

        poly = _downsample_path_xy(coords, p, every=cfg.sample_every_px)

        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        if not xs or not ys:
            continue

        x0 = int(max(0, min(xs)))
        y0 = int(max(0, min(ys)))
        x1 = int(min(w - 1, max(xs)))
        y1 = int(min(h - 1, max(ys)))
        bbox = [x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)]

        edges.append(
            {
                "id": f"e{eid}",
                "source": src,
                "target": dst,
                "directed": False,
                "ambiguous": bool(is_amb),
                "points": poly,
                "bbox": bbox,
                "length_px": int(len(p)),
            }
        )
        eid += 1

    if cfg.write_debug_images:
        _imwrite(os.path.join(out_dir, "20_linework.png"), linework_255)
        _imwrite(os.path.join(out_dir, "21_remove_mask.png"), remove_mask)
        _imwrite(os.path.join(out_dir, "22_connectors.png"), (connectors_01 * 255).astype(np.uint8))
        _imwrite(os.path.join(out_dir, "23_connectors_skeleton.png"), (skel_01 * 255).astype(np.uint8))

    return edges, ambiguous


def _draw_edges_overlay(
    original_bgr: np.ndarray,
    elements: List[Dict],
    containers: List[Dict],
    edges: List[Dict],
    out_path: str,
) -> None:
    img = original_bgr.copy()

    for c in containers:
        x, y, w, h = c["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    for n in elements:
        x, y, w, h = n["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for e in edges:
        pts = e.get("points", [])
        if len(pts) < 2:
            continue
        poly = np.array([[p[0], p[1]] for p in pts], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [poly], isClosed=False, color=(255, 0, 0), thickness=2)
        mid = pts[len(pts) // 2]
        label = f'{e["id"]}:{"amb" if e.get("ambiguous") else "ok"}'
        cv2.putText(
            img,
            label,
            (int(mid[0]) + 3, int(mid[1]) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 0, 0),
            1,
            cv2.LINE_AA,
        )

    _imwrite(out_path, img)


def save_result_json(
    out_dir: str,
    elements: List[Dict],
    containers: List[Dict],
    edges: List[Dict],
) -> str:
    result = {
        "elements": elements,
        "containers": containers,
        "edges": edges,
    }
    path = os.path.join(out_dir, "result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True, help="Path to 06_geom.png")
    ap.add_argument("--cvbin", required=True, help="Path to 05_cv_binary.png")
    ap.add_argument("--original", required=True, help="Path to 00_original.png")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    cfg = DetectConfig()

    geom = _load_image_gray(args.geom)
    cvbin = _load_image_gray(args.cvbin)
    original = _load_image_bgr(args.original)

    elements, containers = detect_nodes(geom, cfg)

    _draw_nodes_overlay(
        original_bgr=original,
        elements=elements,
        containers=containers,
        out_path=os.path.join(args.outdir, "10_overlay_nodes.png"),
    )

    edges, ambiguous = detect_edges(
        cvbin_gray=cvbin,
        elements=elements,
        containers=containers,
        cfg=cfg,
        out_dir=args.outdir,
    )

    _draw_edges_overlay(
        original_bgr=original,
        elements=elements,
        containers=containers,
        edges=edges,
        out_path=os.path.join(args.outdir, "11_overlay_edges.png"),
    )

    result_path = save_result_json(
        out_dir=args.outdir,
        elements=elements,
        containers=containers,
        edges=edges,
    )

    summary = {
        "nodes_count": int(len(elements) + len(containers)),
        "edges_count": int(len(edges)),
        "elements_count": int(len(elements)),
        "containers_count": int(len(containers)),
        "ambiguous_edges_count": int(ambiguous),
        "result_json": result_path,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
