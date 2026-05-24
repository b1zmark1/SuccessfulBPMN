"""
Геометрическая детекция лейнов (swim-lanes) и присвоение lane_role shape-нодам.

Подход:
1. Считаем медианную высоту shape-нод (для LR-диаграмм) или ширину (для TB).
2. Сортируем shapes по Y-центру (LR) или X-центру (TB).
3. Single-link кластеризация: соседние shapes в одной ряду, если разрыв
   между их центрами меньше gap_factor × median_height.
4. Для каждого кластера ищем text-ноды:
   - чьи центры лежат в Y-диапазоне этого ряда (LR),
   - расположенные левее самой левой shape ряда.
   Самый левый из них — имя лейна.
5. Записываем lane_role во все shape-ноды кластера.

Это работает без YOLOX-классов `lane`/`pool` и без полагания на title_hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class LaneDetectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class LaneDetectionConfig:
    min_shapes_per_row: int = 1       # достаточно одной shape в ряду — даже одиночные задачи могут иметь лейн
    gap_factor: float = 0.8           # порог разрыва как доля медианной высоты
    # доля медианной высоты, на которую текст может ЗАЕХАТЬ в shape слева
    # (некоторые диаграммы рисуют label прямо у границы lane без отступа)
    label_overlap_into_shape_factor: float = 0.3
    label_max_text_length: int = 40   # длинный текст вряд ли роль
    label_min_text_length: int = 2    # совсем короткое — мусор
    overlap_y_ratio: float = 0.25     # text-нода считается "в ряду", если её Y пересекается с рядом на эту долю


def detect_lanes_geometrically(
    payload: Dict[str, Any],
    cfg: Optional[LaneDetectionConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = LaneDetectionConfig()

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise LaneDetectionError("payload must contain 'nodes' list")

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    direction = str(meta.get("direction", "LR")).upper()
    if direction not in {"LR", "TB"}:
        direction = "LR"

    parsed_nodes = [n for n in nodes if isinstance(n, dict)]
    shapes = [n for n in parsed_nodes if n.get("type") == "shape"]
    texts = [n for n in parsed_nodes if n.get("type") == "text"]

    if len(shapes) < cfg.min_shapes_per_row:
        return payload

    # Для LR: ряды (строки) — по Y; для TB: колонки — по X.
    primary_axis = 1 if direction == "LR" else 0
    secondary_axis = 0 if direction == "LR" else 1

    def _bbox_dim(n: Dict[str, Any], dim_lo: int, dim_hi: int) -> Optional[Tuple[float, float]]:
        b = n.get("bbox")
        if not (isinstance(b, list) and len(b) == 4):
            return None
        try:
            return float(b[dim_lo]), float(b[dim_hi])
        except Exception:
            return None

    def _bbox(n: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        b = n.get("bbox")
        if not (isinstance(b, list) and len(b) == 4):
            return None
        try:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        except Exception:
            return None

    def _center(n: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        c = n.get("center")
        if not (isinstance(c, list) and len(c) == 2):
            return None
        try:
            return float(c[0]), float(c[1])
        except Exception:
            return None

    # Медианная высота (LR) или ширина (TB) shape-нод.
    sizes: List[float] = []
    for s in shapes:
        rng = _bbox_dim(s, primary_axis, primary_axis + 2)
        if rng is not None:
            sizes.append(abs(rng[1] - rng[0]))
    if not sizes:
        return payload
    sizes.sort()
    median_size = sizes[len(sizes) // 2]
    gap_threshold = max(8.0, median_size * cfg.gap_factor)

    # Сортируем shapes по центру вдоль primary axis.
    def _primary_center(n: Dict[str, Any]) -> float:
        c = _center(n)
        return c[primary_axis] if c else 0.0

    sorted_shapes = sorted(shapes, key=_primary_center)

    # Кластеризация по разрыву.
    clusters: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [sorted_shapes[0]]
    for s in sorted_shapes[1:]:
        prev_c = _primary_center(current[-1])
        cur_c = _primary_center(s)
        if cur_c - prev_c > gap_threshold:
            clusters.append(current)
            current = [s]
        else:
            current.append(s)
    clusters.append(current)

    lane_role_by_shape_id: Dict[str, str] = {}
    # Все shape_id из всех кластеров: для них существующий lane_role (вероятно
    # от text_hooks/title_hints, который любит давать мусор типа "поездов") мы
    # очищаем, если геометрия не нашла лучшего лейна.
    shapes_in_clusters: set[str] = set()
    cluster_meta: List[Dict[str, Any]] = []

    for ci, cluster in enumerate(clusters):
        # Запоминаем все shapes этого кластера для последующей очистки lane_role.
        for s in cluster:
            sid = s.get("id")
            if isinstance(sid, str):
                shapes_in_clusters.add(sid)

        if len(cluster) < cfg.min_shapes_per_row:
            cluster_meta.append({"index": ci, "size": len(cluster), "lane": None, "skipped": True})
            continue

        # Границы ряда вдоль primary axis (для LR — Y_min..Y_max ряда).
        prim_lo = float("inf")
        prim_hi = float("-inf")
        sec_min = float("inf")
        for s in cluster:
            rng = _bbox_dim(s, primary_axis, primary_axis + 2)
            if rng is not None:
                prim_lo = min(prim_lo, rng[0])
                prim_hi = max(prim_hi, rng[1])
            srng = _bbox_dim(s, secondary_axis, secondary_axis + 2)
            if srng is not None:
                sec_min = min(sec_min, srng[0])

        if prim_lo == float("inf") or sec_min == float("inf"):
            cluster_meta.append({"index": ci, "size": len(cluster), "lane": None, "skipped": True})
            continue

        row_span = prim_hi - prim_lo
        # Кандидаты на роль: text-ноды, чей primary-диапазон пересекает диапазон ряда
        # и которые расположены ЛЕВЕЕ (для LR) самой левой shape ряда.
        candidates: List[Tuple[float, str]] = []
        for t in texts:
            tb = _bbox(t)
            if tb is None:
                continue
            t_prim_lo = tb[primary_axis]
            t_prim_hi = tb[primary_axis + 2]
            t_sec_lo = tb[secondary_axis]
            t_sec_hi = tb[secondary_axis + 2]

            # Допускаем что текст слегка заезжает в shape (label_overlap_into_shape_factor).
            # Главное — центр текста должен быть левее левого края первой shape ряда.
            allow_overlap = cfg.label_overlap_into_shape_factor * median_size
            t_sec_center = (t_sec_lo + t_sec_hi) / 2.0
            if t_sec_center > sec_min:
                # Центр текста ПРАВЕЕ первой shape — точно не лейн.
                continue
            if t_sec_lo > sec_min + allow_overlap:
                # Левый край текста сильно справа от shape — не лейн.
                continue

            # Y-пересечение с рядом должно покрывать ≥ overlap_y_ratio высоты текста.
            inter_lo = max(t_prim_lo, prim_lo)
            inter_hi = min(t_prim_hi, prim_hi)
            inter = max(0.0, inter_hi - inter_lo)
            t_span = max(1e-6, t_prim_hi - t_prim_lo)
            if inter / t_span < cfg.overlap_y_ratio and inter / max(1e-6, row_span) < cfg.overlap_y_ratio:
                continue

            text_val = t.get("text")
            if not isinstance(text_val, str):
                continue
            cleaned = text_val.strip()
            if len(cleaned) < cfg.label_min_text_length:
                continue
            if len(cleaned) > cfg.label_max_text_length:
                continue

            # Ранг: чем левее текст, тем приоритетнее (минимум по secondary axis).
            candidates.append((t_sec_lo, cleaned))

        candidates.sort(key=lambda c: c[0])
        lane_label: Optional[str] = candidates[0][1] if candidates else None

        if lane_label:
            for s in cluster:
                sid = s.get("id")
                if isinstance(sid, str):
                    lane_role_by_shape_id[sid] = lane_label

        cluster_meta.append(
            {
                "index": ci,
                "size": len(cluster),
                "lane": lane_label,
                "primary_range": [prim_lo, prim_hi],
                "secondary_min": sec_min,
            }
        )

    # Применяем lane_role. Геом-детекция считается источником истины:
    # - если для shape нашли lane_label — ставим его (перезаписывая шум от text_hooks)
    # - если shape попала в кластер, но lane_label не нашёлся — очищаем существующий
    #   lane_role (он почти наверняка мусор от title_hints вроде "поездов")
    # - если shape вне кластеров — оставляем как есть
    updated_nodes: List[Dict[str, Any]] = []
    overrides = 0
    additions = 0
    cleared = 0
    for n in parsed_nodes:
        out_n = dict(n)
        nid = out_n.get("id")
        if isinstance(nid, str):
            if nid in lane_role_by_shape_id:
                new_lane = lane_role_by_shape_id[nid]
                existing = out_n.get("lane_role")
                if existing:
                    if existing != new_lane:
                        overrides += 1
                else:
                    additions += 1
                out_n["lane_role"] = new_lane
            elif nid in shapes_in_clusters and out_n.get("lane_role"):
                out_n["lane_role"] = None
                cleared += 1
        updated_nodes.append(out_n)

    out = dict(payload)
    out["nodes"] = updated_nodes
    out["lane_detection_stats"] = {
        "direction": direction,
        "median_shape_primary_size": median_size,
        "gap_threshold": gap_threshold,
        "clusters_total": len(clusters),
        "lanes_named": sum(1 for c in cluster_meta if c.get("lane")),
        "shapes_with_lane": len(lane_role_by_shape_id),
        "lane_role_overrides": overrides,
        "lane_role_additions": additions,
        "lane_role_cleared": cleared,
        "clusters": cluster_meta,
    }
    return out
