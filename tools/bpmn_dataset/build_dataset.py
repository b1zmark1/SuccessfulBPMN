import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import cairosvg  # type: ignore
except Exception:  # pragma: no cover - optional dependency handled at runtime
    cairosvg = None


BPMN_NS = {
    'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
    'bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
    'dc': 'http://www.omg.org/spec/DD/20100524/DC',
    'di': 'http://www.omg.org/spec/DD/20100524/DI',
}

MVP_CLASSES = [
    'start_event',
    'intermediate_event',
    'end_event',
    'task',
    'gateway_exclusive',
    'gateway_parallel',
    'gateway_inclusive',
    'subprocess',
    'pool',
    'lane',
    'data_object',
    'text',
    'text_annotation',
    'sequence_flow',
]

CLASS_TO_ID = {name: i for i, name in enumerate(MVP_CLASSES)}

# Padding to capture label text rendered outside BPMNLabel bounds.
TEXT_PAD_X_PCT = 0.25
TEXT_PAD_Y_PCT = 0.75
MIN_TEXT_PAD = 6.0

@dataclass(frozen=True)
class ShapeLabel:
    class_name: str
    bbox: Tuple[float, float, float, float]  # x, y, w, h in BPMN coords


@dataclass(frozen=True)
class DiagramLabels:
    shapes: List[ShapeLabel]
    viewbox: Tuple[float, float, float, float]  # x, y, w, h


def _local_name(tag: str) -> str:
    if '}' in tag:
        return tag.split('}', 1)[1]
    if ':' in tag:
        return tag.split(':', 1)[1]
    return tag


def _build_id_map(root: ET.Element) -> Dict[str, ET.Element]:
    out: Dict[str, ET.Element] = {}
    for el in root.iter():
        el_id = el.attrib.get('id')
        if el_id:
            out[el_id] = el
    return out


def _map_bpmn_tag_to_class(tag: str) -> Optional[str]:
    name = _local_name(tag)

    if name in ('startEvent',):
        return 'start_event'
    if name in ('endEvent',):
        return 'end_event'
    if name in (
        'intermediateCatchEvent',
        'intermediateThrowEvent',
        'boundaryEvent',
    ):
        return 'intermediate_event'

    if name in (
        'task',
        'userTask',
        'serviceTask',
        'manualTask',
        'scriptTask',
        'businessRuleTask',
        'sendTask',
        'receiveTask',
    ):
        return 'task'

    if name == 'exclusiveGateway':
        return 'gateway_exclusive'
    if name == 'parallelGateway':
        return 'gateway_parallel'
    if name == 'inclusiveGateway':
        return 'gateway_inclusive'

    if name in ('subProcess', 'callActivity'):
        return 'subprocess'

    if name == 'participant':
        return 'pool'
    if name == 'lane':
        return 'lane'

    if name in ('dataObject', 'dataObjectReference', 'dataInput', 'dataOutput'):
        return 'data_object'

    if name == 'textAnnotation':
        return 'text_annotation'

    if name in ('sequenceFlow', 'messageFlow', 'association'):
        return 'sequence_flow'

    return None


def _bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0


def _filter_pool_lane_overlaps(shapes: List[ShapeLabel]) -> List[ShapeLabel]:
    pools = [s for s in shapes if s.class_name == 'pool']
    if not pools:
        return shapes

    filtered: List[ShapeLabel] = []
    for s in shapes:
        if s.class_name != 'lane':
            filtered.append(s)
            continue

        drop_lane = False
        for p in pools:
            iou = _bbox_iou(s.bbox, p.bbox)
            if iou >= 0.90:
                # Single-lane pool: lane bbox nearly equals pool bbox.
                drop_lane = True
                break
        if not drop_lane:
            filtered.append(s)

    return filtered


def parse_bpmn_labels(path: Path) -> List[ShapeLabel]:
    tree = ET.parse(path)
    root = tree.getroot()

    id_map = _build_id_map(root)
    labels: List[ShapeLabel] = []

    for shape in root.findall('.//bpmndi:BPMNShape', BPMN_NS):
        bpmn_id = shape.attrib.get('bpmnElement')
        if not bpmn_id:
            continue
        element = id_map.get(bpmn_id)
        if element is None:
            continue
        class_name = _map_bpmn_tag_to_class(element.tag)
        if class_name is None:
            continue

        bounds = shape.find('dc:Bounds', BPMN_NS)
        if bounds is None:
            continue
        x = float(bounds.attrib.get('x', '0'))
        y = float(bounds.attrib.get('y', '0'))
        w = float(bounds.attrib.get('width', '0'))
        h = float(bounds.attrib.get('height', '0'))
        if w <= 1 or h <= 1:
            continue

        labels.append(ShapeLabel(class_name=class_name, bbox=(x, y, w, h)))

    for edge in root.findall('.//bpmndi:BPMNEdge', BPMN_NS):
        bpmn_id = edge.attrib.get('bpmnElement')
        if not bpmn_id:
            continue
        element = id_map.get(bpmn_id)
        if element is None:
            continue
        class_name = _map_bpmn_tag_to_class(element.tag)
        if class_name is None:
            continue
        bbox = _edge_bbox_from_waypoints(edge)
        if bbox is None:
            continue
        labels.append(ShapeLabel(class_name=class_name, bbox=bbox))

    return _filter_pool_lane_overlaps(labels)


def _edge_bbox_from_waypoints(edge: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    points: List[Tuple[float, float]] = []
    for wp in edge.findall('di:waypoint', BPMN_NS):
        try:
            x = float(wp.attrib.get('x', '0'))
            y = float(wp.attrib.get('y', '0'))
        except ValueError:
            continue
        points.append((x, y))
    if len(points) < 2:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    w = max_x - min_x
    h = max_y - min_y

    min_thickness = 6.0
    pad = 2.0
    pad_x = (min_thickness - w) / 2.0 if w < min_thickness else 0.0
    pad_y = (min_thickness - h) / 2.0 if h < min_thickness else 0.0

    min_x -= (pad_x + pad)
    max_x += (pad_x + pad)
    min_y -= (pad_y + pad)
    max_y += (pad_y + pad)

    w = max_x - min_x
    h = max_y - min_y
    if w <= 1 or h <= 1:
        return None
    return min_x, min_y, w, h


def _load_viewbox(meta_path: Path) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]:
    with meta_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    vb = data.get('viewbox') or {}
    outer = (
        float(vb.get('x', 0)),
        float(vb.get('y', 0)),
        float(vb.get('width', 0)),
        float(vb.get('height', 0)),
    )
    inner = vb.get('inner') or {}
    inner_box = (
        float(inner.get('x', outer[0])),
        float(inner.get('y', outer[1])),
        float(inner.get('width', outer[2])),
        float(inner.get('height', outer[3])),
    )
    if inner_box[2] <= 0 or inner_box[3] <= 0:
        inner_box = outer
    text_boxes_raw = data.get('textBoxes') or data.get('text_boxes') or []
    text_boxes: List[Tuple[float, float, float, float]] = []
    for tb in text_boxes_raw:
        try:
            x = float(tb.get('x', 0))
            y = float(tb.get('y', 0))
            w = float(tb.get('width', 0))
            h = float(tb.get('height', 0))
        except Exception:
            continue
        if w <= 1 or h <= 1:
            continue
        text_boxes.append((x, y, w, h))

    # Filter text boxes that are completely outside the inner viewbox.
    if text_boxes:
        ix, iy, iw, ih = inner_box
        margin = 2.0
        filtered: List[Tuple[float, float, float, float]] = []
        for x, y, w, h in text_boxes:
            x2, y2 = x + w, y + h
            if x2 < ix - margin or y2 < iy - margin or x > ix + iw + margin or y > iy + ih + margin:
                continue
            filtered.append((x, y, w, h))
        text_boxes = filtered

    return outer, inner_box, text_boxes


def _normalize_bbox(
    bbox: Tuple[float, float, float, float],
    viewbox: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    vb_x, vb_y, vb_w, vb_h = viewbox
    x0 = (x - vb_x) / vb_w
    y0 = (y - vb_y) / vb_h
    w0 = w / vb_w
    h0 = h / vb_h
    cx = x0 + w0 / 2.0
    cy = y0 + h0 / 2.0
    return cx, cy, w0, h0


def _write_yolo_labels(
    out_path: Path,
    shapes: Iterable[ShapeLabel],
    viewbox: Tuple[float, float, float, float],
) -> None:
    lines: List[str] = []
    for s in shapes:
        class_id = CLASS_TO_ID[s.class_name]
        cx, cy, w, h = _normalize_bbox(s.bbox, viewbox)
        # Clamp to [0,1] for safety
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding='utf-8')


def _render_svg(node_exec: str, renderer_js: Path, bpmn_path: Path, svg_path: Path, meta_path: Path) -> None:
    cmd = [node_exec, str(renderer_js), str(bpmn_path), str(svg_path), str(meta_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Render failed for {bpmn_path}: {result.stderr.strip()}")


def _render_svg_batch(node_exec: str, renderer_js: Path, tasks: List[Dict[str, str]], tmp_dir: Path) -> None:
    if not tasks:
        return
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = tmp_dir / "render_tasks.json"
    tasks_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [node_exec, str(renderer_js), "--batch", str(tasks_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr.strip()
        print(f"[render] batch failed, falling back to single renders. error={err}")
        for t in tasks:
            try:
                _render_svg(node_exec, renderer_js, Path(t['input']), Path(t['outputSvg']), Path(t['outputMeta']))
            except Exception as exc:
                print(f"[render] failed single render for {t.get('input')}: {exc}")


def _svg_to_png(svg_path: Path, png_path: Path, viewbox: Tuple[float, float, float, float]) -> None:
    if cairosvg is None:
        raise RuntimeError('cairosvg is not installed; please pip install cairosvg')
    _, _, vb_w, vb_h = viewbox
    width = int(math.ceil(vb_w))
    height = int(math.ceil(vb_h))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid viewbox size: {viewbox}")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=width,
        output_height=height,
    )


def _collect_bpmn_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob('*.bpmn') if p.is_file()])


def _group_key(path: Path) -> str:
    # Group by language + scenario folder
    parts = path.parts
    if 'BPMN for Research' in parts:
        idx = parts.index('BPMN for Research')
        if len(parts) > idx + 2:
            return os.path.join(parts[idx + 1], parts[idx + 2])
    return str(path.parent)


def _split_groups(paths: List[Path], seed: int) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in paths:
        groups.setdefault(_group_key(p), []).append(p)

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    if len(keys) < 3:
        # Fallback to file-level split if grouping is too small
        all_paths = paths[:]
        random.Random(seed).shuffle(all_paths)
        n = len(all_paths)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        return {
            'train': all_paths[:n_train],
            'val': all_paths[n_train:n_train + n_val],
            'test': all_paths[n_train + n_val:],
        }

    n = len(keys)
    n_train = max(1, int(n * 0.8))
    n_val = max(1, int(n * 0.1))

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])

    return {
        'train': [p for k in train_keys for p in groups[k]],
        'val': [p for k in val_keys for p in groups[k]],
        'test': [p for k in test_keys for p in groups[k]],
    }


def build_dataset(
    bpmn_root: Path,
    out_root: Path,
    node_exec: str,
    renderer_js: Path,
    seed: int,
    max_files: Optional[int],
    batch_size: int,
) -> None:
    bpmn_files = _collect_bpmn_files(bpmn_root)
    if max_files is not None:
        bpmn_files = bpmn_files[:max_files]

    splits = _split_groups(bpmn_files, seed=seed)

    (out_root / 'images').mkdir(parents=True, exist_ok=True)
    (out_root / 'labels').mkdir(parents=True, exist_ok=True)
    (out_root / 'tmp').mkdir(parents=True, exist_ok=True)

    (out_root / 'classes.txt').write_text("\n".join(MVP_CLASSES) + "\n", encoding='utf-8')
    _write_data_yaml(out_root)

    manifest = {
        'classes': MVP_CLASSES,
        'splits': {},
        'source_root': str(bpmn_root),
    }

    total_files = sum(len(v) for v in splits.values())
    print(f"[build] BPMN files total: {total_files}")
    print(f"[build] splits: train={len(splits.get('train', []))}, val={len(splits.get('val', []))}, test={len(splits.get('test', []))}")
    print(f"[build] batch_size={batch_size}")

    for split_name, files in splits.items():
        images_dir = out_root / 'images' / split_name
        labels_dir = out_root / 'labels' / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        manifest['splits'][split_name] = []

        entries: List[Dict[str, object]] = []
        skipped_no_shapes = 0
        for i, bpmn_path in enumerate(files):
            rel_id = bpmn_path.with_suffix('').name + f"_{i:04d}"
            svg_path = out_root / 'tmp' / f"{rel_id}.svg"
            meta_path = out_root / 'tmp' / f"{rel_id}.json"
            png_path = images_dir / f"{rel_id}.png"
            label_path = labels_dir / f"{rel_id}.txt"

            shapes = parse_bpmn_labels(bpmn_path)
            if not shapes:
                skipped_no_shapes += 1
                continue

            entries.append({
                'bpmn': bpmn_path,
                'svg': svg_path,
                'meta': meta_path,
                'png': png_path,
                'label': label_path,
                'shapes': shapes,
            })

        print(f"[{split_name}] files={len(files)} usable={len(entries)} skipped_no_shapes={skipped_no_shapes}")

        # Render in batches to avoid per-file browser startup
        rendered_count = 0
        for idx in range(0, len(entries), batch_size):
            chunk = entries[idx:idx + batch_size]
            tasks = [
                {
                    'input': str(e['bpmn']),
                    'outputSvg': str(e['svg']),
                    'outputMeta': str(e['meta']),
                }
                for e in chunk
            ]
            print(f"[{split_name}] render batch {idx // batch_size + 1}/{(len(entries) + batch_size - 1) // batch_size} (size={len(chunk)})")
            _render_svg_batch(node_exec, renderer_js, tasks, out_root / 'tmp')

            for e in chunk:
                _, inner_viewbox, text_boxes = _load_viewbox(Path(e['meta']))
                _svg_to_png(Path(e['svg']), Path(e['png']), inner_viewbox)
                shapes = list(e['shapes'])
                for tb in text_boxes:
                    shapes.append(ShapeLabel(class_name='text', bbox=tb))
                _write_yolo_labels(Path(e['label']), shapes, inner_viewbox)

                manifest['splits'][split_name].append({
                    'bpmn': str(e['bpmn']),
                    'image': str(e['png']),
                    'label': str(e['label']),
                })
                rendered_count += 1

            print(f"[{split_name}] rendered={rendered_count}/{len(entries)}")

    (out_root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[build] done. manifest={out_root / 'manifest.json'}")


def _write_data_yaml(out_root: Path) -> None:
    lines = [
        f"path: {out_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx, name in enumerate(MVP_CLASSES):
        lines.append(f"  {idx}: {name}")
    (out_root / 'data.yaml').write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bpmn-root', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--node', default='node')
    ap.add_argument('--renderer-js', default=str(Path(__file__).parent / 'render_bpmn_svg.js'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-files', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=25)
    args = ap.parse_args()

    build_dataset(
        bpmn_root=Path(args.bpmn_root),
        out_root=Path(args.out_root),
        node_exec=args.node,
        renderer_js=Path(args.renderer_js),
        seed=args.seed,
        max_files=args.max_files,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
