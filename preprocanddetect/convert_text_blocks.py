from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class XYXY:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, w: int, h: int) -> "XYXY":
        x1 = max(0, min(self.x1, w - 1))
        y1 = max(0, min(self.y1, h - 1))
        x2 = max(0, min(self.x2, w - 1))
        y2 = max(0, min(self.y2, h - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return XYXY(x1, y1, x2, y2)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def _bbox_to_xyxy(bbox: Any) -> XYXY:
    """
    Поддерживаем самые частые форматы:
    1) [x1,y1,x2,y2]
    2) [x,y,w,h] (эвристика)
    3) [[x,y],[x,y],[x,y],[x,y]] (полигон)
    4) [x1,y1,x2,y2,x3,y3,x4,y4] (полигон плоским списком)
    5) {"x1":...,"y1":...,"x2":...,"y2":...} или {"x":...,"y":...,"w":...,"h":...}
    """
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            return XYXY(int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"]))
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            return XYXY(int(x), int(y), int(x + w), int(y + h))
        raise ValueError(f"Unsupported bbox dict keys: {sorted(bbox.keys())}")

    if isinstance(bbox, list) or isinstance(bbox, tuple):
        if len(bbox) == 4 and all(_is_number(v) for v in bbox):
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            # эвристика: если похоже на xyxy — оставляем; иначе считаем это xywh
            if x2 >= x1 and y2 >= y1:
                return XYXY(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            return XYXY(int(round(x1)), int(round(y1)), int(round(x1 + x2)), int(round(y1 + y2)))

        if len(bbox) == 8 and all(_is_number(v) for v in bbox):
            pts = [(float(bbox[i]), float(bbox[i + 1])) for i in range(0, 8, 2)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return XYXY(int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys))))

        if len(bbox) >= 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in bbox):
            pts = [(float(p[0]), float(p[1])) for p in bbox]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return XYXY(int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys))))

    raise ValueError(f"Unsupported bbox format: {type(bbox)} {bbox!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    d = json.loads(inp.read_text(encoding="utf-8"))
    blocks = d.get("blocks", [])
    if not isinstance(blocks, list):
        raise TypeError("JSON has no 'blocks' list")

    converted = 0
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if "bbox_model_xyxy" in b or "bbox_geom_xyxy" in b:
            continue
        if "bbox" not in b:
            continue

        xyxy = _bbox_to_xyxy(b["bbox"])
        b["bbox_model_xyxy"] = [xyxy.x1, xyxy.y1, xyxy.x2, xyxy.y2]
        b["bbox_geom_xyxy"] = [float(xyxy.x1), float(xyxy.y1), float(xyxy.x2), float(xyxy.y2)]
        converted += 1

    d["converted_blocks"] = converted
    out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[convert_text_blocks] wrote: {out} (converted={converted}/{len(blocks)})")


if __name__ == "__main__":
    main()
