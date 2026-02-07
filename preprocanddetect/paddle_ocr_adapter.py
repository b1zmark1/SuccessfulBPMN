from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_paddle_ocr_blocks(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    blocks = raw.get("blocks", []) if isinstance(raw, dict) else []
    if not isinstance(blocks, list):
        return []
    out: List[Dict[str, Any]] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        bbox = b.get("bbox_xyxy")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        text = b.get("text")
        score = b.get("confidence") if b.get("confidence") is not None else b.get("score", 1.0)
        out.append(
            {
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "text": str(text) if isinstance(text, str) else "",
                "score": float(score) if isinstance(score, (int, float)) else 1.0,
            }
        )
    return out


def blocks_to_text_detections(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dets: List[Dict[str, Any]] = []
    for b in blocks:
        bbox = b.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        dets.append(
            {
                "class_id": -1,
                "class_name": "text",
                "score": float(b.get("score", 1.0)),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "source": "paddleocr",
                "text": b.get("text", ""),
            }
        )
    return dets
