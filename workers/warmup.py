from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workers.image_to_text_pipeline import run_image_to_text_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up ML models used by image_to_text pipeline.")
    parser.add_argument(
        "--sample-image",
        default="/app/preprocanddetect/diagram1.jpg",
        help="Path to sample image used for warmup.",
    )
    parser.add_argument(
        "--narrator-mode",
        choices=["text", "table"],
        default="table",
        help="Narrator mode for warmup run.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    sample_path = Path(args.sample_image)
    if not sample_path.exists():
        print(f"[warmup] skipped: sample image not found: {sample_path}")
        return

    started = time.perf_counter()
    print(f"[warmup] started, sample={sample_path}")
    try:
        result = await run_image_to_text_pipeline(str(sample_path), narrator_mode=args.narrator_mode)
        elapsed = time.perf_counter() - started
        print(
            "[warmup] done "
            f"in {elapsed:.1f}s, "
            f"ocr_engine={result.get('ocr_engine', 'unknown')}, "
            f"narrator_status={result.get('narrator_status', 'unknown')}"
        )
    except Exception as exc:
        # Warmup failure should not prevent worker startup.
        elapsed = time.perf_counter() - started
        print(f"[warmup] failed in {elapsed:.1f}s: {exc}")


if __name__ == "__main__":
    asyncio.run(_main())
