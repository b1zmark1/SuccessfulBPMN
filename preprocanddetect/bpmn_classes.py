"""
Единый источник правды для BPMN-классов YOLOX'а.

Раньше эти множества дублировались в 3 файлах (label_res, label_aggregate,
ocr_full_sweep) — при добавлении/переименовании класса приходилось править
все копии. Теперь все модули, которым нужны эти константы, импортируют их
отсюда.

Замечание: `label_assign.py` имеет свои подмножества STRICT_NODE_CLASSES /
LOOSE_NODE_CLASSES, которые описывают стратегию матчинга (а не классификацию),
поэтому остаются локальными в label_assign.py.
"""

from __future__ import annotations

from typing import Set


# Все «узлы процесса» BPMN, которые YOLOX может задетектить.
# Используется как фильтр «оставить только узлы» в label_res / label_aggregate.
NODE_CLASSES: Set[str] = {
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

# Рёбра (последовательности).
EDGE_CLASSES: Set[str] = {
    "sequence_flow",
}

# Контейнеры (содержат другие узлы) — pool/lane/subprocess.
# При crop-based OCR их нарезать не нужно: текст внутри уже захватится
# crop'ами вложенных task'ов, иначе получим дубли.
CONTAINER_CLASSES: Set[str] = {
    "pool",
    "lane",
    "subprocess",
}

# Что crop'ить под OCR в `ocr_full_sweep.py`. Всё что узлы процесса,
# кроме контейнеров.
SHAPE_CLASSES_FOR_OCR: Set[str] = NODE_CLASSES - CONTAINER_CLASSES
