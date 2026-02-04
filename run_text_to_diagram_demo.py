from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from text_to_diagram.orchestrator import run_text_to_diagram_use_case


DEFAULT_PROMPT_RU = """
Сформируй процесс строго по описанию ниже. Ничего не обобщай и не заменяй на шаблонные фразы.

Участники:
- Клиент
- Оператор
- Отдел безопасности

Шаги:
1) Клиент отправляет заявку на подключение услуги.
2) Оператор проверяет полноту данных.
3) Если данных недостаточно:
   - Оператор запрашивает уточнение у клиента.
   - После получения уточнения вернуться к шагу 2.
4) Если данные корректны:
   - Оператор передает заявку в отдел безопасности.
5) Отдел безопасности принимает решение:
   - Одобрить
   - Отклонить
6) Если одобрено:
   - Оформить договор.
   - Отправить клиенту подтверждение.
7) Если отклонено:
   - Отправить клиенту уведомление с причиной отказа.
8) После отправки любого уведомления процесс завершается.

Требования:
- Сохрани формулировки шагов близко к тексту.
- Добавь явные ветки "если/иначе".
- Не используй названия вида "Шаг 1", "n1", "обработка данных" вместо бизнес-смысла.
- Если есть роли, используй lanes по ролям.
""".strip()


def _pick_model(model_arg: str | None) -> str:
    if model_arg:
        return model_arg
    candidates = sorted(Path("narrator").glob("*.gguf"))
    if not candidates:
        raise RuntimeError("Не найдены .gguf модели в папке narrator/")
    # Для split-модели берем первую часть.
    for c in candidates:
        if "00001-of-" in c.name:
            return str(c)
    return str(candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск text->diagram пайплайна с готовым промптом.")
    parser.add_argument("--prompt", type=str, default=None, help="????? ??????? ??? ?????????")
    parser.add_argument("--model-path", type=str, default=None, help="Путь к GGUF модели")
    parser.add_argument("--artifact-type", type=str, default="bpmn", choices=["mermaid", "bpmn", "plantuml"])
    parser.add_argument("--image-format", type=str, default="png", choices=["png", "jpg"])
    parser.add_argument("--output-dir", type=str, default="results/text_to_diagram_demo")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    model_path = _pick_model(args.model_path)
    prompt = args.prompt.strip() if isinstance(args.prompt, str) and args.prompt.strip() else DEFAULT_PROMPT_RU
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out = run_text_to_diagram_use_case(
        source_text=prompt,
        runtime_overrides={
            "model_path": model_path,
            "n_threads": args.threads,
            "max_tokens": 2200,
            "n_ctx": 4096,
            "temperature": 0.0,
        },
        llm_cfg_overrides={
            "max_reasks": 2,
            "max_input_chars": 4000,
            "quality_gate_enabled": True,
        },
        render={
            "enabled": True,
            "artifact_type": args.artifact_type,
            "image_format": args.image_format,
        },
    )

    (save_dir / "diagram.mmd").write_text(out["artifacts"]["mermaid_mmd"] or "", encoding="utf-8")
    (save_dir / "diagram.bpmn").write_text(out["artifacts"]["bpmn_xml"] or "", encoding="utf-8")
    (save_dir / "diagram.puml").write_text(out["artifacts"]["plantuml_puml"] or "", encoding="utf-8")
    (save_dir / "result_meta.json").write_text(json.dumps(out["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
    (save_dir / "result_issues.json").write_text(json.dumps(out["issues"], ensure_ascii=False, indent=2), encoding="utf-8")

    png_b64 = out["artifacts"]["image_png_base64"]
    if png_b64:
        (save_dir / "diagram.png").write_bytes(base64.b64decode(png_b64))
    jpg_b64 = out["artifacts"]["image_jpg_base64"]
    if jpg_b64:
        (save_dir / "diagram.jpg").write_bytes(base64.b64decode(jpg_b64))

    print("STATUS:", out["status"])
    print("MODEL:", model_path)
    print("TOTAL_DURATION_MS:", out["meta"].get("total_duration_ms"))
    print("FIRST_FAIL:", out["meta"].get("first_fail"))
    print("STAGE_DURATIONS_MS:", out["meta"].get("stage_durations_ms"))
    print("ISSUES:", len(out["issues"]))
    for item in out["issues"][:10]:
        print("-", item.get("code"), ":", item.get("message"))
    print("Saved to:", save_dir.resolve())


if __name__ == "__main__":
    main()

