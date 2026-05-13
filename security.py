import re
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_CELL_INJECTION_PATTERNS = [
    # Переопределение роли (EN)
    r"ignore\s+(previous|all|above|prior)\s+instructions?",
    r"forget\s+(previous|all|above|prior|your)\s+instructions?",
    r"disregard\s+(previous|all|above|prior)\s+instructions?",
    r"you\s+are\s+now\s+(?:a|an|the)\s+",
    r"act\s+as\s+(?:a|an|the)\s+",
    r"roleplay\s+as",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"new\s+persona",
    r"system\s*prompt\s*:",
    r"\[\s*system\s*\]",
    r"<\s*system\s*>",
    # Вытащить системный промпт (EN)
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"show\s+me\s+(your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?(system\s+)?instructions?",
    r"what\s+are\s+your\s+instructions?",
    # Переопределение роли (RU)
    r"игнорируй\s+(предыдущие|все|прошлые)\s+инструкции",
    r"забудь\s+(все\s+)?инструкции",
    r"ты\s+теперь\s+(?:являешься|это)",
    r"действуй\s+как",
    r"притворись",
    r"покажи\s+(системный\s+)?промпт",
    r"раскрой\s+(свои\s+)?инструкции",
    # Маркеры форматирования LLM
    r"###\s*instruction",
    r"###\s*system",
    r"<\s*\|?\s*(?:im_start|im_end|endoftext)\s*\|?\s*>",
    r"\[INST\]",
    r"\[/INST\]",
]

_CELL_PATTERNS_COMPILED = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in _CELL_INJECTION_PATTERNS
]

_MAX_CELL_LENGTH = 500
_MAX_SUSPICIOUS_RATIO = 0.01  # 1% ячеек


def _is_suspicious_cell(value: str) -> bool:
    if len(value) > _MAX_CELL_LENGTH:
        return True
    return any(p.search(value) for p in _CELL_PATTERNS_COMPILED)


def sanitize_dataframe(df: pd.DataFrame) -> tuple:
    warnings_list = []
    df = df.copy()
    total_str_cells = 0
    suspicious_count = 0

    for col in df.select_dtypes(include="object").columns:
        for idx in df.index:
            raw = df.at[idx, col]
            if not isinstance(raw, str):
                continue
            total_str_cells += 1
            if _is_suspicious_cell(raw):
                suspicious_count += 1
                warnings_list.append(
                    f"Подозрительная ячейка [{idx}, '{col}']: {raw[:80]!r} → заменена"
                )
                df.at[idx, col] = f"[УДАЛЕНО: подозрительное содержимое, строка {idx}, колонка '{col}']"

    if total_str_cells > 0:
        ratio = suspicious_count / total_str_cells
        if ratio > _MAX_SUSPICIOUS_RATIO:
            raise ValueError(
                f"Файл отклонён: {suspicious_count}/{total_str_cells} "
                f"ячеек ({ratio:.1%}) содержат инструкции — возможна атака через данные."
            )

    return df, warnings_list

_BLOCKED_CODE_PATTERNS = [
    # Опасные импорты
    r"\bimport\s+os\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+shutil\b",
    r"\bimport\s+socket\b",
    r"\bimport\s+requests?\b",
    r"\bimport\s+urllib\b",
    r"\bimport\s+http\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+subprocess\b",
    r"\bfrom\s+socket\b",
    r"__import__\s*\(",
    # Произвольное выполнение кода
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bcompile\s*\(",
    # Файловая система
    r"\bopen\s*\(",
    r"\.unlink\s*\(",
    r"\.rmdir\s*\(",
    r"\.remove\s*\(",
    # Сеть
    r"\bsocket\s*\.",
    r"\brequests?\s*\.",
    r"\burllib\s*\.",
    # Переменные окружения
    r"os\.environ",
    r"os\.getenv",
    # Интроспекция / побег из sandbox
    r"__builtins__",
    r"__globals__",
    r"__subclasses__",
    r"\bglobals\s*\(",
    r"\blocals\s*\(",
]

_BLOCKED_CODE_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in _BLOCKED_CODE_PATTERNS
]

_MAX_CODE_LENGTH = 4000


def validate_code(code: str) -> None:
    if len(code) > _MAX_CODE_LENGTH:
        raise ValueError(
            f"Код слишком длинный ({len(code)} символов, максимум {_MAX_CODE_LENGTH}). "
            "Возможна попытка переполнения контекста."
        )
    for pattern in _BLOCKED_CODE_COMPILED:
        if pattern.search(code):
            raise ValueError(
                f"Запрещённая конструкция в коде: `{pattern.pattern}`. "
                "Разрешены только операции с pandas и numpy."
            )

_MAX_SUMMARY_LEN  = 1000
_MAX_FIELD_LEN    = 500
_MAX_METRICS      = 12
_MAX_INSIGHTS     = 10
_MAX_CHARTS       = 6
_MAX_RECS         = 8
_MAX_CHART_POINTS = 100


def _truncate(value: Any, max_len: int, field: str) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        logger.warning("Поле '%s' обрезано (%d → %d символов)", field, len(value), max_len)
        return value[:max_len] + "…"
    return value


def validate_agent_response(result: dict) -> dict:
    """
    Валидирует и санирует финальный JSON от агента.
    Возвращает очищенный словарь.
    """
    if not isinstance(result, dict):
        raise ValueError("Ответ агента не является словарём.")
    if "raw" in result:
        return result

    cleaned = {}

    cleaned["summary"] = _truncate(result.get("summary", ""), _MAX_SUMMARY_LEN, "summary")

    metrics = result.get("metrics", []) if isinstance(result.get("metrics"), list) else []
    cleaned["metrics"] = [
        {
            "label": _truncate(str(m.get("label", "")), _MAX_FIELD_LEN, "metric.label"),
            "value": _truncate(str(m.get("value", "")), _MAX_FIELD_LEN, "metric.value"),
            "trend": m.get("trend", "neutral") if m.get("trend") in ("up","down","neutral") else "neutral",
        }
        for m in metrics[:_MAX_METRICS] if isinstance(m, dict)
    ]

    insights = result.get("insights", []) if isinstance(result.get("insights"), list) else []
    cleaned["insights"] = [
        {
            "title":       _truncate(str(i.get("title", "")),       _MAX_FIELD_LEN, "insight.title"),
            "description": _truncate(str(i.get("description", "")), _MAX_FIELD_LEN, "insight.description"),
            "severity":    i.get("severity","low") if i.get("severity") in ("high","medium","low") else "low",
        }
        for i in insights[:_MAX_INSIGHTS] if isinstance(i, dict)
    ]

    charts = result.get("charts", []) if isinstance(result.get("charts"), list) else []
    cleaned["charts"] = []
    for ch in charts[:_MAX_CHARTS]:
        if not isinstance(ch, dict):
            continue
        labels = ch.get("labels", []) if isinstance(ch.get("labels"), list) else []
        values = ch.get("values", []) if isinstance(ch.get("values"), list) else []
        labels = labels[:_MAX_CHART_POINTS]
        values = values[:_MAX_CHART_POINTS]
        safe_values = []
        for v in values:
            try:
                safe_values.append(float(v))
            except (TypeError, ValueError):
                safe_values.append(0.0)
        chart_type = ch.get("type", "bar")
        if chart_type not in ("bar","line","pie","scatter","histogram"):
            chart_type = "bar"
        cleaned["charts"].append({
            "type":    chart_type,
            "title":   _truncate(str(ch.get("title","")),   _MAX_FIELD_LEN, "chart.title"),
            "x_label": _truncate(str(ch.get("x_label","")), _MAX_FIELD_LEN, "chart.x_label"),
            "y_label": _truncate(str(ch.get("y_label","")), _MAX_FIELD_LEN, "chart.y_label"),
            "labels":  [str(l)[:200] for l in labels],
            "values":  safe_values,
        })

    recs = result.get("recommendations", []) if isinstance(result.get("recommendations"), list) else []
    cleaned["recommendations"] = [
        _truncate(str(r), _MAX_FIELD_LEN, "recommendation")
        for r in recs[:_MAX_RECS] if isinstance(r, str)
    ]

    return cleaned
