import pandas as pd
import streamlit as st
import json
import re

from openai import OpenAI

from config import API_KEY, BASE_URL, MODEL
from tools import execute_python
from prompts import SYSTEM_PROMPT
from security import validate_agent_response   # ← Уровень 3

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Streamlit AI Agent"
    }
)

def run_agent(df: pd.DataFrame, filename: str):

    tool_def = {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Выполняет Python-код. Датафрейм доступен как `df`. Используй print() для вывода.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python-код (pandas, numpy)"},
                    "description": {"type": "string"}
                },
                "required": ["code", "description"],
            },
        },
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f'Проанализируй датасет "{filename}". '
                f"Размер: {df.shape[0]} строк × {df.shape[1]} столбцов. "
                f"Колонки: {list(df.columns)}."
            ),
        },
    ]

    steps_placeholder = st.empty()
    steps = []
    tool_calls_count = 0

    def update_steps():
        with steps_placeholder.container():
            for s in steps:
                st.markdown(f"✅ `{s}`")

    for iteration in range(8):

        steps.append(f"Итерация {iteration + 1}: вызов API...")
        update_steps()

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4000,
            tools=[tool_def],
            tool_choice="auto",
            messages=messages,
        )

        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in tool_calls
            ] or None
        })

        text_content = msg.content or ""

        if tool_calls:
            for tc in tool_calls:
                tool_calls_count += 1

                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}

                code = args.get("code", "")
                desc = args.get("description", "")

                steps.append(f"Выполняю код ({tool_calls_count}): {desc[:60]}")
                update_steps()

                result = execute_python(code, df)

                steps.append(f"→ {result[:100].replace(chr(10), ' ')}")
                update_steps()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        if response.choices[0].finish_reason == "stop" or not tool_calls:

            text_clean = text_content.strip()

            if "```json" in text_clean:
                text_clean = text_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in text_clean:
                text_clean = text_clean.split("```")[1].split("```")[0].strip()

            try:
                raw_result = json.loads(text_clean)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", text_clean)
                if m:
                    try:
                        raw_result = json.loads(m.group())
                    except Exception:
                        raw_result = {"raw": text_content}
                else:
                    raw_result = {"raw": text_content}

            # ── Уровень 3: валидируем и санируем ответ агента ──
            safe_result = validate_agent_response(raw_result)
            return safe_result, steps, tool_calls_count

    return {"raw": "Агент не вернул результат"}, steps, tool_calls_count
