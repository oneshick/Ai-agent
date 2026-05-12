import streamlit as st
import pandas as pd
import numpy as np

from agent import run_agent
from charts import render_chart
from config import MODEL   # ← ВОТ ЭТОГО НЕ ХВАТАЛО

st.title("ИИ-агент аналитики данных")

with st.sidebar:
    st.header("О приложении")
    st.markdown(f"**Модель:** `{MODEL}`")
    st.markdown("**Провайдер:** qwen")
    st.divider()
    st.markdown("**Как работает агент:**")
    st.markdown(
        "1. Вы загружаете CSV\n"
        "2. LLM получает задачу\n"
        "3. Модель вызывает `execute_python` (tool use)\n"
        "4. Инструмент считает статистику через pandas\n"
        "5. Модель интерпретирует результаты\n"
        "6. Выводятся метрики, графики, инсайты"
    )

uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv","tsv","txt"])

if uploaded_file:
    try:
        sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
        df = pd.read_csv(uploaded_file, sep=sep)
        st.success(f"Файл загружен: **{uploaded_file.name}** — {df.shape[0]} строк × {df.shape[1]} столбцов")

        with st.expander("Предпросмотр данных", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        st.stop()

    if st.button("Запустить анализ ИИ-агентом", type="primary", use_container_width=True):
        st.divider()
        st.subheader("Агент работает...")

        try:
            result, steps, tool_calls = run_agent(df, uploaded_file.name)
        except Exception as e:
            st.error(f"Ошибка агента: {e}")
            st.stop()

        st.success(f"Анализ завершён — выполнено {tool_calls} вызовов инструмента")
        st.divider()

        if result.get("raw"):
            st.subheader("Вывод агента")
            st.text(result["raw"])
            st.stop()

        if result.get("summary"):
            st.subheader("Резюме")
            st.info(result["summary"])

        if result.get("metrics"):
            st.subheader("Ключевые метрики")
            cols = st.columns(min(len(result["metrics"]), 4))
            trend_icon = {"up": "🔺", "down": "🔻", "neutral": "➡️"}

            for i, m in enumerate(result["metrics"]):
                with cols[i % 4]:
                    icon = trend_icon.get(m.get("trend","neutral"), "")
                    st.metric(label=m.get("label",""), value=m.get("value",""), delta=icon or None)

        if result.get("charts"):
            st.subheader("Визуализации")

            charts = result["charts"]

            if len(charts) == 1:
                render_chart(charts[0])
            else:
                for i in range(0, len(charts), 2):
                    c1, c2 = st.columns(2)
                    with c1:
                        render_chart(charts[i])
                    if i + 1 < len(charts):
                        with c2:
                            render_chart(charts[i + 1])

        if result.get("insights"):
            st.subheader("Инсайты")
            sev_map = {"high":("","error"), "medium":("","warning"), "low":("","info")}

            for ins in result["insights"]:
                icon, kind = sev_map.get(ins.get("severity","low"), ("🔵","info"))
                getattr(st, kind)(f"**{icon} {ins.get('title','')}** — {ins.get('description','')}")

        if result.get("recommendations"):
            st.subheader("Рекомендации")
            for i, rec in enumerate(result["recommendations"], 1):
                st.markdown(f"{i}. {rec}")

        with st.expander("Полный JSON-ответ агента", expanded=False):
            st.json(result)

else:
    st.info("Загрузите CSV-файл выше, чтобы начать анализ")

    st.markdown("### Примеры датасетов для теста")

    np.random.seed(42)
    hr_df = pd.DataFrame({
        "отдел": np.random.choice(["Разработка", "Маркетинг", "Продажи", "HR", "Финансы"], 100),
        "возраст": np.random.randint(22, 58, 100),
        "зарплата": np.random.randint(50000, 250000, 100),
        "опыт_лет": np.random.randint(0, 20, 100),
        "оценка": np.random.choice([1, 2, 3, 4, 5], 100, p=[0.05, 0.1, 0.3, 0.4, 0.15]),
        "уволился": np.random.choice(["Да", "Нет"], 100, p=[0.2, 0.8]),
    })

    st.download_button(
        "Скачать демо: HR-данные сотрудников",
        data=hr_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="hr_demo.csv",
        mime="text/csv"
    )