import plotly.express as px
import streamlit as st

def render_chart(chart: dict):
    t = chart.get("type", "bar")
    title  = chart.get("title", "")
    labels = chart.get("labels", [])
    values = chart.get("values", [])
    if not labels or not values:
        return
    try:
        kw = dict(title=title, labels={"x": chart.get("x_label",""), "y": chart.get("y_label","")})
        if t == "pie":
            fig = px.pie(names=labels, values=values, title=title)
        elif t == "line":
            fig = px.line(x=labels, y=values, **kw)
        elif t == "scatter":
            fig = px.scatter(x=labels, y=values, **kw)
        elif t == "histogram":
            fig = px.histogram(x=values, title=title, nbins=20)
        else:
            fig = px.bar(x=labels, y=values, **kw)
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Не удалось построить график «{title}»: {e}")
