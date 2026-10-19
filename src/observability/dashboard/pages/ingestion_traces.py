"""Browse ingestion history and measured phase durations."""

import streamlit as st

from observability.dashboard.services.trace_service import TraceService


st.title("摄取追踪")
st.caption("查看摄取记录、运行结果与主阶段耗时。")
st.button("刷新追踪")
settings = st.session_state["_rag_config_service"].settings
if not settings.observability.get("trace_enabled", True):
    st.info("当前未开启追踪，以下仅展示已有历史记录。")
try:
    service = TraceService.from_settings(settings)
    traces = service.load_traces("ingestion")
except (OSError, UnicodeError, TypeError, ValueError):
    st.error("无法读取追踪文件，请检查 observability.trace_file 路径、编码和访问权限。")
    st.stop()
if service.invalid_line_count:
    st.warning(f"已跳过 {service.invalid_line_count} 行无效或尚未写完的追踪记录，可稍后刷新。")
if not traces:
    st.info("暂无摄取追踪。完成一次摄取后点击刷新追踪。")
    st.stop()

labels = {"completed": "成功", "failed": "失败", "skipped": "跳过", "running": "进行中", "unknown": "未知"}
st.dataframe([{"开始时间": trace.started_at, "来源路径": trace.detail("source_path"),
               "集合": trace.detail("collection"), "状态": labels[trace.status],
               "总耗时 (ms)": trace.total_elapsed_ms, "追踪 ID": trace.trace_id} for trace in traces],
             hide_index=True)
selected = st.selectbox("选择摄取记录", range(len(traces)),
                        format_func=lambda index: f"{traces[index].started_at} · {traces[index].detail('source_path')} · {labels[traces[index].status]}")
trace = traces[selected]
st.subheader("追踪详情")
st.text(f"追踪 ID：{trace.trace_id}")
st.text(f"来源：{trace.detail('source_path') or '未知'}")
st.caption(f"状态：{labels[trace.status]} · 总耗时：{trace.total_elapsed_ms:.2f} ms")
rows = service.ingestion_timings(trace)
if rows:
    st.vega_lite_chart(spec={
        "data": {"values": rows}, "mark": "bar",
        "encoding": {
            "y": {"field": "stage", "type": "nominal", "sort": [row["stage"] for row in rows], "title": "阶段"},
            "x": {"field": "elapsed_ms", "type": "quantitative", "title": "耗时 (ms)"},
            "color": {"field": "status", "type": "nominal", "title": "阶段状态"},
            "tooltip": [{"field": "stage"}, {"field": "elapsed_ms", "format": ".2f"}, {"field": "status"}],
        },
    })
    st.caption("仅展示有计时数据的主阶段；内部子阶段与主阶段可能重叠，未重复累加。")
else:
    st.info("该记录没有主阶段耗时数据。")
for index, stage in enumerate(trace.stages, start=1):
    with st.expander(f"{index}. {stage['stage']}"):
        st.json(stage)
