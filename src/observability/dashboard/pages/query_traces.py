"""Search query history and compare retrieval routes and reranking."""

import streamlit as st

from observability.dashboard.services.trace_service import TraceService


st.title("查询追踪")
st.caption("搜索历史查询，比较两路召回结果与重排前后顺序。")
st.button("刷新追踪")
keyword = st.text_input("搜索查询关键词")
settings = st.session_state["_rag_config_service"].settings
if not settings.observability.get("trace_enabled", True):
    st.info("当前未开启追踪，以下仅展示已有历史记录。")
try:
    service = TraceService.from_settings(settings)
    traces = service.load_traces("query")
except (OSError, UnicodeError, TypeError, ValueError):
    st.error("无法读取追踪文件，请检查 observability.trace_file 路径、编码和访问权限。")
    st.stop()
if service.invalid_line_count:
    st.warning(f"已跳过 {service.invalid_line_count} 行无效或尚未写完的追踪记录，可稍后刷新。")
traces = [trace for trace in traces if keyword.strip().casefold() in trace.detail("query").casefold()]
if not traces:
    st.info("暂无匹配的查询追踪。执行查询后刷新，或调整搜索关键词。")
    st.stop()
labels = {"completed": "成功", "degraded": "降级", "failed": "失败", "running": "进行中", "unknown": "未知"}
st.dataframe([{"开始时间": trace.started_at, "查询": trace.detail("query"), "状态": labels[trace.status],
               "总耗时 (ms)": trace.total_elapsed_ms, "追踪 ID": trace.trace_id} for trace in traces], hide_index=True)
by_id = {trace.trace_id: trace for trace in traces}
selected = st.selectbox("选择查询记录", list(by_id),
                        format_func=lambda identity: f"{by_id[identity].started_at} · {by_id[identity].detail('query')}")
trace = by_id[selected]
st.subheader("查询详情")
st.text(trace.detail("query") or "未记录查询文本")
st.text(f"追踪 ID：{trace.trace_id}")
st.caption(f"状态：{labels[trace.status]} · 总耗时：{trace.total_elapsed_ms:.2f} ms")
timings = service.query_timings(trace)
if timings:
    st.vega_lite_chart(spec={"data": {"values": timings}, "mark": "bar", "encoding": {
        "y": {"field": "stage", "type": "nominal", "sort": [row["stage"] for row in timings], "title": "阶段"},
        "x": {"field": "elapsed_ms", "type": "quantitative", "title": "耗时 (ms)"},
        "color": {"field": "status", "type": "nominal", "title": "阶段状态"},
        "tooltip": [{"field": "stage"}, {"field": "elapsed_ms", "format": ".2f"}],
    }})
    st.caption("Dense 与 Sparse 并行执行，阶段耗时不可直接相加为总耗时。")
else:
    st.info("该记录没有阶段耗时数据。")

st.subheader("Dense / Sparse 召回对比")
for column, name, title in zip(st.columns(2), ("dense_retrieval", "sparse_retrieval"), ("Dense", "Sparse")):
    with column:
        st.markdown(f"**{title}**")
        stage = next((stage for stage in trace.stages if stage["stage"] == name), {})
        if stage.get("fallback"):
            st.warning("该召回路由已降级：" + str(stage.get("error_type") or "不可用"))
        results = service.route_results(trace, name)
        if not results:
            st.info("未记录召回结果或本次没有命中。")
        for result in results:
            with st.expander(f"{result['rank']}. {result['chunk_id']} · score={result['score']}"):
                st.text(result["text"])
                st.json(result["metadata"])
st.caption("两路分数使用各自的度量，排序和共同的 Chunk ID 可用于比较。")

st.subheader("重排前后排名")
rerank = next((stage for stage in trace.stages if stage["stage"] == "rerank"), {})
if rerank.get("fallback"):
    st.warning("重排已回退到融合顺序，原因：" + str(rerank.get("reason") or "未知"))
elif rerank.get("reason") in {"disabled", "skipped", "empty_candidates"}:
    st.info("本次未执行模型重排。")
changes = service.rank_changes(trace)
if changes:
    st.dataframe([{"Chunk ID": row["chunk_id"], "重排前": row["before"], "重排后": row["after"],
                   "上升名次": row["change"], "最终入选": row["selected"]} for row in changes], hide_index=True)
else:
    st.info("该记录没有可比较的重排排名快照。旧日志或空候选可能没有此数据。")
with st.expander("全部阶段数据"):
    st.json(trace.stages)
