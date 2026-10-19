"""Run golden-set evaluations and inspect successful session reports."""

import json
from uuid import uuid4

import streamlit as st

from observability.dashboard.services.evaluation_service import EvaluationInputError, EvaluationService


st.title("评估面板")
st.caption("选择黄金集和评估后端，查看检索指标与逐查询结果。")
config_service = st.session_state["_rag_config_service"]
settings = config_service.settings
service = EvaluationService(settings)
try:
    test_sets = service.test_sets()
except EvaluationInputError as exc:
    st.error(str(exc))
    st.stop()
labels = {"configured": "按配置", "custom": "Custom", "ragas": "Ragas", "all": "全部（Ragas + Custom）"}
backend = st.selectbox("评估后端", list(labels), format_func=labels.get)
test_set = st.selectbox("黄金测试集", test_sets, format_func=str)
default_top_k = settings.evaluation.get("top_k", settings.retrieval.get("fusion_top_k", 10))
if isinstance(default_top_k, bool) or not isinstance(default_top_k, int) or default_top_k < 1:
    st.error("请将 evaluation.top_k 配置为正整数。")
    st.stop()
top_k = st.number_input("Top-K", min_value=1, value=default_top_k, step=1)
st.caption("Custom 评估检索命中与排名；Ragas 还需要样例的待评答案和模型服务。")
history_key = f"_evaluation_history:{config_service.path}"
history = st.session_state.setdefault(history_key, [])
if st.button("运行评估"):
    try:
        with st.spinner("正在运行评估…"):
            report = service.run(test_set, backend, int(top_k))
    except EvaluationInputError as exc:
        st.error(str(exc))
    except ImportError:
        st.error("评估依赖不可用，请安装项目依赖后重试。")
    except Exception:
        st.error("评估失败，请检查检索索引、样例输入、模型配置和服务状态后重试。")
    else:
        history.insert(0, {"id": str(uuid4()), "backend": backend, "report": report})
        st.success("评估完成。")
if not history:
    st.info("尚无成功评估报告。先摄取黄金集对应的源文档，再点击运行评估。")
    st.stop()

st.subheader("本次会话的评估历史")
st.dataframe([{"完成时间": entry["report"].finished_at, "后端": labels[entry["backend"]],
               "黄金集": entry["report"].test_set_path, "Top-K": entry["report"].top_k,
               **entry["report"].metrics} for entry in history], hide_index=True)
by_id = {entry["id"]: entry for entry in history}
selected = st.selectbox("查看成功报告", list(by_id),
                        format_func=lambda identity: f"{by_id[identity]['report'].finished_at} · {labels[by_id[identity]['backend']]}")
entry = by_id[selected]
report = entry["report"]
st.caption(f"此报告后端：{labels[entry['backend']]} · 黄金集：{report.test_set_path} · Top-K：{report.top_k}")
for column, (name, score) in zip(st.columns(len(report.metrics)), report.metrics.items()):
    column.metric(name, f"{score:.4f}")
st.subheader("逐查询结果")
st.dataframe([{"查询": row["query"], "集合": row["collection"] or "全部", **row["metrics"],
               "召回数": len(row["retrieved_ids"])} for row in report.results], hide_index=True)
for index, row in enumerate(report.results, start=1):
    with st.expander(f"{index}. {row['query']}"):
        st.json(row)
st.download_button("下载 JSON 报告", json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                   file_name="evaluation_report.json", mime="application/json")
