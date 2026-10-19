"""System configuration cards and persisted index counts."""

from pathlib import Path

import streamlit as st


service = st.session_state["_rag_config_service"]
settings = service.settings
st.title("系统总览")
st.caption("查看当前组件配置与已摄取数据。")

st.subheader("组件配置")
columns = st.columns(3)
for index, card in enumerate(service.component_configs()):
    with columns[index % 3]:
        with st.container(border=True):
            st.markdown(f"**{card['name']}** · {card['status']}")
            st.text(card["provider"])
            if card["model"]:
                st.text(card["model"])
            if card["details"]:
                st.caption(card["details"])

st.subheader("数据资产")
stats = None
if settings.vector_store.get("provider") != "chroma":
    st.info("当前存储后端暂未提供总览统计。")
else:
    directory = Path(settings.vector_store.get("persist_directory", "data/db/chroma")).expanduser()
    if not (directory / "chroma.sqlite3").is_file():
        stats = {"document_count": 0, "chunk_count": 0, "collection_count": 0}
        st.info("尚未发现已摄取数据。完成首次摄取后，重新加载页面即可查看统计。")
    else:
        try:
            from libs.vector_store.chroma_store import ChromaStore

            stats = ChromaStore(settings.vector_store, create_if_missing=False).get_collection_stats()
        except Exception:
            st.warning("暂时无法读取数据统计，请检查存储路径、集合配置和数据库状态。")

for column, key, label in zip(st.columns(3), ("document_count", "chunk_count", "collection_count"),
                              ("已摄取文档", "文本块", "集合")):
    column.metric(label, stats[key] if stats is not None else "—")
