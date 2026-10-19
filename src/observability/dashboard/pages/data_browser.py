"""Browse indexed documents, chunks and managed images."""

import streamlit as st

from observability.dashboard.services.data_service import DataService


st.title("数据浏览器")
st.caption("按集合浏览已摄取文档，查看文本块、元数据和关联图片。")
st.button("刷新数据")
try:
    service = DataService.from_settings(st.session_state["_rag_config_service"].settings)
    documents = service.list_documents()
except Exception:
    st.error("无法读取文档索引，请检查存储后端、路径和数据库状态。")
    st.stop()

if not documents:
    st.info("尚无已摄取文档。完成摄取后点击刷新数据。")
    st.stop()

collections = sorted({document.collection for document in documents})
selected_collection = st.selectbox("集合筛选", [None, *collections],
                                   format_func=lambda value: "全部集合" if value is None else value)
visible = [doc for doc in documents if selected_collection is None or doc.collection == selected_collection]
st.dataframe([{"来源路径": doc.source_path, "集合": doc.collection, "文本块数": doc.chunk_count,
               "图片数": doc.image_count, "摄入时间": doc.ingested_at or "未知"} for doc in visible],
             hide_index=True)
selection = st.selectbox("选择文档", range(len(visible)),
                         format_func=lambda index: f"{visible[index].source_path} · {visible[index].collection}")
document = visible[selection]
try:
    detail = service.get_document_detail(document)
except KeyError:
    st.warning("该文档已变更或被删除，请刷新数据。")
    st.stop()
except Exception:
    st.error("无法读取文档详情，请检查索引状态后重试。")
    st.stop()

st.subheader("文档详情")
st.text(f"文档 ID：{document.document_id}")
st.caption(f"{len(detail.chunks)} 个文本块 · {len(detail.images)} 张图片")
images = {image["image_id"]: image for image in detail.images}


def show_image(image_id):
    try:
        path = service.get_image_path(image_id)
        if path is None:
            st.warning(f"图片不可用：{image_id}")
            return
        st.image(str(path), caption=image_id)
    except Exception:
        st.warning(f"图片无法预览：{image_id}")


for index, chunk in enumerate(detail.chunks, start=1):
    with st.expander(f"文本块 {index} · {chunk['id']}"):
        st.text(chunk["text"])
        st.json(chunk["metadata"])
        for image_id in chunk["metadata"].get("image_refs", []):
            if image_id in images:
                show_image(image_id)
            else:
                st.warning(f"关联图片未入库：{image_id}")

if images:
    with st.expander("文档图片"):
        for image_id in images:
            show_image(image_id)
