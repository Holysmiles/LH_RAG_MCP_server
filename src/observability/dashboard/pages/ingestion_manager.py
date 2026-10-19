"""Upload a PDF, observe ingestion progress and delete indexed documents."""

import streamlit as st

from observability.dashboard.services.data_service import DataService
from observability.dashboard.services.ingestion_service import IngestionService


st.title("摄取管理")
if "_ingestion_message" in st.session_state:
    st.success(st.session_state.pop("_ingestion_message"))
settings = st.session_state["_rag_config_service"].settings
service = IngestionService(settings)
try:
    documents = DataService.from_settings(settings).list_documents()
except Exception:
    st.error("无法读取文档列表，请检查存储配置和数据库状态。")
    st.stop()

st.subheader("上传 PDF")
uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"])
collections = sorted({document.collection for document in documents})
choice = st.selectbox("目标集合", [None, *collections],
                      format_func=lambda value: "新建集合" if value is None else value)
collection = st.text_input("集合名称", value="default") if choice is None else choice
force = st.checkbox("强制重新摄取", help="已成功处理的相同内容默认跳过。")
if st.button("开始摄取", disabled=uploaded is None):
    try:
        service.normalize_collection(collection)
    except ValueError:
        st.error("集合名称须以字母或数字开头，仅支持字母、数字、点、下划线和连字符。")
    else:
        progress = st.progress(0, text="准备摄取")

        def on_progress(stage, completed, total):
            labels = {"integrity": "完整性检查", "load": "解析文件", "split": "文本切分",
                      "transform": "内容处理", "embed": "向量编码", "upsert": "写入索引"}
            progress.progress(completed / total, text=f"{labels.get(stage, stage)} · {completed}/{total}")

        try:
            result = service.ingest_upload(uploaded.name, uploaded.getvalue(), collection,
                                           force=force, on_progress=on_progress)
        except ValueError:
            st.error("无法摄取，请检查 PDF 文件、集合名称和存储配置。")
        except Exception:
            st.error("摄取失败，请检查文件、模型服务和存储状态后重试。")
        else:
            if result.skipped:
                st.info("该文件已成功摄取，本次已跳过。需要重新处理时可勾选强制重新摄取。")
            else:
                st.success(f"摄取完成：{result.chunk_count} 个文本块，{result.image_count} 张图片。")

st.subheader("已摄取文档")
st.caption("删除会清理向量、BM25、图片和去重索引，源文件保留。")
try:
    documents = DataService.from_settings(settings).list_documents()
except Exception:
    st.error("无法刷新文档列表，请检查存储状态。")
    st.stop()
if not documents:
    st.info("尚无已摄取文档。")
for document in documents:
    with st.container(border=True):
        st.text(document.source_path)
        st.caption(f"集合：{document.collection} · 文本块：{document.chunk_count} · 图片：{document.image_count}")
        if st.button("删除文档", key=f"delete:{document.collection}:{document.source_path}:{document.document_id}"):
            try:
                result = service.delete_document(document.source_path, document.collection)
            except Exception:
                st.error("删除失败，请检查存储状态后重试。")
            else:
                if result.success:
                    st.session_state["_ingestion_message"] = "文档索引已删除，源文件已保留。"
                    st.rerun()
                else:
                    st.error("删除未完成，请检查存储状态后重试。失败阶段：" + ", ".join(result.errors))
