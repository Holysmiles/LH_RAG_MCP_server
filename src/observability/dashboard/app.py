"""Streamlit entry point: shared settings and six-page navigation."""

import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st

from core.settings import SettingsError
from observability.dashboard.services.config_service import ConfigService, DEFAULT_SETTINGS_PATH


st.set_page_config(page_title="RAG 管理平台", page_icon="📚", layout="wide")
settings_path = os.environ.get("RAG_SETTINGS_PATH", str(DEFAULT_SETTINGS_PATH))
try:
    service = ConfigService(settings_path)
except (OSError, SettingsError, ValueError):
    st.error("无法加载配置，请检查配置文件路径、YAML 格式和必填字段。")
    st.stop()

st.session_state["_rag_config_service"] = service
with st.sidebar:
    st.title("知识库管理")
    st.caption("配置、数据与检索追踪")
    st.caption(f"配置文件：{service.path}")

pages = [
    st.Page("pages/overview.py", title="系统总览", default=True),
    st.Page("pages/data_browser.py", title="数据浏览器"),
    st.Page("pages/ingestion_manager.py", title="摄取管理"),
    st.Page("pages/ingestion_traces.py", title="摄取追踪"),
    st.Page("pages/query_traces.py", title="查询追踪"),
    st.Page("pages/evaluation_panel.py", title="评估面板"),
]
st.navigation(pages).run()
