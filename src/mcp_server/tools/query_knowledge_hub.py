"""Expose the shared retrieval pipeline as a lazily initialized MCP tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.response.response_builder import MCPResponse, ResponseBuilder
from core.response.multimodal_assembler import MultimodalAssembler
from core.settings import Settings
from core.trace.trace_collector import query_trace

if TYPE_CHECKING:
    from core.query_engine.hybrid_search import HybridSearch
    from core.query_engine.reranker import Reranker
    from mcp_server.protocol_handler import ProtocolHandler


INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "pattern": r"[A-Za-z\d\u4e00-\u9fff]",
                  "description": "Search query containing an English, Chinese, or numeric term."},
        "top_k": {"type": "integer", "minimum": 1,
                  "description": "Final result count; defaults to retrieval.fusion_top_k."},
        "collection": {"type": "string", "minLength": 1, "pattern": r"\S",
                       "description": "Optional logical collection filter."},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def query_knowledge_hub(
    query: str, top_k: int | None = None, collection: str | None = None, *,
    settings: Settings, hybrid_search: HybridSearch | None = None,
    reranker: Reranker | None = None,
) -> MCPResponse:
    """Search, rerank all candidates, then number the selected citations.

    Production calls reopen persisted indexes for each query so an ingestion
    completed during the session is visible. Providers are never opened by
    tools/list or by argument validation in the protocol handler.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    query = query.strip()
    selected_k = settings.retrieval.get("fusion_top_k", 10) if top_k is None else top_k
    if isinstance(selected_k, bool) or not isinstance(selected_k, int) or selected_k < 1:
        raise ValueError("top_k must be a positive integer")
    if collection is not None:
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError("collection must be a non-empty string")
        collection = collection.strip()

    # The query-engine package imports database and PDF dependencies; defer it
    # until the first valid call, keeping the stdio handshake lightweight.
    from core.query_engine import DenseRetriever, Fusion, HybridSearch, QueryProcessor, Reranker, SparseRetriever
    from libs.vector_store.vector_store_factory import VectorStoreFactory

    processor = QueryProcessor()
    processor.process(query)
    reranker = reranker if reranker is not None else Reranker(settings)
    candidate_k = max(selected_k, settings.retrieval.get("fusion_top_k", 10))
    if reranker.enabled and reranker.provider != "none":
        top_m = settings.rerank.get("top_m", 30)
        if isinstance(top_m, bool) or not isinstance(top_m, int) or top_m < 1:
            raise ValueError("rerank.top_m must be a positive integer")
        candidate_k = max(candidate_k, top_m)
    if hybrid_search is None:
        store = VectorStoreFactory.create(settings)
        hybrid_search = HybridSearch(
            settings, processor, DenseRetriever(settings, vector_store=store),
            SparseRetriever(settings, vector_store=store), Fusion(settings),
        )
    with query_trace(settings) as trace:
        candidates = hybrid_search.search(
            query, top_k=candidate_k,
            filters={"collection": collection} if collection is not None else None, trace=trace,
        )
        results = reranker.rerank(query, candidates, top_k=selected_k, trace=trace)
        assembler = MultimodalAssembler(
            image_root=settings.ingestion.get("image_root", "data/images"),
            image_index_db_path=settings.ingestion.get("image_index_db_path", "data/db/image_index.db"),
        )
        return ResponseBuilder(assembler).build(results, query)


def register_query_tool(handler: ProtocolHandler, settings: Settings) -> None:
    def call(arguments):
        return query_knowledge_hub(**arguments, settings=settings)

    handler.register_tool(
        "query_knowledge_hub",
        "Search the knowledge hub with hybrid retrieval and optional reranking; return source citations.",
        INPUT_SCHEMA, call,
    )
