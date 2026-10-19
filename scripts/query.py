"""Query the persisted RAG indexes through the shared Core retrieval pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import RetrievalResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def non_empty_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search the RAG knowledge hub.")
    parser.add_argument("--query", required=True, type=non_empty_text)
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument("--collection", type=non_empty_text)
    parser.add_argument("--verbose", action="store_true", help="Show intermediate rankings.")
    parser.add_argument("--no-rerank", action="store_true", help="Keep the fusion ranking.")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS_PATH)
    return parser


def print_results(title: str, results: Sequence[RetrievalResult]) -> None:
    """Display fusion scores with citation metadata and compact text excerpts."""
    print(f"\n{title} ({len(results)})")
    for rank, item in enumerate(results, start=1):
        source = item.metadata.get("source_path", "未知来源")
        page = item.metadata.get("page", item.metadata.get("page_number", "未知"))
        excerpt = " ".join(item.text.split())
        if len(excerpt) > 200:
            excerpt = excerpt[:200] + "…"
        print(f"{rank}. score={item.score:.6f} | chunk_id={item.chunk_id}")
        print(f"   来源: {source} | 页码: {page}")
        print(f"   {excerpt}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Help and argument errors do not initialize model or database dependencies.
    from core.query_engine import DenseRetriever, Fusion, HybridSearch, QueryProcessor, Reranker, SparseRetriever
    from core.settings import load_settings
    from core.trace.trace_collector import query_trace
    from core.types import RetrievalResult
    from libs.vector_store.vector_store_factory import VectorStoreFactory
    from observability.logger import get_logger

    logger = get_logger("scripts.query")
    try:
        settings = load_settings(args.settings)
        processor = QueryProcessor()
        # Validate searchable text before opening the database or a provider.
        processor.process(args.query)
        store = VectorStoreFactory.create(settings)
        dense = DenseRetriever(settings, vector_store=store)
        sparse = SparseRetriever(settings, vector_store=store)
        hybrid = HybridSearch(settings, processor, dense, sparse, Fusion(settings))
        reranker = None if args.no_rerank else Reranker(settings)
        candidate_k = max(args.top_k, settings.retrieval.get("fusion_top_k", 10))
        if reranker is not None and reranker.enabled and reranker.provider != "none":
            top_m = settings.rerank.get("top_m", 30)
            if isinstance(top_m, bool) or not isinstance(top_m, int) or top_m < 1:
                raise ValueError("rerank.top_m must be a positive integer")
            candidate_k = max(candidate_k, top_m)
    except (OSError, TypeError, ValueError) as exc:
        logger.error("Cannot start query: %s", exc)
        return 2
    except Exception as exc:
        logger.error("Cannot initialize query components: %s", type(exc).__name__)
        return 1

    try:
        with query_trace(settings, force=args.verbose) as trace:
            candidates = hybrid.search(
                args.query, top_k=candidate_k,
                filters={"collection": args.collection} if args.collection else None,
                trace=trace,
            )
            results = (
                candidates[:args.top_k] if reranker is None
                else reranker.rerank(args.query, candidates, top_k=args.top_k, trace=trace)
            )
            if reranker is None and trace is not None:
                trace.record_stage("rerank", elapsed_ms=0, method="none", provider="none",
                                   fallback=False, reason="skipped", details={"details": {"output_count": len(results)}})
    except Exception as exc:
        logger.error("Query failed: %s", type(exc).__name__)
        return 1

    if args.verbose and trace is not None:
        titles = {"dense_retrieval": "Dense 召回", "sparse_retrieval": "Sparse 召回", "fusion": "Fusion 融合"}
        for stage in trace.stages:
            if stage["stage"] in titles:
                print_results(titles[stage["stage"]], [
                    RetrievalResult(**item) for item in stage["results"]
                ])
                if stage.get("fallback"):
                    print(f"   fallback=true | error={stage['error_type']}")
            elif stage["stage"] == "rerank":
                print_results("Rerank 重排", results)
                print(f"   fallback={str(stage['fallback']).lower()} | reason={stage['reason']}")
        if args.no_rerank:
            print("\nRerank 重排: 已跳过 (--no-rerank)")
    if not results:
        print("未找到相关文档，请先运行 ingest.py 摄取数据，或检查集合筛选条件。")
    else:
        print_results("Top-K 检索结果", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
