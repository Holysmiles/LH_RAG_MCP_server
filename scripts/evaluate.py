"""Run golden-set retrieval evaluation against the configured persisted indexes."""

import argparse
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against a golden test set.")
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "config/settings.yaml")
    parser.add_argument("--test-set", type=Path, default=PROJECT_ROOT / "tests/fixtures/golden_test_set.json")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--collection")
    parser.add_argument("--backend", help="Override evaluation provider (e.g. custom or ragas).")
    parser.add_argument("--output", type=Path, help="Save the full JSON report.")
    args = parser.parse_args(argv)
    from core.settings import load_settings
    from core.query_engine import DenseRetriever, SparseRetriever, Fusion, HybridSearch, QueryProcessor
    from libs.evaluator.evaluator_factory import EvaluatorFactory
    from libs.vector_store.chroma_store import ChromaStore
    from observability.evaluation.eval_runner import EvalRunner

    try:
        settings = load_settings(args.settings)
        if args.top_k is not None:
            settings.evaluation["top_k"] = args.top_k
        if args.collection is not None:
            settings.evaluation["collection"] = args.collection
        if args.backend is not None:
            settings.evaluation.pop("backends", None)
            settings.evaluation["provider"] = args.backend
        # Validate labels before opening a store or initializing model clients.
        EvalRunner.load_test_set(args.test_set)
        top_k = settings.evaluation.get("top_k", settings.retrieval.get("fusion_top_k", 10))
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("evaluation.top_k must be a positive integer")
        if settings.evaluation.get("collection") is not None and not settings.evaluation["collection"].strip():
            raise ValueError("collection must be nonempty")
        if settings.vector_store.get("provider") != "chroma":
            raise ValueError("evaluation CLI currently requires the chroma store")
        directory = Path(settings.vector_store.get("persist_directory", "data/db/chroma")).expanduser()
        if not (directory / "chroma.sqlite3").is_file():
            raise ValueError("no persisted index; ingest the golden set's source documents first")
        store = ChromaStore(settings.vector_store, create_if_missing=False)
        evaluator = EvaluatorFactory.create(settings)
        hybrid = HybridSearch(settings, QueryProcessor(), DenseRetriever(settings, vector_store=store),
                              SparseRetriever(settings, vector_store=store), Fusion(settings))
        runner = EvalRunner(settings, hybrid, evaluator)
    except Exception as exc:
        print(f"Cannot start evaluation ({type(exc).__name__}); check settings, test set, dependencies and ingested indexes.", file=sys.stderr)
        return 2
    try:
        report = runner.run(args.test_set)
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    except Exception as exc:
        print(f"Evaluation failed ({type(exc).__name__}); check retrieval and evaluator inputs or services.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
