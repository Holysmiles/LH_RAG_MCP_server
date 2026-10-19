"""Evaluate a validated golden set against the shared retrieval pipeline."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path

from core.settings import Settings
from core.trace.trace_collector import query_trace
from core.types import RetrievalResult
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator


@dataclass(frozen=True)
class EvalReport:
    test_set_path: str
    started_at: str
    finished_at: str
    top_k: int
    metrics: dict[str, float]
    results: list[dict]

    @property
    def hit_rate(self) -> float:
        return self.metrics["hit_rate"]

    @property
    def mrr(self) -> float:
        return self.metrics["mrr"]

    def to_dict(self) -> dict:
        return asdict(self)


class EvalRunner:
    def __init__(self, settings: Settings, hybrid_search, evaluator: BaseEvaluator) -> None:
        if not isinstance(settings, Settings) or not isinstance(evaluator, BaseEvaluator):
            raise TypeError("EvalRunner requires Settings and a BaseEvaluator")
        if not callable(getattr(hybrid_search, "search", None)):
            raise TypeError("hybrid_search must provide search()")
        self.settings, self.hybrid_search, self.evaluator = settings, hybrid_search, evaluator
        self.top_k = settings.evaluation.get("top_k", settings.retrieval.get("fusion_top_k", 10))
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("evaluation.top_k must be a positive integer")

    @staticmethod
    def load_test_set(path: str | Path) -> list[dict]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = data.get("test_cases") if isinstance(data, dict) else None
        if not isinstance(cases, list) or not cases:
            raise ValueError("Golden set requires a nonempty test_cases list")
        normalized = []
        for index, case in enumerate(cases):
            if not isinstance(case, dict) or not isinstance(case.get("query"), str) or not case["query"].strip():
                raise ValueError(f"test_cases[{index}] requires a nonempty query")
            row = dict(case, query=case["query"].strip())
            for key in ("expected_chunk_ids", "expected_sources"):
                values = row.get(key, [])
                if (not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values)
                        or len(set(values)) != len(values)):
                    raise ValueError(f"test_cases[{index}].{key} must be a list of unique nonempty strings")
                row[key] = values
            if not row["expected_chunk_ids"] and not row["expected_sources"]:
                raise ValueError(f"test_cases[{index}] needs expected_chunk_ids or expected_sources")
            for key in ("collection", "answer", "reference"):
                if key in row and (not isinstance(row[key], str) or not row[key].strip()):
                    raise ValueError(f"test_cases[{index}].{key} must be a nonempty string")
            normalized.append(row)
        return normalized

    @staticmethod
    def _source_matches(source: str, expected: str) -> bool:
        source, expected = source.replace("\\", "/"), expected.replace("\\", "/")
        return source == expected or ("/" not in expected and source.rsplit("/", 1)[-1] == expected)

    def run(self, test_set_path: str | Path) -> EvalReport:
        """Average per-query metrics. Any query/backend failure aborts the report.

        A candidate is relevant when its ID or source matches a golden label.
        Semantic backends receive retrieved text and the sample's answer and
        reference; no answer is generated or inferred from the reference.
        """
        cases = self.load_test_set(test_set_path)
        started = datetime.now(timezone.utc).isoformat()
        rows = []
        for case in cases:
            collection = case.get("collection", self.settings.evaluation.get("collection"))
            if collection is not None and (not isinstance(collection, str) or not collection.strip()):
                raise ValueError("evaluation.collection must be a nonempty string")
            with query_trace(self.settings) as trace:
                hits = self.hybrid_search.search(case["query"], top_k=self.top_k,
                                                 filters={"collection": collection} if collection else None, trace=trace)
                if not isinstance(hits, list) or any(not isinstance(hit, RetrievalResult) for hit in hits):
                    raise TypeError("search must return RetrievalResult objects")
                hits = hits[:self.top_k]
                retrieved = [hit.chunk_id for hit in hits]
                golden = list(case["expected_chunk_ids"])
                for hit in hits:
                    source = hit.metadata.get("source_path", "")
                    if isinstance(source, str) and any(self._source_matches(source, expected) for expected in case["expected_sources"]):
                        if hit.chunk_id not in golden:
                            golden.append(hit.chunk_id)
                metrics = self.evaluator.evaluate_sample(
                    case["query"], retrieved, golden, trace=trace, answer=case.get("answer"),
                    contexts=[hit.text for hit in hits], reference=case.get("reference"))
                if not isinstance(metrics, Mapping):
                    raise TypeError("evaluator must return a metric mapping")
                metrics = dict(metrics)
                for name, value in metrics.items():
                    if (not isinstance(name, str) or not name.strip() or isinstance(value, bool)
                            or not isinstance(value, Real) or not math.isfinite(value)):
                        raise ValueError("evaluator returned an invalid metric")
                    metrics[name] = float(value)
                for name, value in CustomEvaluator().evaluate(case["query"], retrieved, golden).items():
                    metrics.setdefault(name, value)
                rows.append({"query": case["query"], "collection": collection, "retrieved_ids": retrieved,
                             "retrieved_sources": [hit.metadata.get("source_path", "") for hit in hits],
                             "expected_chunk_ids": case["expected_chunk_ids"], "expected_sources": case["expected_sources"],
                             "metrics": metrics, "trace_id": trace.trace_id if trace else None})
        keys = set(rows[0]["metrics"])
        if any(set(row["metrics"]) != keys for row in rows):
            raise ValueError("Evaluator metric names must be consistent across queries")
        averages = {name: sum(row["metrics"][name] for row in rows) / len(rows) for name in sorted(keys)}
        return EvalReport(str(Path(test_set_path).resolve()), started, datetime.now(timezone.utc).isoformat(),
                          self.top_k, averages, rows)
