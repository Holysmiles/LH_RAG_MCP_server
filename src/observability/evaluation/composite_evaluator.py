"""Run independent evaluators concurrently and merge their named metrics."""

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import math
from numbers import Real

from libs.evaluator.base_evaluator import BaseEvaluator


class CompositeEvaluator(BaseEvaluator):
    """Return a complete metric set or propagate a backend failure.

    Metric names must be unique across backends: collisions raise instead of
    silently overwriting scores. Each worker receives its own ID lists. Trace
    is shared using TraceContext's thread-safe recording interface. Concurrent
    callers must use distinct backend instances if those backends hold state.
    """

    def __init__(self, evaluators: Sequence[BaseEvaluator]) -> None:
        if (isinstance(evaluators, (str, bytes)) or not isinstance(evaluators, Sequence)
                or not evaluators or any(not isinstance(item, BaseEvaluator) for item in evaluators)):
            raise ValueError("evaluators must be a nonempty sequence of BaseEvaluator instances")
        if len({id(item) for item in evaluators}) != len(evaluators):
            raise ValueError("evaluators must contain distinct instances")
        self.evaluators = tuple(evaluators)

    def evaluate(self, query: str, retrieved_ids: Sequence[str], golden_ids: Sequence[str],
                 trace=None) -> dict[str, float]:
        if isinstance(retrieved_ids, (str, bytes)) or isinstance(golden_ids, (str, bytes)):
            raise ValueError("retrieved_ids and golden_ids must be sequences of IDs")
        retrieved, golden = list(retrieved_ids), list(golden_ids)
        with ThreadPoolExecutor(max_workers=len(self.evaluators), thread_name_prefix="rag-eval") as pool:
            futures = [pool.submit(evaluator.evaluate, query, list(retrieved), list(golden), trace=trace)
                       for evaluator in self.evaluators]
            results = [future.result() for future in futures]
        merged = {}
        for result in results:
            if not isinstance(result, Mapping):
                raise TypeError("Evaluator must return a metric mapping")
            for name, value in result.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Metric names must be nonempty strings")
                if name in merged:
                    raise ValueError(f"Duplicate metric name across evaluators: {name}")
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(f"Metric '{name}' must be a finite numeric score")
                merged[name] = float(value)
        return merged
