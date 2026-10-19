"""Ragas 0.3 semantic evaluation with explicit answer/context inputs.

Chunk IDs alone cannot measure faithfulness. Callers may pass semantic fields
as keywords, configure samples keyed by query, or inject a sample_resolver that
receives (query, retrieved_ids, golden_ids). LLM/embedding clients use Ragas's
native interfaces (or supported LangChain clients); omitted clients use Ragas
defaults. No evaluation or model construction occurs at module import.
"""

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
import math
from numbers import Real
from typing import Any

from libs.evaluator.base_evaluator import BaseEvaluator


class RagasEvaluator(BaseEvaluator):
    supported_metrics = ("faithfulness", "answer_relevancy", "context_precision")

    def __init__(self, config: Mapping[str, Any] | None = None, *, llm=None, embeddings=None,
                 sample_resolver: Callable | None = None) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("Ragas configuration must be a mapping")
        self.config = dict(config or {})
        metrics = self.config.get("metrics", self.supported_metrics[:2])
        if (isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence) or not metrics
                or any(not isinstance(name, str) or name not in self.supported_metrics for name in metrics)
                or len(set(metrics)) != len(metrics)):
            raise ValueError("evaluation.metrics must be a nonempty unique list of supported Ragas metrics")
        self.metrics = tuple(metrics)
        self.llm = llm if llm is not None else self.config.get("llm")
        self.embeddings = embeddings if embeddings is not None else self.config.get("embeddings")
        self.sample_resolver = sample_resolver
        if sample_resolver is not None and not callable(sample_resolver):
            raise TypeError("sample_resolver must be callable")
        try:
            self._ragas = import_module("ragas")
            self._metrics = import_module("ragas.metrics")
        except ImportError as exc:
            raise ImportError("Ragas evaluation requires ragas>=0.3.9,<0.4.0 and its dependencies; install the project dependencies") from exc

    @staticmethod
    def _text(value, name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Ragas requires a nonempty {name}")
        return value

    @staticmethod
    def _strings(values, name, *, allow_empty=True):
        if (isinstance(values, (str, bytes)) or not isinstance(values, Sequence)
                or (not values and not allow_empty)
                or any(not isinstance(value, str) or not value.strip() for value in values)):
            raise ValueError(f"{name} must be a sequence of nonempty strings")
        return list(values)

    def evaluate(self, query: str, retrieved_ids: Sequence[str] = (), golden_ids: Sequence[str] = (),
                 trace=None, *, answer: str | None = None, contexts: Sequence[str] | None = None,
                 reference: str | None = None) -> dict[str, float]:
        query = self._text(query, "query")
        retrieved = self._strings(retrieved_ids, "retrieved_ids")
        golden = self._strings(golden_ids, "golden_ids")
        samples = self.config.get("samples", {})
        if not isinstance(samples, Mapping):
            raise ValueError("evaluation.samples must be a mapping keyed by query")
        sample = (self.sample_resolver(query, retrieved, golden) if self.sample_resolver is not None
                  else samples.get(query, {}))
        if not isinstance(sample, Mapping):
            raise ValueError("Ragas sample must be a mapping with answer, contexts and optional reference")
        response = self._text(answer if answer is not None else sample.get("answer"), "answer (IDs alone are insufficient)")
        contexts = self._strings(contexts if contexts is not None else sample.get("contexts"),
                                 "contexts", allow_empty=False)
        row = {"user_input": query, "response": response, "retrieved_contexts": contexts,
               "retrieved_context_ids": retrieved, "reference_context_ids": golden}
        reference = reference if reference is not None else sample.get("reference")
        if "context_precision" in self.metrics:
            row["reference"] = self._text(reference, "reference for context_precision")
        elif reference is not None:
            row["reference"] = self._text(reference, "reference")
        classes = {"faithfulness": "Faithfulness", "answer_relevancy": "AnswerRelevancy",
                   "context_precision": "ContextPrecision"}
        # New instances keep Ragas's mutable metric clients isolated between calls.
        metrics = [getattr(self._metrics, classes[name])() for name in self.metrics]
        dataset = self._ragas.EvaluationDataset.from_list([row])
        result = self._ragas.evaluate(dataset=dataset, metrics=metrics, llm=self.llm,
                                      embeddings=self.embeddings, raise_exceptions=True, show_progress=False)
        scores = {}
        for name, metric in zip(self.metrics, metrics):
            values = result[metric.name]
            value = values[0] if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else values
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Ragas returned an invalid or undefined score for {name}")
            scores[name] = float(value)
        return scores

    def evaluate_sample(self, query, retrieved_ids, golden_ids, trace=None, *,
                        answer=None, contexts=None, reference=None) -> dict[str, float]:
        return self.evaluate(query, retrieved_ids, golden_ids, trace, answer=answer,
                             contexts=contexts, reference=reference)
