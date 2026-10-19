"""Build evaluation runs only after an explicit dashboard action."""

from dataclasses import replace
from pathlib import Path

from core.settings import Settings
from core.query_engine import DenseRetriever, SparseRetriever, HybridSearch, QueryProcessor, Fusion
from libs.evaluator.evaluator_factory import EvaluatorFactory
from libs.vector_store.chroma_store import ChromaStore
from observability.evaluation.eval_runner import EvalReport, EvalRunner


DEFAULT_TEST_SET = Path(__file__).resolve().parents[4] / "tests/fixtures/golden_test_set.json"


class EvaluationInputError(ValueError):
    """A user-correctable validation failure with a safe dashboard message."""


class EvaluationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def test_sets(self) -> list[Path]:
        values = self.settings.evaluation.get("test_sets", [self.settings.evaluation.get("test_set_path", DEFAULT_TEST_SET)])
        if (not isinstance(values, (list, tuple)) or not values
                or any(not isinstance(value, (str, Path)) or not str(value).strip() for value in values)):
            raise EvaluationInputError("请在 evaluation.test_sets 中配置非空黄金集路径列表。")
        return list(dict.fromkeys(Path(value).expanduser().resolve() for value in values))

    def run(self, test_set_path: str | Path, backend: str, top_k: int) -> EvalReport:
        if backend not in {"configured", "custom", "ragas", "all"}:
            raise EvaluationInputError("请选择有效的评估后端。")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise EvaluationInputError("Top-K 必须为正整数。")
        try:
            cases = EvalRunner.load_test_set(test_set_path)
        except (OSError, ValueError, TypeError):
            raise EvaluationInputError("无法加载黄金集，请检查文件路径、JSON 格式、query 和预期标签。") from None
        config = dict(self.settings.evaluation, top_k=top_k)
        if backend == "all":
            config["backends"] = ["ragas", "custom"]
        elif backend != "configured":
            config.pop("backends", None)
            config["provider"] = backend
        selected = config.get("backends", [config.get("provider")])
        for entry in selected:
            options = dict(config)
            options.update(entry if isinstance(entry, dict) else {"provider": entry})
            if str(options.get("provider", "")).strip().lower() == "ragas":
                samples = options.get("samples", {})
                for case in cases:
                    sample = samples.get(case["query"], {}) if isinstance(samples, dict) else {}
                    answer = case.get("answer", sample.get("answer")) if isinstance(sample, dict) else None
                    if not isinstance(answer, str) or not answer.strip():
                        raise EvaluationInputError("Ragas 需要每条样例的待评 answer，可在黄金集或 evaluation.samples 中提供。")
                    reference = case.get("reference", sample.get("reference"))
                    if "context_precision" in options.get("metrics", []) and (not isinstance(reference, str) or not reference.strip()):
                        raise EvaluationInputError("Context Precision 需要每条样例的 reference 参考答案。")
        settings = replace(self.settings, evaluation=config)
        if settings.vector_store.get("provider") != "chroma":
            raise EvaluationInputError("当前评估面板支持 Chroma 存储后端。")
        directory = Path(settings.vector_store.get("persist_directory", "data/db/chroma")).expanduser()
        if not (directory / "chroma.sqlite3").is_file():
            raise EvaluationInputError("尚无持久化索引，请先摄取黄金集对应的源文档。")
        store = ChromaStore(settings.vector_store, create_if_missing=False)
        evaluator = EvaluatorFactory.create(settings)
        hybrid = HybridSearch(settings, QueryProcessor(), DenseRetriever(settings, vector_store=store),
                              SparseRetriever(settings, vector_store=store), Fusion(settings))
        return EvalRunner(settings, hybrid, evaluator).run(test_set_path)
