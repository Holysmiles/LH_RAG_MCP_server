"""使用 Cross-Encoder 对 RRF 粗排结果做第二次精排。

它在整体查询架构中的位置如下：

``Dense/Sparse 召回 -> RRF 融合 -> CrossEncoderReranker -> Top-K -> 最终响应``

输入：

- ``query``：用户问题，例如“怎么配置向量数据库？”；
- ``candidates``：RRF 已经找出的候选列表；每个候选必须有 ``text`` 字段；
- 配置中的 ``top_m``：最多让前多少条候选进入成本更高的精排。

模型实际看到的输入是一批文本对：

``[[query, candidate_1.text], [query, candidate_2.text], ...]``

输出：

- 返回一个新列表，但里面仍然是调用方传入的原始候选对象；
- 前 ``top_m`` 条按 Cross-Encoder 分数从高到低排序；
- 超出 ``top_m`` 的候选不计算分数，保持原顺序放在列表尾部；
- 模型加载或打分失败时抛出 ``CrossEncoderRerankerError``，供 Core 层回退。

这里不自己实现神经网络。``sentence-transformers.CrossEncoder`` 负责模型推理，
本文件只负责项目输入校验、批量组装、分数校验和对象顺序恢复。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Protocol

from libs.reranker.base_reranker import BaseReranker, CandidateT

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


__all__ = ["CrossEncoderReranker", "CrossEncoderRerankerError"]


class CrossEncoderRerankerError(RuntimeError):
    """告诉上层“精排失败，请使用 RRF 原顺序”的统一异常。"""


class CrossEncoderScorer(Protocol):
    """Cross-Encoder 最小能力：接收文本对，返回每一对的相关性分数。"""

    def predict(
        self,
        sentences: Sequence[Sequence[str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> object:
        """为每个 ``[query, candidate_text]`` 返回一个分数。"""
        ...


ScorerFactory = Callable[[str], CrossEncoderScorer]


class CrossEncoderReranker(BaseReranker):
    """使用本地 Cross-Encoder 分数重新排列 Top-M 候选。

    测试可以直接注入假的 ``scorer``，因此不用联网、不用下载模型，也不会
    依赖 GPU。生产环境不传 ``scorer``，第一次调用 ``rerank`` 时才按配置的
    ``model`` 创建真实模型。这叫“延迟加载”：没启用的功能不消耗启动时间和内存。
    """

    provider_name: ClassVar[str] = "cross_encoder"
    default_model: ClassVar[str] = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    default_top_m: ClassVar[int] = 30
    default_batch_size: ClassVar[int] = 16

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        scorer: CrossEncoderScorer | None = None,
        scorer_factory: ScorerFactory | None = None,
    ) -> None:
        """读取配置，但此时故意不下载、不加载真实模型。"""
        if not isinstance(config, Mapping):
            self._raise_configuration("configuration must be a mapping")
        if scorer is not None and scorer_factory is not None:
            self._raise_configuration(
                "scorer and scorer_factory cannot be provided together"
            )
        if scorer is not None and not callable(getattr(scorer, "predict", None)):
            self._raise_configuration("scorer must provide predict()")
        if scorer_factory is not None and not callable(scorer_factory):
            self._raise_configuration("scorer_factory must be callable")

        # 复制配置，避免 Provider 修改全局 Settings 中的原始字典。
        self.config = dict(config)
        self.model = self._read_non_empty_string(
            config,
            "model",
            self.default_model,
        )
        self.top_m = self._read_positive_int(
            config,
            "top_m",
            self.default_top_m,
        )
        self.batch_size = self._read_positive_int(
            config,
            "batch_size",
            self.default_batch_size,
        )
        self._scorer = scorer
        self._scorer_factory = (
            scorer_factory or self._create_sentence_transformer_scorer
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[CandidateT],
        trace: TraceContext | None = None,
    ) -> list[CandidateT]:
        """给前 Top-M 个候选打分，并返回重新排序后的原始对象。

        整个方法可以按下面五步阅读：

        1. 检查 query 和 candidates，错误输入不进入模型；
        2. 切出需要精排的前 Top-M 条，剩余候选先放到一边；
        3. 把每条候选转换为 ``[query, text]``；
        4. 一次批量预测并检查分数数量、类型和 NaN/Infinity；
        5. 按分数降序排列 Top-M，再把未精排的尾部原样接回去。
        """
        del trace
        if not isinstance(query, str) or not query.strip():
            self._raise_validation("query must be a non-empty string")
        if isinstance(candidates, (str, bytes)) or not isinstance(
            candidates, Sequence
        ):
            self._raise_validation("candidates must be a sequence")
        if not candidates:
            # 没有候选就没有排序工作，直接返回，避免加载模型。
            return []

        candidate_list = list(candidates)
        candidates_to_score = candidate_list[: self.top_m]
        untouched_tail = candidate_list[self.top_m :]
        text_pairs = self._build_text_pairs(
            query.strip(),
            candidates_to_score,
        )

        try:
            scorer = self._get_scorer()
            raw_scores = scorer.predict(
                text_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            # 不回显底层异常详情，因为其中可能包含模型缓存路径或远程地址。
            raise CrossEncoderRerankerError(
                "cross_encoder reranker failed: "
                f"{type(exc).__name__}"
            ) from None

        scores = self._read_scores(
            raw_scores,
            expected_count=len(candidates_to_score),
        )

        # sorted 是稳定排序：两个分数相同时，候选仍保持 RRF 原先的相对顺序。
        ranked_indices = sorted(
            range(len(candidates_to_score)),
            key=lambda index: scores[index],
            reverse=True,
        )
        ranked_head = [candidates_to_score[index] for index in ranked_indices]
        return ranked_head + untouched_tail

    def _build_text_pairs(
        self,
        query: str,
        candidates: Sequence[CandidateT],
    ) -> list[list[str]]:
        """只读取模型真正需要的 text，不复制 metadata 等业务字段。"""
        pairs: list[list[str]] = []
        for index, candidate in enumerate(candidates):
            text = self._read_candidate_text(candidate)
            if not isinstance(text, str):
                self._raise_validation(
                    f"candidates[{index}].text must be a string"
                )
            pairs.append([query, text])
        return pairs

    @staticmethod
    def _read_candidate_text(candidate: object) -> object:
        """同时支持当前字典候选和未来 Core 层的 dataclass 候选。"""
        if isinstance(candidate, Mapping):
            return candidate.get("text")
        return getattr(candidate, "text", None)

    def _get_scorer(self) -> CrossEncoderScorer:
        """第一次调用时创建模型，之后复用同一个模型实例。"""
        if self._scorer is None:
            self._scorer = self._scorer_factory(self.model)
            if not callable(getattr(self._scorer, "predict", None)):
                raise TypeError("scorer_factory must return an object with predict()")
        return self._scorer

    @staticmethod
    def _create_sentence_transformer_scorer(
        model_name: str,
    ) -> CrossEncoderScorer:
        """按模型名创建 sentence-transformers 的真实 CrossEncoder。"""
        # 局部导入是延迟加载的关键：只有真正启用本 Provider 才导入重型依赖。
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name)

    @staticmethod
    def _read_scores(
        raw_scores: object,
        *,
        expected_count: int,
    ) -> list[float]:
        """把 numpy/list 等输出统一成有限浮点数列表。"""
        if isinstance(raw_scores, (str, bytes, Mapping)):
            raise CrossEncoderRerankerError(
                "cross_encoder response error: scores must be a sequence"
            )
        try:
            score_items = list(raw_scores)  # type: ignore[arg-type]
        except TypeError:
            raise CrossEncoderRerankerError(
                "cross_encoder response error: scores must be a sequence"
            ) from None

        if len(score_items) != expected_count:
            raise CrossEncoderRerankerError(
                "cross_encoder response error: "
                f"expected {expected_count} scores, got {len(score_items)}"
            )

        scores: list[float] = []
        for index, score in enumerate(score_items):
            if isinstance(score, bool) or not isinstance(score, Real):
                raise CrossEncoderRerankerError(
                    "cross_encoder response error: "
                    f"scores[{index}] must be a real number"
                )
            number = float(score)
            if not math.isfinite(number):
                raise CrossEncoderRerankerError(
                    "cross_encoder response error: "
                    f"scores[{index}] must be finite"
                )
            scores.append(number)
        return scores

    @classmethod
    def _read_non_empty_string(
        cls,
        config: Mapping[str, Any],
        field: str,
        default: str,
    ) -> str:
        value = config.get(field, default)
        if not isinstance(value, str) or not value.strip():
            cls._raise_configuration(f"{field} must be a non-empty string")
        return value.strip()

    @classmethod
    def _read_positive_int(
        cls,
        config: Mapping[str, Any],
        field: str,
        default: int,
    ) -> int:
        value = config.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            cls._raise_configuration(f"{field} must be a positive integer")
        return value

    @classmethod
    def _raise_configuration(cls, detail: str) -> NoReturn:
        raise ValueError(f"{cls.provider_name} configuration error: {detail}")

    @classmethod
    def _raise_validation(cls, detail: str) -> NoReturn:
        raise ValueError(f"{cls.provider_name} validation error: {detail}")
