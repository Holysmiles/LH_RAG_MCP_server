"""B7.8 Cross-Encoder Reranker 的离线单元测试。

测试全部使用 ``FakeScorer``。它只返回预先准备好的分数，不下载模型、不访问
网络，方便我们精确验证“输入文本对是什么、分数如何排序、失败如何回退”。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import pytest

from core.settings import Settings
from libs.reranker.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerError,
)
from libs.reranker.reranker_factory import RerankerFactory


class FakeScorer:
    """模拟 CrossEncoder.predict，并记录收到的文本对和批大小。"""

    def __init__(
        self,
        scores: object,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.scores = scores
        self.failure = failure
        self.calls: list[dict[str, Any]] = []

    def predict(
        self,
        sentences: Sequence[Sequence[str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> object:
        self.calls.append(
            {
                "sentences": [list(pair) for pair in sentences],
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
            }
        )
        if self.failure is not None:
            raise self.failure
        return self.scores


@pytest.fixture
def candidates() -> list[dict[str, Any]]:
    """构造三个携带不同 metadata 的候选，便于检查对象有没有被替换。"""
    return [
        {
            "chunk_id": "chunk-1",
            "text": "Python list basics.",
            "score": 0.9,
            "metadata": {"source": "python.pdf"},
        },
        {
            "chunk_id": "chunk-2",
            "text": "Cross-Encoder reranking details.",
            "score": 0.8,
            "metadata": {"source": "rerank.pdf"},
        },
        {
            "chunk_id": "chunk-3",
            "text": "An unrelated passage.",
            "score": 0.7,
            "metadata": {"source": "other.pdf"},
        },
    ]


@pytest.mark.unit
def test_factory_creates_cross_encoder_without_loading_model() -> None:
    """工厂只创建适配器；在真正 rerank 前不能下载或加载模型。"""
    config = {
        "provider": "cross_encoder",
        "model": "cross-encoder/test-model",
        "top_m": 20,
        "batch_size": 8,
    }

    reranker = RerankerFactory.create(Settings(rerank=config))

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.config == config
    assert reranker.config is not config
    assert reranker.model == "cross-encoder/test-model"
    assert reranker.top_m == 20
    assert reranker.batch_size == 8
    assert reranker._scorer is None


@pytest.mark.unit
def test_rerank_builds_query_text_pairs_and_preserves_original_objects(
    candidates: list[dict[str, Any]],
) -> None:
    """最高模型分数应排第一，但返回值仍是原来的候选字典。"""
    scorer = FakeScorer([0.2, 0.95, -0.1])
    reranker = CrossEncoderReranker(
        {"batch_size": 2},
        scorer=scorer,
    )

    ranked = reranker.rerank("How does reranking work?", candidates)

    assert [item["chunk_id"] for item in ranked] == [
        "chunk-2",
        "chunk-1",
        "chunk-3",
    ]
    assert ranked[0] is candidates[1]
    assert ranked[1] is candidates[0]
    assert ranked[0]["metadata"] == {"source": "rerank.pdf"}
    assert scorer.calls == [
        {
            "sentences": [
                ["How does reranking work?", "Python list basics."],
                [
                    "How does reranking work?",
                    "Cross-Encoder reranking details.",
                ],
                ["How does reranking work?", "An unrelated passage."],
            ],
            "batch_size": 2,
            "show_progress_bar": False,
        }
    ]


@pytest.mark.unit
def test_top_m_limits_scoring_and_keeps_tail_order(
    candidates: list[dict[str, Any]],
) -> None:
    """只精排前两条；第三条既不送入模型，也不被丢弃。"""
    scorer = FakeScorer([0.1, 0.9])
    reranker = CrossEncoderReranker({"top_m": 2}, scorer=scorer)

    ranked = reranker.rerank("query", candidates)

    assert [item["chunk_id"] for item in ranked] == [
        "chunk-2",
        "chunk-1",
        "chunk-3",
    ]
    assert len(scorer.calls[0]["sentences"]) == 2


@pytest.mark.unit
def test_equal_scores_preserve_rrf_order(
    candidates: list[dict[str, Any]],
) -> None:
    """分数相同时使用稳定排序，不能随机打乱 RRF 已有顺序。"""
    reranker = CrossEncoderReranker({}, scorer=FakeScorer([0.5, 0.5, 0.5]))

    ranked = reranker.rerank("query", candidates)

    assert ranked == candidates
    assert all(
        actual is original
        for actual, original in zip(ranked, candidates, strict=True)
    )


@pytest.mark.unit
def test_dataclass_candidates_are_supported() -> None:
    """未来 Core 层使用 dataclass 时也能直接读取它的 text 属性。"""

    @dataclass
    class Candidate:
        chunk_id: str
        text: str

    first = Candidate("one", "First text")
    second = Candidate("two", "Second text")
    reranker = CrossEncoderReranker({}, scorer=FakeScorer([0.1, 0.8]))

    ranked = reranker.rerank("query", [first, second])

    assert ranked == [second, first]
    assert ranked[0] is second


@pytest.mark.unit
def test_empty_candidates_do_not_load_or_call_scorer() -> None:
    """没有候选时直接返回，避免无意义的模型加载。"""
    created_models: list[str] = []

    def scorer_factory(model_name: str) -> FakeScorer:
        created_models.append(model_name)
        return FakeScorer([])

    reranker = CrossEncoderReranker({}, scorer_factory=scorer_factory)

    assert reranker.rerank("query", []) == []
    assert created_models == []


@pytest.mark.unit
def test_scorer_factory_loads_once_and_reuses_model(
    candidates: list[dict[str, Any]],
) -> None:
    """连续查询复用同一个模型，不能每次重新下载和加载。"""
    created_models: list[str] = []
    scorer = FakeScorer([0.3, 0.2, 0.1])

    def scorer_factory(model_name: str) -> FakeScorer:
        created_models.append(model_name)
        return scorer

    reranker = CrossEncoderReranker(
        {"model": "local/model"},
        scorer_factory=scorer_factory,
    )

    reranker.rerank("first query", candidates)
    reranker.rerank("second query", candidates)

    assert created_models == ["local/model"]
    assert len(scorer.calls) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ([0.1, 0.2], "expected 3 scores, got 2"),
        ([0.1, "bad", 0.3], r"scores\[1\].*real number"),
        ([0.1, math.nan, 0.3], r"scores\[1\].*finite"),
        ([0.1, math.inf, 0.3], r"scores\[1\].*finite"),
        (0.5, "scores must be a sequence"),
    ],
)
def test_invalid_scores_raise_readable_fallback_signal(
    candidates: list[dict[str, Any]],
    scores: object,
    message: str,
) -> None:
    """坏分数绝不能参与排序，应交给上层回退到原顺序。"""
    reranker = CrossEncoderReranker({}, scorer=FakeScorer(scores))

    with pytest.raises(CrossEncoderRerankerError, match=message):
        reranker.rerank("query", candidates)


@pytest.mark.unit
def test_scorer_timeout_becomes_safe_fallback_signal(
    candidates: list[dict[str, Any]],
) -> None:
    """超时统一成一个异常类型，并且不泄露底层错误详情。"""
    scorer = FakeScorer(
        [],
        failure=TimeoutError("private model path"),
    )
    reranker = CrossEncoderReranker({}, scorer=scorer)

    with pytest.raises(
        CrossEncoderRerankerError,
        match="cross_encoder reranker failed: TimeoutError",
    ) as error:
        reranker.rerank("query", candidates)

    assert "private model path" not in str(error.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("candidate_list", "message"),
    [
        ([{"chunk_id": "missing-text"}], r"candidates\[0\]\.text"),
        ([{"chunk_id": "wrong-text", "text": 42}], "text must be a string"),
    ],
)
def test_invalid_candidates_fail_before_scorer_call(
    candidate_list: list[dict[str, Any]],
    message: str,
) -> None:
    """本地候选数据不合法时，不能浪费模型计算。"""
    scorer = FakeScorer([])
    reranker = CrossEncoderReranker({}, scorer=scorer)

    with pytest.raises(ValueError, match=message):
        reranker.rerank("query", candidate_list)

    assert scorer.calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"model": ""}, "model"),
        ({"top_m": 0}, "top_m"),
        ({"top_m": True}, "top_m"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_invalid_configuration_fails_before_model_loading(
    config: dict[str, Any],
    message: str,
) -> None:
    """配置错误必须在构造阶段指出，不能等模型推理时才爆炸。"""
    with pytest.raises(ValueError, match=message):
        CrossEncoderReranker(config, scorer=FakeScorer([]))
