"""D4 RRF Fusion 的公式、确定性与失败边界测试。"""

from __future__ import annotations

from typing import Any

import pytest

from core import RetrievalResult
from core.query_engine import Fusion, RRFFusion
from core.settings import Settings


def candidate(
    chunk_id: str,
    *,
    score: float = 0.0,
    text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text if text is not None else f"text-{chunk_id}",
        metadata=metadata if metadata is not None else {"source_path": f"{chunk_id}.pdf"},
    )


@pytest.mark.unit
def test_rrf_sums_reciprocal_ranks_and_ignores_provider_scores() -> None:
    dense = [candidate("a", score=-100.0), candidate("b", score=999.0)]
    sparse = [candidate("b", score=-5.0), candidate("c", score=10_000.0)]

    results = Fusion(k=60, top_k=3).fuse(dense, sparse)

    assert [item.chunk_id for item in results] == ["b", "a", "c"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61)
    assert results[2].score == pytest.approx(1 / 62)


@pytest.mark.unit
def test_settings_and_explicit_overrides_control_k_and_top_k() -> None:
    settings = Settings(retrieval={"rrf_k": 1, "fusion_top_k": 2})
    fusion = Fusion(settings)

    configured = fusion.fuse(
        [candidate("a"), candidate("b")],
        [candidate("b"), candidate("c")],
    )
    overridden = fusion.fuse(
        [candidate("a"), candidate("b")],
        [candidate("b"), candidate("c")],
        top_k=1,
    )

    assert fusion.k == 1
    assert [item.chunk_id for item in configured] == ["b", "a"]
    assert configured[0].score == pytest.approx(1 / 3 + 1 / 2)
    assert [item.chunk_id for item in overridden] == ["b"]


@pytest.mark.unit
def test_single_route_fallback_preserves_its_ranking() -> None:
    dense = [candidate("first"), candidate("second"), candidate("third")]

    assert [item.chunk_id for item in Fusion().fuse(dense, [])] == [
        "first",
        "second",
        "third",
    ]
    assert Fusion().fuse([], []) == []


@pytest.mark.unit
def test_equal_rrf_scores_preserve_first_seen_order_deterministically() -> None:
    fusion = Fusion()

    first = fusion.fuse([candidate("z")], [candidate("a")])
    repeated = fusion.fuse([candidate("z")], [candidate("a")])

    assert [item.chunk_id for item in first] == ["z", "a"]
    assert [item.chunk_id for item in repeated] == ["z", "a"]


@pytest.mark.unit
def test_same_chunk_keeps_payload_and_returns_fresh_result() -> None:
    original = candidate(
        "shared",
        score=0.9,
        metadata={"source_path": "shared.pdf", "tags": ["rag"]},
    )

    fused = Fusion().fuse(
        [original],
        [
            candidate(
                "shared",
                score=12.0,
                metadata={"source_path": "shared.pdf", "tags": ["rag"]},
            )
        ],
    )[0]

    assert fused is not original
    assert fused.text == original.text
    assert fused.metadata == original.metadata
    assert fused.metadata is not original.metadata


@pytest.mark.unit
def test_duplicate_id_inside_one_route_is_rejected() -> None:
    duplicate = candidate("same")

    with pytest.raises(ValueError, match="dense.*duplicate"):
        Fusion().fuse([duplicate, duplicate], [])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("different", "field"),
    [
        (candidate("shared", text="different"), "payloads"),
        (
            candidate("shared", metadata={"source_path": "other.pdf"}),
            "payloads",
        ),
    ],
)
def test_conflicting_cross_route_payload_is_rejected(
    different: RetrievalResult,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        Fusion().fuse([candidate("shared")], [different])


@pytest.mark.unit
@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(k: object) -> None:
    with pytest.raises(ValueError, match="k"):
        Fusion(k=k)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        Fusion(top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.unit
def test_invalid_rankings_and_items_are_rejected() -> None:
    with pytest.raises(TypeError, match="dense_results"):
        Fusion().fuse("not-a-ranking", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"sparse_results\[0\]"):
        Fusion().fuse([], [{}])  # type: ignore[list-item]


@pytest.mark.unit
def test_algorithm_alias_points_to_same_implementation() -> None:
    assert RRFFusion is Fusion
