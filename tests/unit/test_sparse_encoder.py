"""Contract tests for BM25 sparse encoding statistics."""

from __future__ import annotations

import pytest

from core import Chunk
from ingestion.embedding import SparseEncoder


def make_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        text=text,
        metadata={"source_path": "docs/source.pdf", "chunk_index": index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


@pytest.mark.unit
def test_encode_returns_ordered_tf_records_and_corpus_statistics() -> None:
    result = SparseEncoder().encode(
        [
            make_chunk(0, "Azure endpoint endpoint"),
            make_chunk(1, "endpoint vector"),
        ]
    )

    assert [record.chunk_id for record in result] == ["chunk-0", "chunk-1"]
    assert result[0].term_weights == {"azure": 1.0, "endpoint": 2.0}
    assert result[0].doc_length == 3
    assert result.document_count == 2
    assert result.document_frequency == {"azure": 1, "endpoint": 2, "vector": 1}
    assert result.avg_doc_length == 2.5
    assert result.to_dict()["records"][0]["chunk_id"] == "chunk-0"


@pytest.mark.unit
def test_tokenization_is_case_insensitive_and_supports_chinese_and_numbers() -> None:
    result = SparseEncoder().encode([make_chunk(0, "Azure AZURE 配置 3.11")])

    assert result[0].term_weights == {"azure": 2.0, "配置": 1.0, "3.11": 1.0}


@pytest.mark.unit
def test_stopwords_are_filtered_without_changing_other_terms() -> None:
    result = SparseEncoder(stopwords={"the", "and"}).encode(
        [make_chunk(0, "the endpoint and vector")]
    )

    assert result[0].term_weights == {"endpoint": 1.0, "vector": 1.0}
    assert result[0].doc_length == 2


@pytest.mark.unit
def test_empty_batch_and_empty_text_have_explicit_zero_statistics() -> None:
    empty = SparseEncoder().encode([])
    blank = SparseEncoder().encode([make_chunk(0, "   --- ... ")])

    assert len(empty) == 0
    assert empty.document_count == 0
    assert empty.document_frequency == {}
    assert empty.avg_doc_length == 0.0
    assert blank[0].term_weights == {}
    assert blank[0].doc_length == 0
    assert blank.avg_doc_length == 0.0


@pytest.mark.unit
def test_invalid_inputs_are_rejected_at_the_public_boundary() -> None:
    with pytest.raises(TypeError, match="stopwords"):
        SparseEncoder(stopwords="the")
    with pytest.raises(TypeError, match="stopwords"):
        SparseEncoder(stopwords=["ok", 1])  # type: ignore[list-item]
    encoder = SparseEncoder()
    with pytest.raises(TypeError, match="chunks must be"):
        encoder.encode("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"chunks\[0\]"):
        encoder.encode([object()])  # type: ignore[list-item]
