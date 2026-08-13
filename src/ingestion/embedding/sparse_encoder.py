"""Create the sparse statistics consumed by the BM25 indexer.

``Chunk[] -> tokenize -> term frequencies -> BM25 batch statistics``

This stage does not calculate IDF and does not write files.  It only turns
each chunk into a deterministic ``term -> tf`` mapping and reports the corpus
statistics that C11 needs.  Keeping tokenization here makes the indexer and
the future sparse retriever use exactly the same representation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import re
from typing import Any, ClassVar

from core import Chunk


@dataclass(frozen=True, slots=True)
class SparseChunkEncoding:
    """One chunk's sparse representation.

    ``term_weights`` contains raw term frequency (TF), not IDF.  C11 adds the
    corpus-level IDF after it has seen all records.
    """

    chunk_id: str
    term_weights: dict[str, float]
    doc_length: int

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible shape expected by the storage layer."""
        return {
            "chunk_id": self.chunk_id,
            "term_weights": dict(self.term_weights),
            "doc_length": self.doc_length,
        }


@dataclass(frozen=True, slots=True)
class SparseEncodingBatch:
    """Corpus-level sparse statistics plus per-chunk records.

    ``document_frequency`` counts a term once per document, even when its TF
    is greater than one.  ``avg_doc_length`` is zero for an empty batch.
    """

    records: tuple[SparseChunkEncoding, ...]
    document_count: int
    document_frequency: dict[str, int]
    avg_doc_length: float

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SparseChunkEncoding:
        return self.records[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "document_count": self.document_count,
            "document_frequency": dict(self.document_frequency),
            "avg_doc_length": self.avg_doc_length,
        }


class SparseEncoder:
    """Build deterministic BM25-ready term statistics from ``Chunk`` objects.

    Inputs:
        A non-string sequence of ``Chunk`` objects.  ``stopwords`` is optional
        and can be injected by callers that need domain-specific filtering.
    Outputs:
        ``SparseEncodingBatch`` with one ordered record per input chunk and
        corpus document frequency statistics for the next indexing stage.
    Failure boundaries:
        Invalid public input raises immediately.  Empty or whitespace-only
        text is valid and produces an empty term map with length zero; this is
        important because a sparse route must not invent a keyword for it.
    """

    _token_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+"
    )

    def __init__(self, *, stopwords: Iterable[str] | None = None) -> None:
        if stopwords is None:
            self.stopwords: frozenset[str] = frozenset()
        else:
            normalized: set[str] = set()
            for word in stopwords:
                if not isinstance(word, str):
                    raise TypeError("stopwords must contain only strings")
                if word.strip():
                    normalized.add(word.strip().lower())
            self.stopwords = frozenset(normalized)

    def encode(self, chunks: Sequence[Chunk]) -> SparseEncodingBatch:
        """Tokenize a batch and calculate TF, DF, and average document length."""
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a non-string sequence")

        records: list[SparseChunkEncoding] = []
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
            terms = self._tokenize(chunk.text)
            frequencies = Counter(terms)
            weights = {term: float(count) for term, count in frequencies.items()}
            record = SparseChunkEncoding(
                chunk_id=chunk.id,
                term_weights=weights,
                doc_length=len(terms),
            )
            records.append(record)
            total_length += record.doc_length
            document_frequency.update(frequencies.keys())

        document_count = len(records)
        return SparseEncodingBatch(
            records=tuple(records),
            document_count=document_count,
            document_frequency=dict(sorted(document_frequency.items())),
            avg_doc_length=(total_length / document_count if document_count else 0.0),
        )

    def _tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("chunk text must be a string")
        tokens: list[str] = []
        for match in self._token_pattern.finditer(text):
            token = match.group(0).lower()
            if token not in self.stopwords:
                tokens.append(token)
        return tokens


__all__ = ["SparseChunkEncoding", "SparseEncoder", "SparseEncodingBatch"]
