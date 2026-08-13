"""Persistent BM25 inverted index for sparse ingestion records.

``SparseEncodingBatch -> BM25Indexer -> JSON index -> ranked chunk results``

The indexer owns corpus-level IDF and postings.  SparseEncoder owns
tokenization and term frequency, so query tokenization reuses the same rules.
Updates are implemented as replace-and-rebuild over the in-memory record map;
that is slightly more work than mutating individual posting lists, but it
guarantees that an upsert or document removal cannot leave stale postings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from ingestion.embedding.sparse_encoder import (
    SparseChunkEncoding,
    SparseEncoder,
    SparseEncodingBatch,
)


class BM25Indexer:
    """Build, persist, update, remove, and query a local BM25 index.

    Inputs:
        ``SparseEncodingBatch`` or a sequence of ``SparseChunkEncoding``
        records.  ``source`` is optional document ownership metadata used by
        ``remove_document``; callers should pass the same source on upsert.
    Outputs:
        A JSON index containing IDF, postings, document lengths, and source
        ownership.  ``query`` returns ranked ``chunk_id`` and numeric score.
    Failure boundaries:
        Invalid records, paths, query limits, or malformed persisted JSON raise
        clear errors.  Empty builds and empty queries are valid and return an
        empty index/result list.
    """

    default_path = Path("data/db/bm25")
    index_filename = "index.json"

    def __init__(
        self,
        path: str | Path = default_path,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        auto_load: bool = True,
    ) -> None:
        if isinstance(path, str) and not path.strip():
            raise ValueError("path must not be empty")
        if isinstance(path, (str, Path)):
            raw_path = Path(path)
        else:
            raise TypeError("path must be a string or Path")
        if isinstance(k1, bool) or not isinstance(k1, (int, float)) or not math.isfinite(float(k1)) or k1 <= 0:
            raise ValueError("k1 must be a positive finite number")
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(float(b)) or not 0 <= b <= 1:
            raise ValueError("b must be a finite number between 0 and 1")

        self.index_path = raw_path if raw_path.suffix.lower() == ".json" else raw_path / self.index_filename
        self.k1 = float(k1)
        self.b = float(b)
        self._tokenizer = SparseEncoder()
        self._records: dict[str, SparseChunkEncoding] = {}
        self._chunk_sources: dict[str, str] = {}
        self._document_frequency: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._postings: dict[str, list[dict[str, Any]]] = {}
        self._document_count = 0
        self._avg_doc_length = 0.0
        if auto_load and self.index_path.exists():
            self._load_from_disk()

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def avg_doc_length(self) -> float:
        return self._avg_doc_length

    @property
    def idf(self) -> dict[str, float]:
        return dict(self._idf)

    @property
    def postings(self) -> dict[str, list[dict[str, Any]]]:
        return {term: [dict(item) for item in values] for term, values in self._postings.items()}

    def build(
        self,
        encoded: SparseEncodingBatch | Sequence[SparseChunkEncoding],
        *,
        source: str | None = None,
        persist: bool = True,
    ) -> None:
        """Replace the whole corpus with encoded records and rebuild postings."""
        records = self._coerce_records(encoded)
        self._records = {}
        self._chunk_sources = {}
        self._replace_records(records, source=source)
        self._rebuild()
        if persist:
            self.save()

    def rebuild(self, encoded: SparseEncodingBatch | Sequence[SparseChunkEncoding], *, source: str | None = None, persist: bool = True) -> None:
        """Explicit alias for ``build`` used by maintenance jobs."""
        self.build(encoded, source=source, persist=persist)

    def upsert(
        self,
        encoded: SparseEncodingBatch | Sequence[SparseChunkEncoding],
        *,
        source: str | None = None,
        persist: bool = True,
    ) -> None:
        """Replace records with matching IDs, then rebuild and optionally save."""
        records = self._coerce_records(encoded)
        self._replace_records(records, source=source)
        self._rebuild()
        if persist:
            self.save()

    def remove_document(self, source: str, *, persist: bool = True) -> int:
        """Remove all records owned by ``source`` and return the removed count."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        normalized = source.strip()
        matching = {
            chunk_id
            for chunk_id, owner in self._chunk_sources.items()
            if owner == normalized
        }
        # Also support callers that used document IDs directly as chunk IDs.
        if not matching and normalized in self._records:
            matching.add(normalized)
        for chunk_id in matching:
            self._records.pop(chunk_id, None)
            self._chunk_sources.pop(chunk_id, None)
        self._rebuild()
        if persist:
            self.save()
        return len(matching)

    def query(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return BM25-ranked chunks as ``[{chunk_id, score}, ...]``."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        terms = self._tokenizer._tokenize(query)
        if not terms or not self._records:
            return []

        query_terms = set(terms)
        scores: dict[str, float] = {}
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for posting in self._postings.get(term, []):
                chunk_id = str(posting["chunk_id"])
                tf = float(posting["tf"])
                doc_length = int(posting["doc_length"])
                normalization = 1.0 - self.b
                if self._avg_doc_length > 0:
                    normalization += self.b * doc_length / self._avg_doc_length
                term_score = idf * (tf * (self.k1 + 1.0)) / (
                    tf + self.k1 * normalization
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"chunk_id": chunk_id, "score": float(score)}
            for chunk_id, score in ranked[:top_k]
        ]

    def save(self, path: str | Path | None = None) -> Path:
        """Persist a deterministic JSON snapshot and return its path."""
        target = self._normalize_index_path(path) if path is not None else self.index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "document_count": self._document_count,
            "avg_doc_length": self._avg_doc_length,
            "document_frequency": dict(sorted(self._document_frequency.items())),
            "idf": dict(sorted(self._idf.items())),
            "postings": {
                term: sorted(values, key=lambda item: str(item["chunk_id"]))
                for term, values in sorted(self._postings.items())
            },
            "chunk_sources": dict(sorted(self._chunk_sources.items())),
            "records": {
                chunk_id: record.to_dict()
                for chunk_id, record in sorted(self._records.items())
            },
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BM25Indexer":
        """Load a previously saved JSON index into a new indexer instance."""
        indexer = cls(path, auto_load=False)
        indexer._load_from_disk()
        return indexer

    def _load_from_disk(self) -> None:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"BM25 index does not exist: {self.index_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid BM25 index JSON: {self.index_path}") from exc
        if not isinstance(payload, Mapping) or payload.get("version") != 1:
            raise ValueError("Unsupported BM25 index format")
        records = payload.get("records", {})
        if not isinstance(records, Mapping):
            raise ValueError("BM25 index records must be a mapping")
        parsed: list[SparseChunkEncoding] = []
        for chunk_id, value in records.items():
            if not isinstance(value, Mapping):
                raise ValueError("BM25 index record must be a mapping")
            parsed.append(self._record_from_mapping(value, fallback_id=str(chunk_id)))
        sources = payload.get("chunk_sources", {})
        if sources is not None and not isinstance(sources, Mapping):
            raise ValueError("BM25 index chunk_sources must be a mapping")
        self._records = {record.chunk_id: record for record in parsed}
        self._chunk_sources = {
            str(key): str(value)
            for key, value in (sources or {}).items()
            if isinstance(value, str)
        }
        self._rebuild()

    def _replace_records(self, records: Sequence[SparseChunkEncoding], *, source: str | None) -> None:
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError("source must be a non-empty string or None")
        for record in records:
            self._records[record.chunk_id] = record
            if source is not None:
                self._chunk_sources[record.chunk_id] = source.strip()

    def _rebuild(self) -> None:
        self._document_count = len(self._records)
        self._document_frequency = {}
        total_length = 0
        postings: dict[str, list[dict[str, Any]]] = {}
        for record in self._records.values():
            total_length += record.doc_length
            for term, tf in record.term_weights.items():
                self._document_frequency[term] = self._document_frequency.get(term, 0) + 1
                postings.setdefault(term, []).append(
                    {"chunk_id": record.chunk_id, "tf": float(tf), "doc_length": record.doc_length}
                )
        self._avg_doc_length = total_length / self._document_count if self._document_count else 0.0
        self._idf = {
            term: math.log(
                (self._document_count - df + 0.5) / (df + 0.5)
            )
            for term, df in self._document_frequency.items()
        }
        self._postings = {
            term: sorted(values, key=lambda item: str(item["chunk_id"]))
            for term, values in sorted(postings.items())
        }

    @staticmethod
    def _coerce_records(
        encoded: SparseEncodingBatch | Sequence[SparseChunkEncoding],
    ) -> list[SparseChunkEncoding]:
        if isinstance(encoded, SparseEncodingBatch):
            records = list(encoded.records)
        elif isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes)):
            records = list(encoded)
        else:
            raise TypeError("encoded must be SparseEncodingBatch or a sequence")
        seen: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, SparseChunkEncoding):
                raise TypeError(f"encoded[{index}] must be SparseChunkEncoding")
            if record.chunk_id in seen:
                raise ValueError(f"encoded contains duplicate chunk_id '{record.chunk_id}'")
            seen.add(record.chunk_id)
        return records

    @staticmethod
    def _record_from_mapping(value: Mapping[str, Any], *, fallback_id: str) -> SparseChunkEncoding:
        chunk_id = value.get("chunk_id", fallback_id)
        weights = value.get("term_weights")
        doc_length = value.get("doc_length")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("BM25 record chunk_id must be a non-empty string")
        if not isinstance(weights, Mapping):
            raise ValueError("BM25 record term_weights must be a mapping")
        if isinstance(doc_length, bool) or not isinstance(doc_length, int) or doc_length < 0:
            raise ValueError("BM25 record doc_length must be a non-negative integer")
        normalized: dict[str, float] = {}
        for term, tf in weights.items():
            if not isinstance(term, str) or not term.strip():
                raise ValueError("BM25 record terms must be non-empty strings")
            if isinstance(tf, bool) or not isinstance(tf, (int, float)) or not math.isfinite(float(tf)) or tf < 0:
                raise ValueError("BM25 record term weights must be finite non-negative numbers")
            normalized[term] = float(tf)
        return SparseChunkEncoding(chunk_id=chunk_id, term_weights=normalized, doc_length=doc_length)

    @staticmethod
    def _normalize_index_path(path: str | Path) -> Path:
        if not isinstance(path, (str, Path)) or (isinstance(path, str) and not path.strip()):
            raise ValueError("path must be a non-empty string or Path")
        raw = Path(path)
        return raw if raw.suffix.lower() == ".json" else raw / BM25Indexer.index_filename


__all__ = ["BM25Indexer"]
