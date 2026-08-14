"""为稀疏编码记录提供可持久化的 BM25 倒排索引。

整体链路：``SparseEncodingBatch -> BM25Indexer -> JSON 索引 -> 排序结果``。

本模块负责语料级 IDF 和倒排表，SparseEncoder 负责分词与词频，查询必须复用同一套
分词规则。更新采用“替换内存记录后整体重建”，计算量略高，但能从根源上保证新增、
覆盖或删除文档后不会残留旧 posting。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from ingestion.embedding.sparse_encoder import (
    SparseChunkEncoding,
    SparseEncoder,
    SparseEncodingBatch,
)


class BM25Indexer:
    """构建、保存、更新、删除并查询本地 BM25 索引。

    输入：``SparseEncodingBatch`` 或 SparseChunkEncoding 序列；可选 ``source`` 用于
    记录 Chunk 属于哪篇源文档。输出：包含参数、记录、IDF、倒排表和归属关系的 JSON，
    查询返回按分数排序的 ``chunk_id``。失败边界：非法记录、路径、查询数量或被篡改的
    JSON 立即报错；空索引与空查询是合法状态。保存使用同目录临时文件原子替换，进程
    中断不会把半截 JSON 暴露为正式索引。
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
        self.index_path = raw_path if raw_path.suffix.lower() == ".json" else raw_path / self.index_filename
        self.k1 = self._validate_k1(k1)
        self.b = self._validate_b(b)
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
        """用输入记录替换整个语料，并重新计算所有倒排数据。"""
        records = self._coerce_records(encoded)
        self._records = {}
        self._chunk_sources = {}
        self._replace_records(records, source=source)
        self._rebuild()
        if persist:
            self.save()

    def rebuild(self, encoded: SparseEncodingBatch | Sequence[SparseChunkEncoding], *, source: str | None = None, persist: bool = True) -> None:
        """供维护任务显式表达“重建索引”的 ``build`` 别名。"""
        self.build(encoded, source=source, persist=persist)

    def upsert(
        self,
        encoded: SparseEncodingBatch | Sequence[SparseChunkEncoding],
        *,
        source: str | None = None,
        persist: bool = True,
    ) -> None:
        """按 ID 覆盖记录，随后重建索引并按需保存。"""
        records = self._coerce_records(encoded)
        self._replace_records(records, source=source)
        self._rebuild()
        if persist:
            self.save()

    def remove_document(self, source: str, *, persist: bool = True) -> int:
        """删除属于 ``source`` 的全部记录，并返回删除数量。"""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        normalized = source.strip()
        matching = {
            chunk_id
            for chunk_id, owner in self._chunk_sources.items()
            if owner == normalized
        }
        # 兼容早期直接把文档 ID 当作 Chunk ID 的调用方式。
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
        """返回 BM25 排序结果：``[{chunk_id, score}, ...]``。"""
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
        """原子保存确定性 JSON 快照，并返回正式索引路径。"""
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
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(serialized)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, target)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BM25Indexer":
        """从 JSON 快照恢复一个新的索引器实例。"""
        indexer = cls(path, auto_load=False)
        indexer._load_from_disk()
        return indexer

    def _load_from_disk(self) -> None:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"BM25 index does not exist: {self.index_path}") from exc
        except OSError as exc:
            raise OSError(f"Could not read BM25 index: {self.index_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid BM25 index JSON: {self.index_path}") from exc
        if not isinstance(payload, Mapping) or payload.get("version") != 1:
            raise ValueError("Unsupported BM25 index format")
        self.k1 = self._validate_k1(payload.get("k1"))
        self.b = self._validate_b(payload.get("b"))
        records = payload.get("records", {})
        if not isinstance(records, Mapping):
            raise ValueError("BM25 index records must be a mapping")
        parsed: list[SparseChunkEncoding] = []
        for chunk_id, value in records.items():
            if not isinstance(value, Mapping):
                raise ValueError("BM25 index record must be a mapping")
            record = self._record_from_mapping(value, fallback_id=str(chunk_id))
            if record.chunk_id != str(chunk_id):
                raise ValueError("BM25 record key and chunk_id must match")
            parsed.append(record)
        sources = payload.get("chunk_sources", {})
        if sources is not None and not isinstance(sources, Mapping):
            raise ValueError("BM25 index chunk_sources must be a mapping")
        self._records = {record.chunk_id: record for record in parsed}
        normalized_sources: dict[str, str] = {}
        for key, value in (sources or {}).items():
            if not isinstance(key, str) or key not in self._records:
                raise ValueError("BM25 chunk_sources contains an unknown chunk_id")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("BM25 chunk_sources values must be non-empty strings")
            normalized_sources[key] = value.strip()
        self._chunk_sources = normalized_sources
        self._rebuild()
        self._validate_persisted_statistics(payload)

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
            BM25Indexer._validate_record(record)
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
            if isinstance(tf, bool) or not isinstance(tf, (int, float)) or not math.isfinite(float(tf)) or tf <= 0:
                raise ValueError("BM25 record term weights must be finite positive numbers")
            normalized[term] = float(tf)
        record = SparseChunkEncoding(
            chunk_id=chunk_id,
            term_weights=normalized,
            doc_length=doc_length,
        )
        BM25Indexer._validate_record(record)
        return record

    @staticmethod
    def _validate_record(record: SparseChunkEncoding) -> None:
        """校验词频记录内部是否自洽，拒绝无法由分词结果产生的数据。"""
        if not isinstance(record.chunk_id, str) or not record.chunk_id.strip():
            raise ValueError("BM25 record chunk_id must be a non-empty string")
        if isinstance(record.doc_length, bool) or not isinstance(record.doc_length, int) or record.doc_length < 0:
            raise ValueError("BM25 record doc_length must be a non-negative integer")
        total_frequency = 0.0
        for term, tf in record.term_weights.items():
            if not isinstance(term, str) or not term.strip():
                raise ValueError("BM25 record terms must be non-empty strings")
            if isinstance(tf, bool) or not isinstance(tf, (int, float)) or not math.isfinite(float(tf)) or tf <= 0:
                raise ValueError("BM25 record term weights must be finite positive numbers")
            total_frequency += float(tf)
        if not math.isclose(total_frequency, float(record.doc_length)):
            raise ValueError("BM25 record term weights must sum to doc_length")

    def _validate_persisted_statistics(self, payload: Mapping[str, Any]) -> None:
        """重新计算并核对派生统计，防止加载部分损坏或被篡改的快照。"""
        document_count = payload.get("document_count")
        if isinstance(document_count, bool) or not isinstance(document_count, int):
            raise ValueError("BM25 document_count must be an integer")
        if document_count != self._document_count:
            raise ValueError("BM25 document_count does not match records")

        avg_doc_length = payload.get("avg_doc_length")
        if isinstance(avg_doc_length, bool) or not isinstance(avg_doc_length, (int, float)):
            raise ValueError("BM25 avg_doc_length must be numeric")
        if not math.isfinite(float(avg_doc_length)) or not math.isclose(
            float(avg_doc_length), self._avg_doc_length
        ):
            raise ValueError("BM25 avg_doc_length does not match records")

        document_frequency = payload.get("document_frequency")
        if not isinstance(document_frequency, Mapping):
            raise ValueError("BM25 document_frequency must be a mapping")
        if dict(document_frequency) != self._document_frequency:
            raise ValueError("BM25 document_frequency does not match records")

        persisted_idf = payload.get("idf")
        if not isinstance(persisted_idf, Mapping) or set(persisted_idf) != set(self._idf):
            raise ValueError("BM25 idf does not match records")
        for term, expected in self._idf.items():
            value = persisted_idf[term]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("BM25 idf values must be numeric")
            if not math.isfinite(float(value)) or not math.isclose(float(value), expected):
                raise ValueError("BM25 idf does not match records")

        if payload.get("postings") != self._postings:
            raise ValueError("BM25 postings do not match records")

    @staticmethod
    def _validate_k1(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("k1 must be a positive finite number")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError("k1 must be a positive finite number")
        return normalized

    @staticmethod
    def _validate_b(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("b must be a finite number between 0 and 1")
        normalized = float(value)
        if not math.isfinite(normalized) or not 0 <= normalized <= 1:
            raise ValueError("b must be a finite number between 0 and 1")
        return normalized

    @staticmethod
    def _normalize_index_path(path: str | Path) -> Path:
        if not isinstance(path, (str, Path)) or (isinstance(path, str) and not path.strip()):
            raise ValueError("path must be a non-empty string or Path")
        raw = Path(path)
        return raw if raw.suffix.lower() == ".json" else raw / BM25Indexer.index_filename


__all__ = ["BM25Indexer"]
