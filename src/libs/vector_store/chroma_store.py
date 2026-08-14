"""项目向量存储契约的 ChromaDB 适配器。

依赖链保持简单：``Settings -> VectorStoreFactory -> ChromaStore -> ChromaDB``。
ChromaDB 负责数据库能力，本适配器只做两件事：把项目稳定记录转换成 Chroma 的并行
列表输入，再把 Chroma 的嵌套查询响应还原为 ``VectorSearchResult``。复杂 metadata
会被完整编码进保留 JSON 字段，同时保留顶层标量供数据库过滤。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

import chromadb

from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    StoredVectorRecord,
    VectorRecord,
    VectorSearchResult,
)

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


__all__ = ["ChromaStore"]


class ChromaStore(BaseVectorStore):
    """把稠密向量、原文和元数据存入本地 Chroma 数据库。

    单条项目记录形状如下：

        {
            "id": "chunk-001",
            "text": "原始 Chunk 文本",
            "metadata": {"source_path": "guide.pdf"},
            "dense_vector": [0.12, 0.34, ...],
        }

    Chroma 要求四个同序列表，每个列表相同下标共同构成一条记录。``upsert`` 完成正向
    转换，``query`` 负责反向还原，并对不可信的数据库响应再次校验。
    """

    provider_name: ClassVar[str] = "chroma"
    default_persist_directory: ClassVar[str] = "./data/db/chroma"
    default_collection_name: ClassVar[str] = "knowledge_hub"
    _metadata_json_field: ClassVar[str] = "__rag_metadata_json"

    def __init__(self, config: Mapping[str, object]) -> None:
        """打开配置指定的持久化数据库和 Collection。"""
        if not isinstance(config, Mapping):
            self._raise_configuration("configuration must be a mapping")

        # 保存配置副本，避免外部后续修改 Settings 时悄悄改变已创建 Store 的含义。
        self.config = dict(config)
        persist_directory = self._read_non_empty_string(
            config,
            "persist_directory",
            self.default_persist_directory,
        )
        self.collection_name = self._read_non_empty_string(
            config,
            "collection_name",
            self.default_collection_name,
        )
        self.persist_directory = Path(persist_directory).expanduser()

        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # 向量由应用自己生成；禁用 Chroma 内置模型可避免重复下载和重复编码。
        # Collection 使用余弦距离，与项目语义检索的分数定义保持一致。
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def upsert(
        self,
        records: Sequence[VectorRecord],
        trace: TraceContext | None = None,
    ) -> None:
        """插入新记录或按相同 ID 覆盖旧记录；整个输入只发起一次批调用。"""
        del trace
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            self._raise_validation("records must be a sequence of mappings")
        if not records:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        # 即使原 metadata 为空，也会写入完整 JSON 保留字段，因此这里通常不会产生 {}。
        metadatas: list[dict[str, Any] | None] = []
        seen_ids: set[str] = set()
        vector_size: int | None = None

        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                self._raise_validation(f"records[{index}] must be a mapping")

            missing_fields = {
                "id",
                "text",
                "metadata",
                "dense_vector",
            }.difference(record)
            if missing_fields:
                self._raise_validation(
                    f"records[{index}] is missing fields: "
                    f"{sorted(missing_fields)}"
                )

            chunk_id = record["id"]
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                self._raise_validation(
                    f"records[{index}].id must be a non-empty string"
                )
            if chunk_id in seen_ids:
                self._raise_validation(
                    f"records contains duplicate id '{chunk_id}'"
                )
            seen_ids.add(chunk_id)

            text = record["text"]
            if not isinstance(text, str):
                self._raise_validation(
                    f"records[{index}].text must be a string"
                )

            metadata = record["metadata"]
            if not isinstance(metadata, Mapping):
                self._raise_validation(
                    f"records[{index}].metadata must be a mapping"
                )

            vector = self._read_vector(
                record["dense_vector"],
                field=f"records[{index}].dense_vector",
            )
            if vector_size is None:
                vector_size = len(vector)
            elif len(vector) != vector_size:
                self._raise_validation(
                    "all dense vectors in one batch must have the same length"
                )

            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(text)
            metadatas.append(self._encode_metadata(metadata))

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[VectorSearchResult]:
        """返回与查询向量最接近的 ``top_k`` 条记录。

        Chroma 返回越小越好的 distance，项目返回越大越好的 score。余弦距离下使用
        ``score = 1 - distance`` 即可转换回余弦相似度。
        """
        del trace
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            self._raise_validation("top_k must be a positive integer")

        query_vector = self._read_vector(vector, field="vector")
        where = self._build_where(filters)
        response = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._convert_query_response(response)

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: TraceContext | None = None,
    ) -> list[StoredVectorRecord]:
        """批量补全 Sparse 命中的正文和 metadata，并恢复调用方请求顺序。"""
        del trace
        normalized_ids = self._validate_record_ids(ids)
        if not normalized_ids:
            return []
        response = self._collection.get(
            ids=normalized_ids,
            include=["documents", "metadatas"],
        )
        return self._convert_get_response(response, normalized_ids)

    @classmethod
    def _encode_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把任意 JSON metadata 转成 Chroma 可接受的标量字典。

        输入是项目完整 metadata；输出保留所有顶层标量用于过滤，并额外保存一份确定性
        JSON。列表、字典和 ``None`` 不直接交给 Chroma，查询时从 JSON 无损恢复。
        """
        complete = dict(metadata)
        try:
            serialized = json.dumps(
                complete,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            cls._raise_validation(
                f"metadata must contain finite JSON-compatible values: {exc}"
            )

        encoded: dict[str, Any] = {cls._metadata_json_field: serialized}
        for key, value in complete.items():
            if not isinstance(key, str) or not key.strip():
                continue
            if key == cls._metadata_json_field:
                continue
            if isinstance(value, bool) or isinstance(value, (str, int)):
                encoded[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                encoded[key] = value
        return encoded

    @classmethod
    def _decode_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """优先从保留 JSON 字段恢复完整 metadata，并兼容旧版标量记录。"""
        if metadata is None:
            return {}
        raw = metadata.get(cls._metadata_json_field)
        if raw is None:
            return dict(metadata)
        if not isinstance(raw, str):
            raise RuntimeError(
                "chroma response error: encoded metadata must be a string"
            )
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "chroma response error: encoded metadata is invalid JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise RuntimeError(
                "chroma response error: encoded metadata must be an object"
            )
        return dict(decoded)

    @classmethod
    def _read_vector(cls, value: object, *, field: str) -> list[float]:
        """校验一个向量，并把所有实数规范成普通 float。"""
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            cls._raise_validation(f"{field} must be a sequence of numbers")
        if not value:
            cls._raise_validation(f"{field} must not be empty")

        vector: list[float] = []
        for position, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, Real):
                cls._raise_validation(
                    f"{field}[{position}] must be a real number"
                )
            number = float(item)
            if not math.isfinite(number):
                cls._raise_validation(
                    f"{field}[{position}] must be finite"
                )
            vector.append(number)
        return vector

    @classmethod
    def _build_where(
        cls,
        filters: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """把项目的简单等值过滤条件转换成 Chroma 的 ``$and`` 语法。"""
        if filters is None:
            return None
        if not isinstance(filters, Mapping):
            cls._raise_validation("filters must be a mapping or None")
        if not filters:
            return None

        conditions: list[dict[str, Any]] = []
        for key, value in filters.items():
            if not isinstance(key, str) or not key.strip():
                cls._raise_validation(
                    "filter keys must be non-empty strings"
                )
            if key == cls._metadata_json_field:
                cls._raise_validation("filters cannot access reserved metadata fields")
            if isinstance(value, float) and not math.isfinite(value):
                cls._raise_validation(
                    f"filter '{key}' must contain a finite value"
                )
            if not isinstance(value, (str, int, float, bool)):
                cls._raise_validation(
                    f"filter '{key}' must use a string, number, or boolean value"
                )
            conditions.append({key: value})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @classmethod
    def _convert_query_response(
        cls,
        response: Mapping[str, Any],
    ) -> list[VectorSearchResult]:
        """把 Chroma 单查询的嵌套批响应展开为项目结果。"""
        ids = cls._first_batch(response, "ids")
        documents = cls._first_batch(response, "documents")
        metadatas = cls._first_batch(response, "metadatas")
        distances = cls._first_batch(response, "distances")

        expected_size = len(ids)
        if not all(
            len(items) == expected_size
            for items in (documents, metadatas, distances)
        ):
            raise RuntimeError(
                "chroma response error: query fields have different lengths"
            )

        results: list[VectorSearchResult] = []
        for index, chunk_id in enumerate(ids):
            document = documents[index]
            metadata = metadatas[index]
            distance = distances[index]
            if not isinstance(chunk_id, str) or not isinstance(document, str):
                raise RuntimeError(
                    "chroma response error: invalid id or document"
                )
            if metadata is not None and not isinstance(metadata, Mapping):
                raise RuntimeError(
                    "chroma response error: metadata must be a mapping or None"
                )
            if isinstance(distance, bool) or not isinstance(distance, Real):
                raise RuntimeError(
                    "chroma response error: distance must be numeric"
                )
            if not math.isfinite(float(distance)):
                raise RuntimeError(
                    "chroma response error: distance must be finite"
                )

            results.append(
                {
                    "chunk_id": chunk_id,
                    "score": 1.0 - float(distance),
                    "text": document,
                    "metadata": cls._decode_metadata(metadata),
                }
            )
        return results

    @classmethod
    def _convert_get_response(
        cls,
        response: Mapping[str, Any],
        requested_ids: Sequence[str],
    ) -> list[StoredVectorRecord]:
        """校验 Chroma 的扁平 get 响应，并按 requested_ids 重排。"""
        if not isinstance(response, Mapping):
            raise RuntimeError("chroma response error: get response must be a mapping")
        ids = response.get("ids")
        documents = response.get("documents")
        metadatas = response.get("metadatas")
        if not all(isinstance(items, list) for items in (ids, documents, metadatas)):
            raise RuntimeError("chroma response error: invalid get response fields")
        if not (len(ids) == len(documents) == len(metadatas)):
            raise RuntimeError(
                "chroma response error: get fields have different lengths"
            )

        requested = set(requested_ids)
        by_id: dict[str, StoredVectorRecord] = {}
        for index, chunk_id in enumerate(ids):
            document = documents[index]
            metadata = metadatas[index]
            if not isinstance(chunk_id, str) or not isinstance(document, str):
                raise RuntimeError("chroma response error: invalid get id or document")
            if chunk_id not in requested:
                raise RuntimeError("chroma response error: get returned an unknown id")
            if chunk_id in by_id:
                raise RuntimeError("chroma response error: get returned a duplicate id")
            if metadata is not None and not isinstance(metadata, Mapping):
                raise RuntimeError(
                    "chroma response error: get metadata must be a mapping or None"
                )
            by_id[chunk_id] = {
                "id": chunk_id,
                "text": document,
                "metadata": cls._decode_metadata(metadata),
            }
        return [by_id[chunk_id] for chunk_id in requested_ids if chunk_id in by_id]

    @classmethod
    def _validate_record_ids(cls, ids: object) -> list[str]:
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
            cls._raise_validation("ids must be a non-string sequence")
        normalized: list[str] = []
        seen: set[str] = set()
        for index, chunk_id in enumerate(ids):
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                cls._raise_validation(f"ids[{index}] must be a non-empty string")
            value = chunk_id.strip()
            if value in seen:
                cls._raise_validation(f"ids contains duplicate id '{value}'")
            seen.add(value)
            normalized.append(value)
        return normalized

    @staticmethod
    def _first_batch(response: Mapping[str, Any], field: str) -> list[Any]:
        """读取单个查询向量对应的第一层结果列表。"""
        batches = response.get(field)
        if not isinstance(batches, list) or not batches:
            raise RuntimeError(
                f"chroma response error: missing query field '{field}'"
            )
        first_batch = batches[0]
        if not isinstance(first_batch, list):
            raise RuntimeError(
                f"chroma response error: field '{field}' is not a result list"
            )
        return first_batch

    @classmethod
    def _read_non_empty_string(
        cls,
        config: Mapping[str, object],
        field: str,
        default: str,
    ) -> str:
        value = config.get(field, default)
        if not isinstance(value, str) or not value.strip():
            cls._raise_configuration(f"{field} must be a non-empty string")
        return value.strip()

    @classmethod
    def _raise_configuration(cls, detail: str) -> NoReturn:
        raise ValueError(f"{cls.provider_name} configuration error: {detail}")

    @classmethod
    def _raise_validation(cls, detail: str) -> NoReturn:
        raise ValueError(f"{cls.provider_name} validation error: {detail}")
