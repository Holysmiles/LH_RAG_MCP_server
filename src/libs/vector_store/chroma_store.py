"""ChromaDB adapter for the project's vector-store contract.

The dependency chain is intentionally short::

    settings.yaml
      -> Settings.vector_store
      -> VectorStoreFactory
      -> ChromaStore
      -> chromadb.PersistentClient

ChromaDB does the difficult database work.  This adapter only translates the
project's stable record format into Chroma's API and translates Chroma's query
response back into :class:`VectorSearchResult` dictionaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import chromadb

from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    VectorSearchResult,
)

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


__all__ = ["ChromaStore"]


class ChromaStore(BaseVectorStore):
    """Store dense vectors and their source text in a local Chroma database.

    One project record looks like this::

        {
            "id": "chunk-001",
            "text": "The original chunk text",
            "metadata": {"source_path": "guide.pdf"},
            "dense_vector": [0.12, 0.34, ...],
        }

    Chroma expects the same data as four parallel lists.  Item zero in every
    list belongs to the same record.  ``upsert`` performs that conversion;
    ``query`` converts Chroma's nested batch response in the opposite direction.
    """

    provider_name: ClassVar[str] = "chroma"
    default_persist_directory: ClassVar[str] = "./data/db/chroma"
    default_collection_name: ClassVar[str] = "knowledge_hub"

    def __init__(self, config: Mapping[str, object]) -> None:
        """Open the configured local database and select one collection.

        ``PersistentClient`` stores its files under ``persist_directory``.
        Creating another ``ChromaStore`` with the same directory and collection
        therefore opens the same data instead of creating an in-memory database.
        """
        if not isinstance(config, Mapping):
            self._raise_configuration("configuration must be a mapping")

        # Keep a private copy so later mutations to Settings cannot silently
        # change the meaning of an already-created store.
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

        # The application supplies embeddings itself, so embedding_function=None
        # prevents Chroma from downloading or running a second embedding model.
        # Cosine distance matches the semantic-vector retrieval design in DEV_SPEC.
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
        """Insert new records or replace records that have the same ID.

        Chroma receives one batch call rather than one call per record.  This is
        both faster and safer: the four lists below are built together, so their
        positions cannot drift apart.
        """
        del trace
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            self._raise_validation("records must be a sequence of mappings")
        if not records:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        # Chroma uses None to represent an empty metadata dictionary.  Passing
        # an actual {} raises an error in ChromaDB 1.x.
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
            metadatas.append(dict(metadata) or None)

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
        """Return the ``top_k`` records nearest to one query vector.

        Chroma returns *distance*, where a smaller number is better.  The rest
        of this project uses *score*, where a larger number is better.  Because
        this collection uses cosine distance, ``score = 1 - distance`` converts
        the value back to cosine similarity.
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

    @classmethod
    def _read_vector(cls, value: object, *, field: str) -> list[float]:
        """Validate one vector and convert every number to a plain float."""
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
        """Translate simple project equality filters into Chroma syntax.

        Chroma accepts ``{"collection": "docs"}`` for one condition, but
        multiple conditions must be wrapped in ``$and``.  Callers should not
        need to know that provider-specific detail.
        """
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
        """Flatten Chroma's one-query batch response into project results."""
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

            results.append(
                {
                    "chunk_id": chunk_id,
                    "score": 1.0 - float(distance),
                    "text": document,
                    "metadata": dict(metadata) if metadata is not None else {},
                }
            )
        return results

    @staticmethod
    def _first_batch(response: Mapping[str, Any], field: str) -> list[Any]:
        """Read the first nested result list produced by one query vector."""
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
    def _raise_configuration(cls, detail: str) -> None:
        raise ValueError(f"{cls.provider_name} configuration error: {detail}")

    @classmethod
    def _raise_validation(cls, detail: str) -> None:
        raise ValueError(f"{cls.provider_name} validation error: {detail}")
