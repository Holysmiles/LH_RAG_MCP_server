"""离线摄取主链路的事务边界与阶段编排。

Pipeline 不实现 PDF 解析、切分、增强、编码或数据库算法；它只回答三个系统级问题：

1. 一个文件是否需要处理；
2. 各阶段必须以什么顺序交换数据；
3. 什么时候可以把整个文件声明为成功。

完整路径是 ``integrity -> load -> split -> transform -> encode -> store``。Chroma、
BM25、图片目录和 SQLite 无法共享一个物理事务，因此这里采用可重试的一致性策略：
所有写入使用稳定 ID，任一步失败都把完整性状态记为 ``failed``，只有全部存储成功才
提交 ``success``。重试会通过 upsert 覆盖同一批记录，而不会制造重复数据。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Literal

from core import Chunk, ChunkRecord, Document
from core.settings import Settings
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessResult, BatchProcessor
from ingestion.embedding.sparse_encoder import SparseChunkEncoding
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import (
    DEFAULT_INTEGRITY_DB_PATH,
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)
from libs.loader.pdf_loader import PdfLoader
from observability.logger import get_logger


_DEFAULT_IMAGE_ROOT = Path("data/images")
_DEFAULT_IMAGE_INDEX_PATH = Path("data/db/image_index.db")
_SAFE_COLLECTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class IngestionPipelineError(RuntimeError):
    """指出摄取失败的准确阶段，同时保留原始异常作为 ``__cause__``。"""

    def __init__(self, stage: str, source_path: Path, cause: Exception) -> None:
        self.stage = stage
        self.source_path = source_path
        self.cause = cause
        super().__init__(
            f"Ingestion pipeline failed at stage '{stage}' for "
            f"'{source_path}': {type(cause).__name__}: {cause}"
        )


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """一次文件摄取的轻量结果，不把向量等大对象泄漏给 CLI/Dashboard。"""

    source_path: str
    collection: str
    file_hash: str
    status: Literal["success", "skipped"]
    document_id: str | None
    chunk_count: int
    image_count: int
    batch_count: int
    record_ids: tuple[str, ...]

    @property
    def skipped(self) -> bool:
        return self.status == "skipped"

    def to_dict(self) -> dict[str, Any]:
        """返回可直接 JSON 序列化的运行摘要。"""
        return {
            "source_path": self.source_path,
            "collection": self.collection,
            "file_hash": self.file_hash,
            "status": self.status,
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "batch_count": self.batch_count,
            "record_ids": list(self.record_ids),
        }


class IngestionPipeline:
    """按稳定契约串联摄取组件，并集中管理成功/失败状态。

    默认组件按需创建：如果完整性检查发现文件未变化，Pipeline 不会初始化 Chroma、
    Embedding 或 LLM。测试和特殊部署可以注入任意已实现项目接口的组件。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        integrity_checker: FileIntegrityChecker | None = None,
        loader: BaseLoader | None = None,
        chunker: DocumentChunker | None = None,
        transforms: Sequence[BaseTransform] | None = None,
        batch_processor: BatchProcessor | None = None,
        vector_upserter: VectorUpserter | None = None,
        bm25_indexer: BM25Indexer | None = None,
        image_storage: ImageStorage | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings instance")
        if not isinstance(settings.ingestion, Mapping):
            raise TypeError("configuration section 'ingestion' must be a mapping")
        self.settings = settings
        self._integrity_checker = self._optional_instance(
            integrity_checker, FileIntegrityChecker, "integrity_checker"
        )
        self._loader = self._optional_instance(loader, BaseLoader, "loader")
        self._chunker = self._optional_instance(chunker, DocumentChunker, "chunker")
        self._batch_processor = self._optional_instance(
            batch_processor, BatchProcessor, "batch_processor"
        )
        self._vector_upserter = self._optional_instance(
            vector_upserter, VectorUpserter, "vector_upserter"
        )
        self._bm25_indexer = self._optional_instance(
            bm25_indexer, BM25Indexer, "bm25_indexer"
        )
        self._image_storage = self._optional_instance(
            image_storage, ImageStorage, "image_storage"
        )
        self._transforms = self._validate_transforms(transforms)

        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("logger must be logging.Logger or None")
        log_level = settings.observability.get("log_level", logging.INFO)
        self.logger = logger or get_logger("ingestion.pipeline", log_level)

    def run(
        self,
        source_path: str | Path,
        collection: str = "default",
        *,
        force: bool = False,
    ) -> IngestionResult:
        """摄取一个文件；除非 ``force``，未变化时返回 ``skipped``。"""
        source = self._normalize_source_path(source_path)
        normalized_collection = self._normalize_collection(collection)
        if not isinstance(force, bool):
            raise TypeError("force must be a boolean")
        checker = self._get_integrity_checker()
        file_hash: str | None = None
        marked_processing = False
        stage = "integrity"

        try:
            self._log_stage(stage, "started", source, normalized_collection)
            file_hash = checker.compute_sha256(source)
            if checker.should_skip(file_hash) and not force:
                self._log_stage(stage, "skipped", source, normalized_collection)
                return IngestionResult(
                    source_path=str(source),
                    collection=normalized_collection,
                    file_hash=file_hash,
                    status="skipped",
                    document_id=None,
                    chunk_count=0,
                    image_count=0,
                    batch_count=0,
                    record_ids=(),
                )
            if force:
                self._log_stage(stage, "forced", source, normalized_collection)

            file_size = source.stat().st_size
            checker.mark_processing(file_hash, source, file_size=file_size)
            marked_processing = True
            self._log_stage(stage, "completed", source, normalized_collection)

            stage = "load"
            self._log_stage(stage, "started", source, normalized_collection)
            document = self._get_loader().load(source)
            document = self._prepare_document(
                document,
                collection=normalized_collection,
                file_hash=file_hash,
            )
            self._log_stage(stage, "completed", source, normalized_collection)

            stage = "split"
            self._log_stage(stage, "started", source, normalized_collection)
            chunks = self._get_chunker().split_document(document)
            if not chunks:
                raise ValueError("document produced no chunks")
            self._log_stage(
                stage,
                "completed",
                source,
                normalized_collection,
                chunk_count=len(chunks),
            )

            for transform in self._get_transforms():
                transform_name = type(transform).__name__
                stage = f"transform.{transform_name}"
                self._log_stage(stage, "started", source, normalized_collection)
                transformed = transform.transform(chunks)
                self._validate_transform_output(chunks, transformed, transform_name)
                chunks = transformed
                self._log_stage(
                    stage,
                    "completed",
                    source,
                    normalized_collection,
                    chunk_count=len(chunks),
                )

            stage = "encode"
            self._log_stage(stage, "started", source, normalized_collection)
            encoded = self._get_batch_processor().process(chunks)
            self._validate_encoding_alignment(chunks, encoded)
            self._log_stage(
                stage,
                "completed",
                source,
                normalized_collection,
                batch_count=encoded.batch_count,
            )

            # 图片先进入托管目录，随后写入 Chroma 的 metadata 才会指向最终路径。
            stage = "store.images"
            self._log_stage(stage, "started", source, normalized_collection)
            stored_chunks, image_count = self._store_images(
                document,
                chunks,
                normalized_collection,
            )
            self._log_stage(
                stage,
                "completed",
                source,
                normalized_collection,
                image_count=image_count,
            )

            stage = "store.vector"
            self._log_stage(stage, "started", source, normalized_collection)
            vector_records = self._get_vector_upserter().upsert(
                stored_chunks,
                encoded.dense_vectors,
            )
            self._validate_vector_records(stored_chunks, vector_records)
            self._log_stage(
                stage,
                "completed",
                source,
                normalized_collection,
                record_count=len(vector_records),
            )

            stage = "store.bm25"
            self._log_stage(stage, "started", source, normalized_collection)
            sparse_records = self._align_sparse_ids(
                chunks,
                encoded.sparse_records,
                vector_records,
            )
            self._get_bm25_indexer().upsert(
                sparse_records,
                source=str(document.metadata["source_path"]),
            )
            self._log_stage(
                stage,
                "completed",
                source,
                normalized_collection,
                record_count=len(sparse_records),
            )

            stage = "integrity.commit"
            checker.mark_success(
                file_hash,
                source,
                file_size=file_size,
                chunk_count=len(chunks),
            )
            self._log_stage(stage, "completed", source, normalized_collection)
            return IngestionResult(
                source_path=str(source),
                collection=normalized_collection,
                file_hash=file_hash,
                status="success",
                document_id=document.id,
                chunk_count=len(chunks),
                image_count=image_count,
                batch_count=encoded.batch_count,
                record_ids=tuple(record.id for record in vector_records),
            )
        except Exception as exc:
            if marked_processing and file_hash is not None:
                self._mark_failed_without_masking(
                    checker,
                    file_hash,
                    source,
                    exc,
                )
            self.logger.error(
                "ingestion stage=%s status=failed source=%s collection=%s error=%s",
                stage,
                source,
                normalized_collection,
                type(exc).__name__,
            )
            raise IngestionPipelineError(stage, source, exc) from exc

    @staticmethod
    def _prepare_document(
        document: Document,
        *,
        collection: str,
        file_hash: str,
    ) -> Document:
        if not isinstance(document, Document):
            raise TypeError("loader must return a Document")
        metadata = deepcopy(document.metadata)
        metadata["collection"] = collection
        metadata["file_hash"] = file_hash
        metadata["document_id"] = document.id
        return Document(id=document.id, text=document.text, metadata=metadata)

    @staticmethod
    def _validate_transform_output(
        before: Sequence[Chunk],
        after: object,
        transform_name: str,
    ) -> None:
        """Transform 可以改正文/元数据，但不能破坏 Chunk 的身份与溯源顺序。"""
        if not isinstance(after, list):
            raise TypeError(f"{transform_name}.transform must return list[Chunk]")
        if len(after) != len(before):
            raise ValueError(
                f"{transform_name} changed chunk count: {len(before)} -> {len(after)}"
            )
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if not isinstance(new, Chunk):
                raise TypeError(f"{transform_name} output[{index}] must be a Chunk")
            old_identity = (
                old.id,
                old.start_offset,
                old.end_offset,
                old.source_ref,
            )
            new_identity = (
                new.id,
                new.start_offset,
                new.end_offset,
                new.source_ref,
            )
            if new_identity != old_identity:
                raise ValueError(
                    f"{transform_name} changed chunk identity at index {index}"
                )

    @staticmethod
    def _validate_encoding_alignment(
        chunks: Sequence[Chunk],
        encoded: BatchProcessResult,
    ) -> None:
        if not isinstance(encoded, BatchProcessResult):
            raise TypeError("batch processor must return BatchProcessResult")
        if len(encoded.chunks) != len(chunks):
            raise ValueError("batch processor changed chunk count")
        if [item.id for item in encoded.chunks] != [item.id for item in chunks]:
            raise ValueError("batch processor changed chunk order or identity")

    def _store_images(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        collection: str,
    ) -> tuple[list[Chunk], int]:
        raw_images = document.metadata.get("images", [])
        if isinstance(raw_images, (str, bytes)) or not isinstance(raw_images, Sequence):
            raise TypeError("document metadata.images must be a sequence")
        if not raw_images:
            return list(chunks), 0

        stored_paths = self._get_image_storage().store_many(
            raw_images,
            collection,
            doc_hash=document.id,
        )
        if len(stored_paths) != len(raw_images):
            raise RuntimeError("image storage output count does not match input count")
        managed_paths: dict[str, str] = {}
        for index, (image, path) in enumerate(
            zip(raw_images, stored_paths, strict=True)
        ):
            if not isinstance(image, Mapping):
                raise TypeError(f"document metadata.images[{index}] must be a mapping")
            image_id = image.get("id")
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValueError(f"document metadata.images[{index}].id is invalid")
            managed_paths[image_id] = str(path)

        return [self._replace_image_paths(chunk, managed_paths) for chunk in chunks], len(
            stored_paths
        )

    @staticmethod
    def _replace_image_paths(chunk: Chunk, managed_paths: Mapping[str, str]) -> Chunk:
        metadata = deepcopy(chunk.metadata)
        images = metadata.get("images")
        if images is not None:
            updated_images: list[dict[str, Any]] = []
            for image in images:
                updated = dict(image)
                image_id = updated.get("id")
                if image_id not in managed_paths:
                    raise ValueError(f"no managed path for image ID: {image_id}")
                updated["path"] = managed_paths[image_id]
                updated_images.append(updated)
            metadata["images"] = updated_images
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )

    @staticmethod
    def _validate_vector_records(
        chunks: Sequence[Chunk],
        records: object,
    ) -> None:
        if not isinstance(records, list):
            raise TypeError("vector upserter must return list[ChunkRecord]")
        if len(records) != len(chunks):
            raise ValueError("vector upserter output count does not match chunks")
        if any(not isinstance(record, ChunkRecord) for record in records):
            raise TypeError("vector upserter output must contain only ChunkRecord")
        record_ids = [record.id for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("vector upserter returned duplicate record IDs")

    @staticmethod
    def _align_sparse_ids(
        chunks: Sequence[Chunk],
        sparse_records: Sequence[SparseChunkEncoding],
        vector_records: Sequence[ChunkRecord],
    ) -> list[SparseChunkEncoding]:
        """让两路索引共享主键，这是 Dense/Sparse 结果能做 RRF 融合的前提。"""
        if not (
            len(chunks) == len(sparse_records) == len(vector_records)
        ):
            raise ValueError("dense and sparse record counts are not aligned")
        aligned: list[SparseChunkEncoding] = []
        for index, (chunk, sparse, vector) in enumerate(
            zip(chunks, sparse_records, vector_records, strict=True)
        ):
            if sparse.chunk_id != chunk.id:
                raise ValueError(
                    f"sparse record identity is misaligned at index {index}"
                )
            aligned.append(
                SparseChunkEncoding(
                    chunk_id=vector.id,
                    term_weights=dict(sparse.term_weights),
                    doc_length=sparse.doc_length,
                )
            )
        return aligned

    def _mark_failed_without_masking(
        self,
        checker: FileIntegrityChecker,
        file_hash: str,
        source: Path,
        original_error: Exception,
    ) -> None:
        try:
            checker.mark_failed(
                file_hash,
                f"{type(original_error).__name__}: {original_error}",
                file_path=source,
                file_size=source.stat().st_size if source.is_file() else None,
            )
        except Exception as status_error:
            self.logger.error(
                "could not persist failed ingestion status source=%s error=%s",
                source,
                type(status_error).__name__,
            )

    def _get_integrity_checker(self) -> FileIntegrityChecker:
        if self._integrity_checker is None:
            self._integrity_checker = SQLiteIntegrityChecker(
                self._ingestion_path("integrity_db_path", DEFAULT_INTEGRITY_DB_PATH)
            )
        return self._integrity_checker

    def _get_loader(self) -> BaseLoader:
        if self._loader is None:
            self._loader = PdfLoader(
                image_root=self._ingestion_path("image_root", _DEFAULT_IMAGE_ROOT)
            )
        return self._loader

    def _get_chunker(self) -> DocumentChunker:
        if self._chunker is None:
            self._chunker = DocumentChunker(self.settings)
        return self._chunker

    def _get_transforms(self) -> tuple[BaseTransform, ...]:
        if self._transforms is None:
            self._transforms = (
                ChunkRefiner(self.settings),
                MetadataEnricher(self.settings),
                ImageCaptioner(self.settings),
            )
        return self._transforms

    def _get_batch_processor(self) -> BatchProcessor:
        if self._batch_processor is None:
            self._batch_processor = BatchProcessor(self.settings)
        return self._batch_processor

    def _get_vector_upserter(self) -> VectorUpserter:
        if self._vector_upserter is None:
            self._vector_upserter = VectorUpserter(self.settings)
        return self._vector_upserter

    def _get_bm25_indexer(self) -> BM25Indexer:
        if self._bm25_indexer is None:
            self._bm25_indexer = BM25Indexer(
                self._ingestion_path("bm25_path", BM25Indexer.default_path)
            )
        return self._bm25_indexer

    def _get_image_storage(self) -> ImageStorage:
        if self._image_storage is None:
            self._image_storage = ImageStorage(
                root=self._ingestion_path("image_root", _DEFAULT_IMAGE_ROOT),
                db_path=self._ingestion_path(
                    "image_index_db_path", _DEFAULT_IMAGE_INDEX_PATH
                ),
            )
        return self._image_storage

    def _ingestion_path(self, key: str, default: Path) -> Path:
        value = self.settings.ingestion.get(key, default)
        if not isinstance(value, (str, Path)) or (
            isinstance(value, str) and not value.strip()
        ):
            raise ValueError(f"ingestion.{key} must be a non-empty path")
        return Path(value)

    def _log_stage(
        self,
        stage: str,
        status: str,
        source: Path,
        collection: str,
        **details: Any,
    ) -> None:
        suffix = " ".join(f"{key}={value}" for key, value in details.items())
        self.logger.info(
            "ingestion stage=%s status=%s source=%s collection=%s%s",
            stage,
            status,
            source,
            collection,
            f" {suffix}" if suffix else "",
        )

    @staticmethod
    def _normalize_source_path(source_path: object) -> Path:
        if not isinstance(source_path, (str, Path)) or (
            isinstance(source_path, str) and not source_path.strip()
        ):
            raise ValueError("source_path must be a non-empty string or Path")
        return Path(source_path)

    @staticmethod
    def _normalize_collection(collection: object) -> str:
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError("collection must be a non-empty string")
        normalized = collection.strip()
        if not _SAFE_COLLECTION.fullmatch(normalized):
            raise ValueError(
                "collection may contain only letters, numbers, dots, underscores, and hyphens"
            )
        return normalized

    @staticmethod
    def _optional_instance(value: Any, expected: type, field: str) -> Any:
        if value is not None and not isinstance(value, expected):
            raise TypeError(f"{field} must be {expected.__name__} or None")
        return value

    @staticmethod
    def _validate_transforms(
        transforms: Sequence[BaseTransform] | None,
    ) -> tuple[BaseTransform, ...] | None:
        if transforms is None:
            return None
        if isinstance(transforms, (str, bytes)) or not isinstance(
            transforms, Sequence
        ):
            raise TypeError("transforms must be a non-string sequence or None")
        for index, transform in enumerate(transforms):
            if not isinstance(transform, BaseTransform):
                raise TypeError(f"transforms[{index}] must implement BaseTransform")
        return tuple(transforms)


__all__ = [
    "IngestionPipeline",
    "IngestionPipelineError",
    "IngestionResult",
]
