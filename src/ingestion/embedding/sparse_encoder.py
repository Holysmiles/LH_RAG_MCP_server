"""生成 BM25 索引器需要的稀疏统计数据。

整体链路：``Chunk[] -> 分词 -> 词频 -> BM25 批次统计``。

本阶段不计算 IDF，也不写文件；它只把每个 Chunk 转成确定性的 ``term -> tf``，并
汇总 C11 所需的文档频率和平均长度。索引和查询共用这里的分词规则，才能保证写入时
的词与检索时的词处于同一个表示空间。
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
    """单个 Chunk 的稀疏表示；``term_weights`` 是原始词频，不是 IDF。"""

    chunk_id: str
    term_weights: dict[str, float]
    doc_length: int

    def to_dict(self) -> dict[str, Any]:
        """返回存储层可直接序列化为 JSON 的字典。"""
        return {
            "chunk_id": self.chunk_id,
            "term_weights": dict(self.term_weights),
            "doc_length": self.doc_length,
        }


@dataclass(frozen=True, slots=True)
class SparseEncodingBatch:
    """一批 Chunk 的记录及语料级统计。

    ``document_frequency`` 在每篇文档中最多计数一次；空批次的平均长度为 0。
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
    """从 Chunk 生成确定性的 BM25 词项统计。

    输入：非字符串的 Chunk 序列，以及可选的领域停用词集合。
    输出：``SparseEncodingBatch``，其中每个输入 Chunk 有一条同序记录，并包含语料级
    文档频率。失败边界：非法公开参数立即报错；空文本合法，产生长度为 0 的空词表，
    因为稀疏检索不能凭空为原文制造关键词。
    """

    _token_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+"
    )

    def __init__(self, *, stopwords: Iterable[str] | None = None) -> None:
        if stopwords is None:
            self.stopwords: frozenset[str] = frozenset()
        else:
            if isinstance(stopwords, (str, bytes)):
                raise TypeError("stopwords must be a non-string iterable")
            normalized: set[str] = set()
            for word in stopwords:
                if not isinstance(word, str):
                    raise TypeError("stopwords must contain only strings")
                if word.strip():
                    normalized.add(word.strip().lower())
            self.stopwords = frozenset(normalized)

    def encode(self, chunks: Sequence[Chunk]) -> SparseEncodingBatch:
        """对一个批次分词，并计算 TF、DF 与平均文档长度。"""
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
