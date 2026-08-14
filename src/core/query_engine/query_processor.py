"""把用户查询拆成 Dense、Sparse 和过滤三条下游输入。

第一性原理上，查询预处理不是改写用户问题：Dense Embedding 需要完整语义，BM25
需要规范化关键词，存储后端需要独立的结构化 filters。因此本模块保留原查询，只对
Sparse 支路做确定性分词和停用词过滤，并把显式 filters 复制进稳定契约。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any, ClassVar

from core.types import ProcessedQuery


class QueryProcessor:
    """执行无外部服务依赖、可预测且可测试的查询预处理。"""

    # 必须与当前 SparseEncoder/BM25Indexer 使用同一种词项表示；否则 QueryProcessor
    # 看似分词更漂亮，索引里却没有同样的 token，最终会直接降低召回率。
    _token_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+"
    )
    default_stopwords: ClassVar[frozenset[str]] = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "at",
            "be",
            "by",
            "for",
            "how",
            "in",
            "is",
            "of",
            "on",
            "or",
            "please",
            "the",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "与",
            "了",
            "什么",
            "和",
            "在",
            "如何",
            "是",
            "的",
            "请",
            "还是",
        }
    )

    def __init__(self, *, stopwords: Iterable[str] | None = None) -> None:
        custom = self._normalize_stopwords(stopwords)
        self.stopwords = self.default_stopwords.union(custom)

    def process(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
    ) -> ProcessedQuery:
        """保留原查询，提取有序唯一关键词，并规范化显式过滤条件。"""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        tokens = [
            match.group(0).casefold()
            for match in self._token_pattern.finditer(query)
        ]
        if not tokens:
            raise ValueError("query must contain at least one searchable term")
        filtered = [token for token in tokens if token not in self.stopwords]
        # 对只有停用词的合法查询保留原 token。空关键词比低信息关键词更危险：它会让
        # Sparse 路径无条件返回空结果，也违背 ProcessedQuery 的稳定契约。
        keywords = self._unique_in_order(filtered or tokens)

        normalized_filters: Mapping[str, Any]
        if filters is None:
            normalized_filters = {}
        elif not isinstance(filters, Mapping):
            raise TypeError("filters must be a mapping or None")
        else:
            normalized_filters = filters
        return ProcessedQuery(
            query=normalized_query,
            keywords=keywords,
            filters=dict(normalized_filters),
        )

    @staticmethod
    def _normalize_stopwords(
        stopwords: Iterable[str] | None,
    ) -> frozenset[str]:
        if stopwords is None:
            return frozenset()
        if isinstance(stopwords, (str, bytes)):
            raise TypeError("stopwords must be a non-string iterable")
        normalized: set[str] = set()
        for index, word in enumerate(stopwords):
            if not isinstance(word, str):
                raise TypeError(f"stopwords[{index}] must be a string")
            if word.strip():
                normalized.add(word.strip().casefold())
        return frozenset(normalized)

    @staticmethod
    def _unique_in_order(tokens: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(tokens))


__all__ = ["QueryProcessor"]
