"""Offline recall gate through real ingestion, persisted stores and hybrid search.

Hashed term vectors are a deterministic embedding double, so this gate catches
pipeline/ranking regressions without measuring a production model's semantic
quality. Real sample PDFs and unrelated text fixtures pass through ingestion;
both retrieval routes use the reopened Chroma and BM25 indexes.
"""

import hashlib
import json
import math
from pathlib import Path
import re

import pytest

from core import Document
from core.settings import Settings
from core.query_engine import DenseRetriever, SparseRetriever, HybridSearch, QueryProcessor, Fusion
from ingestion import IngestionPipeline
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from ingestion.storage import BM25Indexer, ImageStorage, VectorUpserter
from libs.embedding.base_embedding import BaseEmbedding
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.loader import BaseLoader, PdfLoader, SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore
from observability.evaluation.eval_runner import EvalRunner


pytestmark = pytest.mark.e2e
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLDEN = FIXTURES / "golden_test_set.json"
MIN_HIT_AT_1 = .8
MIN_HIT_AT_3 = 1.
MIN_MRR_AT_3 = .85


class TermHashEmbedding(BaseEmbedding):
    """Encode text using fixed hashing; never inspect source names or labels."""

    def embed(self, texts, trace=None):
        vectors = []
        for text in texts:
            vector = [0.] * 256
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                slot = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % len(vector)
                vector[slot] += 1.
            norm = math.sqrt(sum(value * value for value in vector))
            if not norm:
                vector[0], norm = 1., 1.
            vectors.append([value / norm for value in vector])
        return vectors


class TextFixtureLoader(BaseLoader):
    """Load temporary distractors through the pipeline's loader extension point."""

    def load(self, path):
        path = Path(path)
        return Document(hashlib.sha256(path.read_bytes()).hexdigest(), path.read_text(encoding="utf-8"),
                        {"source_path": str(path), "images": []})


@pytest.fixture
def corpus(tmp_path):
    settings = Settings(
        llm={"provider": "openai", "model": "unused"}, vision_llm={"enabled": False},
        embedding={"provider": "openai", "model": "offline-term-hash"},
        splitter={"provider": "recursive", "chunk_size": 300, "chunk_overlap": 30},
        vector_store={"provider": "chroma", "persist_directory": str(tmp_path / "chroma"),
                      "collection_name": "recall_regression"},
        retrieval={"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 3},
        evaluation={"provider": "custom", "top_k": 3},
        observability={"log_level": "WARNING", "trace_file": str(tmp_path / "traces.jsonl")},
    )
    store = ChromaStore(settings.vector_store)
    bm25 = BM25Indexer(tmp_path / "bm25")
    checker = SQLiteIntegrityChecker(tmp_path / "history.db")
    images = ImageStorage(tmp_path / "images", tmp_path / "images.db")
    embedding = TermHashEmbedding()

    def pipeline(loader):
        return IngestionPipeline(
            settings, integrity_checker=checker, loader=loader,
            batch_processor=BatchProcessor(settings, dense_encoder=DenseEncoder(settings, embedding=embedding),
                                           sparse_encoder=SparseEncoder()),
            vector_upserter=VectorUpserter(settings, store=store), bm25_indexer=bm25, image_storage=images)

    pdf_pipeline = pipeline(PdfLoader(image_root=tmp_path / "extracted"))
    cases = EvalRunner.load_test_set(GOLDEN)
    for source in sorted({source for case in cases for source in case["expected_sources"]}):
        result = pdf_pipeline.run(FIXTURES / "sample_documents" / source, collection="recall")
        assert result.status == "success" and result.chunk_count > 0
    text_pipeline = pipeline(TextFixtureLoader())
    for index, text in enumerate(("Gardening soil pruning watering roses and flowers.",
                                  "Python package virtual environment dependency installation.",
                                  "Invoice total payment currency account balance.")):
        path = tmp_path / f"distractor-{index}.txt"
        path.write_text(text, encoding="utf-8")
        text_pipeline.run(path, collection="recall")

    # Reload persisted records instead of handing retrieval ingestion's in-memory index.
    reopened = ChromaStore(settings.vector_store, create_if_missing=False)
    reopened_bm25 = BM25Indexer(tmp_path / "bm25")
    assert len(reopened.list_records()) > 3
    assert reopened_bm25.document_count == len(reopened.list_records())
    hybrid = HybridSearch(settings, QueryProcessor(),
                          DenseRetriever(settings, embedding_client=embedding, vector_store=reopened),
                          SparseRetriever(settings, bm25_indexer=reopened_bm25, vector_store=reopened), Fusion(settings))
    return settings, hybrid, reopened


def assert_quality(report, min_hit, min_mrr=0.):
    details = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    assert report.hit_rate >= min_hit, f"hit_rate below {min_hit}:\n{details}"
    assert report.mrr >= min_mrr, f"mrr below {min_mrr}:\n{details}"


@pytest.mark.parametrize("top_k,min_hit,min_mrr", [(1, MIN_HIT_AT_1, 0.), (3, MIN_HIT_AT_3, MIN_MRR_AT_3)])
def test_golden_recall_reaches_fixed_thresholds(corpus, top_k, min_hit, min_mrr):
    settings, hybrid, _ = corpus
    settings.evaluation["top_k"] = top_k
    report = EvalRunner(settings, hybrid, CustomEvaluator()).run(GOLDEN)
    assert_quality(report, min_hit, min_mrr)
    assert all(len(row["retrieved_ids"]) <= top_k for row in report.results)
    assert len(report.results) >= 6


def test_recall_gate_detects_missing_relevant_documents(corpus):
    settings, hybrid, store = corpus
    for source in {case["expected_sources"][0] for case in EvalRunner.load_test_set(GOLDEN)}:
        for record in store.list_records():
            if Path(record["metadata"]["source_path"]).name == source:
                store.delete_by_metadata({"source_path": record["metadata"]["source_path"]})
    report = EvalRunner(settings, hybrid, CustomEvaluator()).run(GOLDEN)
    assert report.hit_rate == report.mrr == 0.
    with pytest.raises(AssertionError, match="hit_rate below"):
        assert_quality(report, MIN_HIT_AT_3, MIN_MRR_AT_3)
