"""Unit and benchmark tests for Stage 2 hybrid retrieval."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evals.retrieval_benchmark import run_benchmark
from t_edu_agent.config import RetrievalConfig, load_config
from t_edu_agent.retrieval.bm25 import BM25Index
from t_edu_agent.retrieval.hybrid import HybridRetriever
from t_edu_agent.retrieval.models import SearchFilters
from t_edu_agent.retrieval.text_processing import expand_query_aliases, infer_query_hints, tokenize


class TestTextProcessingAndBM25(unittest.TestCase):
    """Verify domain-aware tokenization, stemming, alias expansion, and BM25 ranking."""

    def test_tokenize_preserves_tech_tokens_and_stems_russian(self) -> None:
        self.assertEqual(tokenize("Анализ данных и аналитика"), tokenize("анализу данных и аналитике"))
        self.assertEqual(tokenize("стажировка"), tokenize("стажировку"))

        tokens = tokenize("C++ A/B тестирование T-ID Т-Старт")
        self.assertIn("c++", tokens)
        self.assertIn("a/b", tokens)
        self.assertIn("t-id", tokens)
        self.assertIn("т-старт", tokens)

    def test_expand_query_aliases(self) -> None:
        expanded = expand_query_aliases("Пока не знаю питон и тестировщик на удаленке")
        self.assertIn("python", expanded)
        self.assertIn("qa", expanded)
        self.assertIn("онлайн", expanded)

    def test_bm25_ranks_rare_terms(self) -> None:
        corpus = [
            "Интенсив по BI-аналитике. Визуализация в Proteus и Superset.",
            "Национальная олимпиада по анализу данных DANO для школьников.",
            "Тренажёр Число Т для подготовки к ЕГЭ по физике и математике.",
            "Олимпиада PROD по промышленной разработке.",
        ]
        index = BM25Index().fit(corpus)
        self.assertEqual(index.search("работа с дашбордами в Proteus", top_n=1)[0][0], 0)
        self.assertEqual(index.search("олимпиада DANO", top_n=1)[0][0], 1)
        self.assertEqual(index.search("тренажер Число Т", top_n=1)[0][0], 2)
        self.assertEqual(index.search("олимпиада PROD", top_n=1)[0][0], 3)


class TestHybridRetrieverOffline(unittest.TestCase):
    """Test diversity caps, negative skill constraints, doc_ids comparisons, and cache invalidation."""

    def setUp(self) -> None:
        self.cfg = load_config()
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.offline_cfg = self.cfg.retrieval.model_copy(update={"embedding_backend": "tfidf"})
        self.retriever = HybridRetriever(
            self.cfg.paths.processed_dir,
            self.tmp_dir / "index",
            self.offline_cfg,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_diversity_cap_enforced(self) -> None:
        res = self.retriever.search("анализ данных онлайн", top_k=8, max_chunks_per_doc=2)
        counts: dict[str, int] = {}
        for item in res.chunks:
            counts[item.chunk.doc_id] = counts.get(item.chunk.doc_id, 0) + 1
        self.assertTrue(all(c <= 2 for c in counts.values()))
        self.assertGreaterEqual(len(counts), 4)

    def test_exclude_required_skills_prioritizes_no_python_options(self) -> None:
        res = self.retriever.search(
            "Анализ данных онлайн, но я пока не знаю Python",
            filters=SearchFilters(exclude_required_skills=["python"]),
            top_k=6,
        )
        top_files = [c.chunk.source_file for c in res.chunks[:4]]
        no_python_docs = {"school_data_analysis.md", "start_dwh.md", "start_analyst.md"}
        self.assertTrue(any(f in no_python_docs for f in top_files), msg=f"Got: {top_files}")

        first_no_python = min(i for i, f in enumerate(top_files) if f in no_python_docs)
        python_mandatory = {"intensive_bi_analytics.md", "intensive_data.md"}
        mandatory_positions = [i for i, f in enumerate(top_files) if f in python_mandatory]
        if mandatory_positions:
            self.assertLess(first_no_python, min(mandatory_positions), msg=f"Got: {top_files}")

    def test_doc_ids_comparison_retrieves_both_programs_symmetrically(self) -> None:
        by_file = {d.source_file: d.doc_id for d in self.retriever.documents}
        bi_id = by_file["intensive_bi_analytics.md"]
        prod_id = by_file["intensive_product_analytics.md"]

        res = self.retriever.search(
            "Сравни программы, требования и длительность",
            filters=SearchFilters(doc_ids=[bi_id, prod_id]),
            top_k=6,
        )
        returned_ids = {c.chunk.doc_id for c in res.chunks}
        self.assertEqual(returned_ids, {bi_id, prod_id})

    def test_embedding_cache_hit_and_invalidation(self) -> None:
        proc_dir = self.tmp_dir / "processed_copy"
        idx_dir = self.tmp_dir / "index_cache_test"
        shutil.copytree(self.cfg.paths.processed_dir, proc_dir)

        r1 = HybridRetriever(proc_dir, idx_dir, self.offline_cfg)
        self.assertFalse(r1.cache_hit)

        r2 = HybridRetriever(proc_dir, idx_dir, self.offline_cfg)
        self.assertTrue(r2.cache_hit)

        # Modify one chunk text -> fingerprint must change and trigger rebuild
        chunks_path = proc_dir / "chunks.jsonl"
        lines = chunks_path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["search_text"] += " обновлено"
        lines[0] = json.dumps(first, ensure_ascii=False)
        chunks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        r3 = HybridRetriever(proc_dir, idx_dir, self.offline_cfg)
        self.assertFalse(r3.cache_hit)

    def test_audience_detection_and_open_all_eligibility(self) -> None:
        hints, _ = infer_query_hints("Я первокурсник, знаю Python")
        self.assertEqual(hints.audience, ["студенты"])

        res = self.retriever.search("Анализ данных и программирование онлайн", filters=hints, top_k=8)
        reasons_by_file = {c.chunk.source_file: c.match_reasons for c in res.chunks}
        for fname, reasons in reasons_by_file.items():
            if fname == "school_data_analysis.md":
                self.assertIn("audience_open_all", reasons)
                self.assertNotIn("audience_mismatch", reasons)
            elif fname == "school_mshp.md":
                self.assertIn("audience_mismatch", reasons)


class TestRealIndexGoldenBenchmark(unittest.TestCase):
    """Verify configured real neural index passes >=90% Doc Hit@5 and >=0.80 MRR offline."""

    def test_golden_benchmark_meets_quality_bar(self) -> None:
        cfg = load_config()
        manifest_path = cfg.paths.index_dir / "index_manifest.json"
        if not manifest_path.exists():
            self.skipTest("Index has not been built yet; run scripts/build_index.py first.")

        retriever = HybridRetriever(cfg.paths.processed_dir, cfg.paths.index_dir, cfg.retrieval)
        self.assertEqual(retriever.embedder.backend_name, "fastembed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            report = run_benchmark(
                cfg=cfg,
                report_path=Path(tmp_dir) / "report.json",
                retriever=retriever,
            )
            hybrid_metrics = report["summary"]["hybrid"]
            self.assertGreaterEqual(hybrid_metrics["doc_hit_at_5"], 0.90, msg=hybrid_metrics)
            self.assertGreaterEqual(hybrid_metrics["mrr"], 0.80, msg=hybrid_metrics)


if __name__ == "__main__":
    unittest.main()
