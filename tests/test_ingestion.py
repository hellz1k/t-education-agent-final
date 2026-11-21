"""Unit and integration tests for Stage 1 data loading, parsing, chunking, and ingestion."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from docx import Document as DocxBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t_edu_agent.config import IngestionConfig, load_config
from t_edu_agent.ingestion.cli import main as ingest_cli_main
from t_edu_agent.ingestion.pipeline import add_file_and_reingest, run_ingestion


class TestRealKnowledgeBaseIngestion(unittest.TestCase):
    """Validate ingestion over all 28 real documents in data/raw/."""

    def test_real_kb_pipeline_and_metadata(self) -> None:
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "processed"
            docs, chunks, registry, stats = run_ingestion(
                cfg.paths.raw_dir,
                out_dir,
                cfg.ingestion,
            )

            self.assertEqual(stats.total_raw_files, 28)
            self.assertEqual(stats.unique_documents, 28)
            self.assertEqual(len(docs), 28)
            self.assertEqual(stats.duplicates_skipped, 0)
            self.assertEqual(stats.version_conflicts, [])
            self.assertEqual(stats.errors, [])
            self.assertGreater(len(chunks), 200)
            self.assertEqual(len(registry), 28)
            self.assertTrue(all(entry.status == "active" for entry in registry))

            # Every chunk must fit safely inside 512-token multilingual windows and preserve provenance
            doc_ids = {d.doc_id for d in docs}
            for chunk in chunks:
                self.assertLessEqual(chunk.search_char_count, 1450, msg=chunk.chunk_id)
                self.assertIn(chunk.doc_id, doc_ids)
                self.assertIn(chunk.title, chunk.search_text)
                self.assertIn(chunk.section, chunk.search_text)

            by_file = {d.source_file: d for d in docs}

            # QA Backend: Java explicitly optional, online only
            qa = by_file["intensive_qa_backend.md"]
            self.assertIn("java", qa.metadata.optional_or_not_required_skills)
            self.assertNotIn("java", qa.metadata.required_skills)
            self.assertEqual(qa.metadata.format_modes, ["онлайн"])
            self.assertEqual(qa.metadata.enrollment_status, "closed")

            # JavaScript: status is unknown (NOT closed), cost uncertain, online + offline Moscow
            js = by_file["intensive_javascript.md"]
            self.assertEqual(js.metadata.enrollment_status, "unknown")
            self.assertIsNone(js.metadata.is_free)
            self.assertCountEqual(js.metadata.format_modes, ["онлайн", "очно"])
            self.assertEqual(js.metadata.offline_cities, ["Москва"])

            # School Data Analysis: school_course, no selection, rolling, free, no required skills
            school = by_file["school_data_analysis.md"]
            self.assertEqual(school.metadata.category, "school_course")
            self.assertIs(school.metadata.has_selection, False)
            self.assertEqual(school.metadata.enrollment_status, "rolling")
            self.assertIs(school.metadata.is_free, True)
            self.assertEqual(school.metadata.required_skills, [])
            self.assertIn("школьники", school.metadata.audience)
            self.assertIn("все желающие", school.metadata.audience)
            self.assertEqual(school.metadata.format_modes, ["онлайн"])

            # Risk Analytics: Moscow + St. Petersburg offline, 30 hours workload
            risk = by_file["intensive_risk_analytics.md"]
            self.assertIn("Москва", risk.metadata.offline_cities)
            self.assertIn("Санкт-Петербург", risk.metadata.offline_cities)
            self.assertIsNotNone(risk.metadata.workload_note)
            self.assertIn("30", risk.metadata.workload_note or "")

            # System Analysis: Yekaterinburg offline, closed
            sys_an = by_file["intensive_system_analysis.md"]
            self.assertEqual(sys_an.metadata.offline_cities, ["Екатеринбург"])
            self.assertEqual(sys_an.metadata.enrollment_status, "closed")

            # Online-only closed free intensives
            for fname in (
                "intensive_bi_analytics.md",
                "intensive_product_analytics.md",
                "intensive_data.md",
            ):
                doc = by_file[fname]
                self.assertEqual(doc.metadata.format_modes, ["онлайн"], msg=fname)
                self.assertIs(doc.metadata.is_free, True, msg=fname)
                self.assertEqual(doc.metadata.enrollment_status, "closed", msg=fname)

            # Olympiads sharing catalog root /school/ URL must both stay active with external URLs
            prod = by_file["school_prod.md"]
            self.assertEqual(prod.metadata.category, "olympiad")
            self.assertIn("https://prodcontest.com", prod.additional_urls)

            dano = by_file["school_dano.md"]
            self.assertEqual(dano.metadata.category, "olympiad")
            self.assertIn("https://dano.hse.ru", dano.additional_urls)

            # Paid school MSHP
            mshp = by_file["school_mshp.md"]
            self.assertIs(mshp.metadata.is_free, False)
            self.assertIs(mshp.metadata.has_selection, True)
            self.assertEqual(mshp.metadata.required_skills, [])
            self.assertIn("Москва", mshp.metadata.offline_cities)
            self.assertIn("Екатеринбург", mshp.metadata.offline_cities)

            # Internships (T-Start)
            for fname in ("start_analyst.md", "start_dwh.md"):
                doc = by_file[fname]
                self.assertEqual(doc.metadata.category, "internship", msg=fname)
                self.assertEqual(doc.metadata.enrollment_status, "unknown", msg=fname)
                self.assertCountEqual(doc.metadata.format_modes, ["онлайн", "очно"], msg=fname)

            dwh = by_file["start_dwh.md"]
            self.assertIn("python", dwh.metadata.optional_or_not_required_skills)
            self.assertNotIn("python", dwh.metadata.required_skills)

            # T-Academy & related programs
            for fname in ("t_academy_qa.md", "t_academy_software_development.md"):
                doc = by_file[fname]
                self.assertEqual(doc.metadata.category, "academy", msg=fname)
                self.assertEqual(doc.metadata.enrollment_status, "closed", msg=fname)
                self.assertIs(doc.metadata.is_free, True, msg=fname)
                self.assertIn("14", doc.metadata.workload_note or "", msg=fname)

            alg = by_file["t_academy_algorithms.md"]
            self.assertEqual(alg.metadata.enrollment_status, "unknown")
            self.assertIsNone(alg.metadata.is_free)

            workshop = by_file["t_academy_project_workshop.md"]
            self.assertEqual(workshop.metadata.enrollment_status, "closed")
            self.assertIs(workshop.metadata.is_free, True)

            # Free rolling school courses with no prerequisites or selection
            for fname in (
                "school_android.md",
                "school_ios.md",
                "school_informatics.md",
                "school_math.md",
            ):
                doc = by_file[fname]
                self.assertIs(doc.metadata.has_selection, False, msg=fname)
                self.assertEqual(doc.metadata.enrollment_status, "rolling", msg=fname)
                self.assertIs(doc.metadata.is_free, True, msg=fname)
                self.assertEqual(doc.metadata.required_skills, [], msg=fname)

            # Output artifacts created
            for artifact in (
                "documents.jsonl",
                "chunks.jsonl",
                "source_registry.json",
                "kb_stats.json",
                "kb_report.md",
            ):
                self.assertTrue((out_dir / artifact).exists(), msg=artifact)


class TestIngestionEdgeCases(unittest.TestCase):
    """Test deduplication, versioning, HTML/DOCX loaders, errors, and idempotency."""

    def test_exact_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            out_dir = Path(tmp_dir) / "processed"
            raw_dir.mkdir()

            content = (
                "# Тестовый курс\n\n"
                "Источник: https://education.tbank.ru/study/test/\n"
                "Проверено: 21.11.2026\n\n"
                "## Описание\n\n"
                "Онлайн-курс по основам Python и SQL.\n"
            )
            (raw_dir / "a_course.md").write_text(content, encoding="utf-8")
            (raw_dir / "b_copy.md").write_text(content, encoding="utf-8")

            docs, chunks, registry, stats = run_ingestion(raw_dir, out_dir, IngestionConfig())
            self.assertEqual(stats.unique_documents, 1)
            self.assertEqual(stats.duplicates_skipped, 1)
            self.assertEqual(len(docs), 1)
            self.assertTrue(all(c.doc_id == docs[0].doc_id for c in chunks))
            statuses = {e.source_file: e.status for e in registry}
            self.assertEqual(statuses["a_course.md"], "active")
            self.assertEqual(statuses["b_copy.md"], "duplicate_skipped")

    def test_version_conflict_resolution_keeps_latest_verified_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            out_dir = Path(tmp_dir) / "processed"
            raw_dir.mkdir()

            old_doc = (
                "# Интенсив по SQL\n\n"
                "Источник: https://education.tbank.ru/academy/intensive/sql/\n"
                "Проверено: 01.07.2026\n\n"
                "## Описание\n\nСтарая редакция программы.\n"
            )
            new_doc = (
                "# Интенсив по SQL\n\n"
                "Источник: https://education.tbank.ru/academy/intensive/sql/\n"
                "Проверено: 21.11.2026\n\n"
                "## Описание\n\nНовая актуальная редакция программы.\n"
            )
            (raw_dir / "sql_old.md").write_text(old_doc, encoding="utf-8")
            (raw_dir / "sql_new.md").write_text(new_doc, encoding="utf-8")

            docs, chunks, registry, stats = run_ingestion(raw_dir, out_dir, IngestionConfig())
            self.assertEqual(stats.unique_documents, 1)
            self.assertEqual(len(stats.version_conflicts), 1)
            self.assertEqual(docs[0].source_file, "sql_new.md")
            self.assertEqual(docs[0].verified_at, "2026-11-21")
            self.assertTrue(all(c.doc_id == docs[0].doc_id for c in chunks))

            statuses = {e.source_file: e.status for e in registry}
            self.assertEqual(statuses["sql_new.md"], "active")
            self.assertEqual(statuses["sql_old.md"], "archived_older_version")

    def test_html_cleaning_preserves_content_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            out_dir = Path(tmp_dir) / "processed"
            raw_dir.mkdir()

            html_doc = """<!DOCTYPE html>
            <html>
            <head>
              <title>Курс</title>
              <style>.menu { color: red; }</style>
              <script>console.log("ignore_script_noise");</script>
            </head>
            <body>
              <nav><a href="/">Навигация сайта меню</a></nav>
              <div class="cookie-banner">Примите cookie</div>
              <main>
                <h1>Т-Курс по алгоритмам</h1>
                <p>Источник: https://education.tbank.ru/study/algo/</p>
                <p>Проверено: 21.11.2026</p>
                <h2>Описание</h2>
                <p>Практический онлайн-курс по структурам данных.</p>
                <ul>
                  <li>Графы и динамическое программирование</li>
                  <li>Разбор решений с наставником</li>
                </ul>
                <table>
                  <tr><th>Этап</th><th>Срок</th></tr>
                  <tr><td>Отбор</td><td>До 10 октября</td></tr>
                </table>
              </main>
              <footer>Подвал сайта и юридическая информация</footer>
            </body>
            </html>
            """
            (raw_dir / "algo.html").write_text(html_doc, encoding="utf-8")

            docs, _, _, stats = run_ingestion(raw_dir, out_dir, IngestionConfig())
            self.assertEqual(stats.errors, [])
            self.assertEqual(len(docs), 1)
            full_text = "\n".join(s.text for s in docs[0].sections)
            self.assertEqual(docs[0].title, "Т-Курс по алгоритмам")
            self.assertIn("Графы и динамическое программирование", full_text)
            self.assertIn("| Этап | Срок |", full_text)
            self.assertNotIn("ignore_script_noise", full_text)
            self.assertNotIn("Навигация сайта", full_text)
            self.assertNotIn("Примите cookie", full_text)
            self.assertNotIn("Подвал сайта", full_text)

    def test_docx_loading_headings_paragraphs_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            out_dir = Path(tmp_dir) / "processed"
            raw_dir.mkdir()

            docx_path = raw_dir / "brief.docx"
            builder = DocxBuilder()
            builder.add_heading("Справка по Т-Старту", level=1)
            builder.add_paragraph("Источник: https://education.tbank.ru/start/")
            builder.add_paragraph("Проверено: 21.11.2026")
            builder.add_heading("Условия стажировки", level=2)
            builder.add_paragraph("Оплачиваемая стажировка от 3 до 6 месяцев.")
            table = builder.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "Направление"
            table.cell(0, 1).text = "Формат"
            table.cell(1, 0).text = "Backend"
            table.cell(1, 1).text = "Гибрид"
            builder.save(str(docx_path))

            docs, _, _, stats = run_ingestion(raw_dir, out_dir, IngestionConfig())
            self.assertEqual(stats.errors, [])
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].title, "Справка по Т-Старту")
            self.assertEqual(docs[0].source_url, "https://education.tbank.ru/start/")
            self.assertIn("| Направление | Формат |", docs[0].sections[0].text)
            self.assertIn("| Backend | Гибрид |", docs[0].sections[0].text)

    def test_unsupported_pdf_logged_as_error_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "raw"
            out_dir = Path(tmp_dir) / "processed"
            raw_dir.mkdir()

            (raw_dir / "valid.md").write_text(
                "# Валидный курс\n\n## Описание\n\nКороткое описание курса.\n",
                encoding="utf-8",
            )
            (raw_dir / "scan.pdf").write_bytes(b"%PDF-1.4 dummy binary")

            docs, _, _, stats = run_ingestion(raw_dir, out_dir, IngestionConfig())
            self.assertEqual(len(docs), 1)
            self.assertEqual(len(stats.errors), 1)
            self.assertEqual(stats.errors[0].source_file, "scan.pdf")
            self.assertEqual(stats.errors[0].error_type, "UnsupportedFormatError")

    def test_idempotency_and_add_file_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_dir = tmp_path / "raw"
            out_dir = tmp_path / "processed"
            extra_dir = tmp_path / "incoming"
            raw_dir.mkdir()
            extra_dir.mkdir()

            (raw_dir / "first.md").write_text(
                "# Первая программа\n\n## Описание\n\nОбучение аналитике данных.\n",
                encoding="utf-8",
            )

            docs1, chunks1, _, _ = run_ingestion(raw_dir, out_dir, IngestionConfig())
            docs2, chunks2, _, _ = run_ingestion(raw_dir, out_dir, IngestionConfig())
            self.assertEqual([d.doc_id for d in docs1], [d.doc_id for d in docs2])
            self.assertEqual([c.chunk_id for c in chunks1], [c.chunk_id for c in chunks2])

            new_file = extra_dir / "second.md"
            new_file.write_text(
                "# Вторая программа\n\n## Описание\n\nСтажировка по разработке.\n",
                encoding="utf-8",
            )
            exit_code = ingest_cli_main(
                [
                    "--raw-dir",
                    str(raw_dir),
                    "--out-dir",
                    str(out_dir),
                    "--add-file",
                    str(new_file),
                ]
            )
            self.assertEqual(exit_code, 0)

            stats_json = json.loads((out_dir / "kb_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats_json["unique_documents"], 2)

            # Verify helper directly too
            docs3, _, _, stats3 = add_file_and_reingest(
                new_file, raw_dir, out_dir, IngestionConfig()
            )
            self.assertEqual(len(docs3), 2)
            self.assertEqual(stats3.unique_documents, 2)


if __name__ == "__main__":
    unittest.main()
