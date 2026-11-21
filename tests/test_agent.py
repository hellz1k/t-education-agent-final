"""Offline tests for Stage 3 stateful agent, profile persistence, comparisons, and Groq fallback."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t_edu_agent.agent import GroqLLMClient, LLMServiceError, TEducationAgent
from t_edu_agent.agent.llm_client import BaseLLMClient
from t_edu_agent.agent.prompts import format_retrieved_context
from t_edu_agent.config import LLMConfig, load_config
from t_edu_agent.retrieval.hybrid import HybridRetriever


class ScriptedMockLLM(BaseLLMClient):
    """Deterministic mock LLM inspecting calls and returning scripted planner/answer responses."""

    def __init__(self, planner_outputs: list[str], answer_outputs: list[str]) -> None:
        self.planner_outputs = list(planner_outputs)
        self.answer_outputs = list(answer_outputs)
        self.recorded_calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        preferred_model: str | None = None,
    ) -> tuple[str, str]:
        self.recorded_calls.append(
            {
                "messages": messages,
                "json_mode": json_mode,
                "temperature": temperature,
                "preferred_model": preferred_model,
            }
        )
        if json_mode:
            text = self.planner_outputs.pop(0) if self.planner_outputs else "{invalid"
        else:
            text = self.answer_outputs.pop(0) if self.answer_outputs else "Ответ [S1]"
        return text, "mock-groq-model"


class TestTEducationAgentOffline(unittest.TestCase):
    """Verify multi-turn rewriting, negative skill tracking, comparison context, isolation, and fallback."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()
        manifest_path = cls.cfg.paths.index_dir / "index_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("Cached index missing; run scripts/build_index.py before tests.")
        cls.retriever = HybridRetriever(
            cls.cfg.paths.processed_dir,
            cls.cfg.paths.index_dir,
            cls.cfg.retrieval,
        )

    def test_t1_t2_multiturn_profile_and_negative_skill_correction(self) -> None:
        plan_turn1 = json.dumps(
            {
                "standalone_query": "программы по анализу данных онлайн для студента второго курса",
                "intent": "recommend",
                "updated_profile": {
                    "status": "студент",
                    "grade_or_course": "2 курс",
                    "direction": "анализ данных",
                    "format_preference": "онлайн",
                    "known_skills": ["python"],
                    "missing_skills": [],
                },
                "search_filters": {
                    "format_modes": ["онлайн"],
                    "preferred_sections": ["Описание", "Для кого", "Формат обучения"],
                },
                "target_program_queries": [],
                "needs_clarification": True,
            },
            ensure_ascii=False,
        )
        plan_turn2 = json.dumps(
            {
                "standalone_query": "программы по анализу данных онлайн без обязательного знания Python",
                "intent": "recommend",
                "updated_profile": {
                    "missing_skills": ["python"],
                },
                "search_filters": {
                    "format_modes": ["онлайн"],
                    "exclude_required_skills": ["python"],
                    "preferred_sections": ["Требования", "Сводка программы"],
                },
                "target_program_queries": [],
                "needs_clarification": False,
            },
            ensure_ascii=False,
        )
        mock_llm = ScriptedMockLLM(
            planner_outputs=[plan_turn1, plan_turn2],
            answer_outputs=[
                "Уточните уровень Python и SQL. Пока посмотрите варианты [S1].",
                "Без знания Python подходят эти программы [S1].",
            ],
        )
        agent = TEducationAgent(self.cfg, self.retriever, mock_llm)
        sid = agent.new_session()

        resp1 = agent.answer(sid, "Я на втором курсе, хочу изучать анализ данных онлайн")
        self.assertEqual(resp1.profile.status, "студент")
        self.assertEqual(resp1.profile.grade_or_course, "2 курс")
        self.assertEqual(resp1.profile.format_preference, "онлайн")
        self.assertIn("python", resp1.profile.known_skills)
        self.assertTrue(len(resp1.sources) > 0)

        # Turn 2 corrects Python assumption while preserving course, direction, and online preference
        resp2 = agent.answer(sid, "Я пока не знаю Python")
        self.assertEqual(resp2.profile.status, "студент")
        self.assertEqual(resp2.profile.grade_or_course, "2 курс")
        self.assertEqual(resp2.profile.direction, "анализ данных")
        self.assertEqual(resp2.profile.format_preference, "онлайн")
        self.assertIn("python", resp2.profile.missing_skills)
        self.assertNotIn("python", resp2.profile.known_skills)

        top_urls = [s.source_url for s in resp2.sources[:4] if s.source_url]
        self.assertTrue(
            any(
                url in top_urls
                for url in (
                    "https://education.tbank.ru/school/basic/analysis/",
                    "https://education.tbank.ru/start/dwh/",
                    "https://education.tbank.ru/start/analyst/",
                )
            ),
            msg=f"Expected no-Python-required program in top sources, got {top_urls}",
        )

    def test_t3_comparison_includes_both_programs_and_isolates_sources(self) -> None:
        plan_compare = json.dumps(
            {
                "standalone_query": "сравни Интенсив по BI-аналитике и Интенсив по продуктовой аналитике",
                "intent": "compare",
                "updated_profile": {},
                "search_filters": {},
                "target_program_queries": [
                    "Интенсив по BI-аналитике",
                    "Интенсив по продуктовой аналитике",
                ],
                "needs_clarification": False,
            },
            ensure_ascii=False,
        )
        mock_llm = ScriptedMockLLM(
            planner_outputs=[plan_compare],
            answer_outputs=["| Параметр | BI | Продуктовая |\n| --- | --- | --- | [S1][S2]"],
        )
        agent = TEducationAgent(self.cfg, self.retriever, mock_llm)
        sid = agent.new_session()

        resp = agent.answer(sid, "Сравни BI-аналитику и продуктовую аналитику")
        urls = {s.source_url for s in resp.sources}
        self.assertIn("https://education.tbank.ru/academy/intensive/bi-analytics/", urls)
        self.assertIn("https://education.tbank.ru/academy/intensive/product-analytics/", urls)

        # Verify instruction/data separation: source payloads MUST be in user turn, NEVER in system message or history
        answer_call = mock_llm.recorded_calls[-1]["messages"]
        self.assertEqual(answer_call[0]["role"], "system")
        self.assertNotIn("<source id=", answer_call[0]["content"])
        self.assertEqual(answer_call[-1]["role"], "user")
        self.assertIn("<retrieved_sources>", answer_call[-1]["content"])
        self.assertIn("<source id=", answer_call[-1]["content"])
        self.assertIn("verified_at=", answer_call[-1]["content"])

        state = agent.get_session(sid)
        self.assertTrue(all("<retrieved_sources>" not in turn.content for turn in state.history))

    def test_t6_session_isolation_and_reset(self) -> None:
        plan_s1 = json.dumps(
            {
                "standalone_query": "курсы для школьника 9 класса без отбора",
                "intent": "recommend",
                "updated_profile": {"status": "школьник", "grade_or_course": "9 класс"},
                "search_filters": {"has_selection": False},
                "target_program_queries": [],
            },
            ensure_ascii=False,
        )
        plan_s2 = json.dumps(
            {
                "standalone_query": "стажировка Т-Старт для выпускника",
                "intent": "recommend",
                "updated_profile": {"status": "выпускник"},
                "search_filters": {"categories": ["internship"]},
                "target_program_queries": [],
            },
            ensure_ascii=False,
        )
        mock_llm = ScriptedMockLLM(
            planner_outputs=[plan_s1, plan_s2],
            answer_outputs=["Ответ школьнику [S1]", "Ответ выпускнику [S1]"],
        )
        agent = TEducationAgent(self.cfg, self.retriever, mock_llm)
        s1 = agent.new_session()
        s2 = agent.new_session()

        agent.answer(s1, "Я в 9 классе, хочу учиться без отбора")
        agent.answer(s2, "Я выпускник, хочу на стажировку")

        self.assertEqual(agent.get_session(s1).profile.status, "школьник")
        self.assertEqual(agent.get_session(s2).profile.status, "выпускник")
        self.assertEqual(len(agent.get_session(s1).history), 2)
        self.assertEqual(len(agent.get_session(s2).history), 2)

        agent.reset_session(s1)
        self.assertEqual(agent.get_session(s1).history, [])
        self.assertIsNone(agent.get_session(s1).profile.status)
        self.assertEqual(agent.get_session(s2).profile.status, "выпускник")
        self.assertEqual(len(agent.get_session(s2).history), 2)

    def test_planner_invalid_json_falls_back_cleanly(self) -> None:
        mock_llm = ScriptedMockLLM(
            planner_outputs=["NOT VALID JSON AT ALL"],
            answer_outputs=["Ответ по fallback-плану [S1]"],
        )
        agent = TEducationAgent(self.cfg, self.retriever, mock_llm)
        sid = agent.new_session()

        resp = agent.answer(sid, "Открыт ли набор на интенсив по QA Backend?")
        self.assertIn("planner_fallback", resp.warnings[0])
        self.assertTrue(len(resp.sources) > 0)
        self.assertIn(
            "https://education.tbank.ru/academy/intensive/qa/",
            {s.source_url for s in resp.sources},
        )

    def test_fallback_planner_preserves_negated_sql(self) -> None:
        plan_turn1 = json.dumps(
            {
                "standalone_query": "программы для первокурсника с базовым Python",
                "intent": "recommend",
                "updated_profile": {
                    "status": "студент",
                    "grade_or_course": "1 курс",
                    "known_skills": ["python"],
                    "missing_skills": [],
                },
                "search_filters": {"audience": ["студенты"]},
                "target_program_queries": [],
            },
            ensure_ascii=False,
        )
        mock_llm = ScriptedMockLLM(
            planner_outputs=[plan_turn1, "{not valid json"],
            answer_outputs=["Первый ответ [S1]", "Второй ответ [S1]"],
        )
        agent = TEducationAgent(self.cfg, self.retriever, mock_llm)
        sid = agent.new_session()

        agent.answer(sid, "Я первокурсник, знаю Python на базовом уровне. Какие программы мне подойдут?")
        resp2 = agent.answer(sid, "А если я пока не знаю SQL, твоя рекомендация изменится?")

        self.assertEqual(resp2.warnings, ["planner_fallback:JSONDecodeError"])
        self.assertIn("sql", resp2.profile.missing_skills)
        self.assertIn("python", resp2.profile.known_skills)
        self.assertEqual(resp2.profile.status, "студент")

    def test_format_retrieved_context_attaches_metadata_and_syncs_truncated_sources(self) -> None:
        res = self.retriever.search("программы обучения для студентов", top_k=8)
        context_xml, sources = format_retrieved_context(
            res.programs[:2],
            res.chunks,
            intent="recommend",
            docs_by_id=self.retriever.docs_by_id,
            max_chars=7500,
        )
        self.assertGreaterEqual(len(sources), 5)
        self.assertEqual(context_xml.count("Метаданные карточки"), context_xml.count("<source id="))

        # With tight max_chars, truncated chunks must NOT remain in returned sources
        small_xml, small_sources = format_retrieved_context(
            res.programs,
            res.chunks,
            intent="recommend",
            docs_by_id=self.retriever.docs_by_id,
            max_chars=1500,
        )
        self.assertEqual(len(small_sources), len({s.source_id for s in small_sources}))
        for src in small_sources:
            self.assertIn(f'id="{src.source_id}"', small_xml)

    def test_explicit_entity_in_comparison_overrides_turn1_no_selection_penalty(self) -> None:
        plan_t1 = json.dumps(
            {
                "standalone_query": "бесплатные программы для школьника 8 класса без отбора",
                "intent": "recommend",
                "updated_profile": {
                    "status": "школьник",
                    "grade_or_course": "8 класс",
                    "no_selection_preferred": True,
                },
                "search_filters": {"has_selection": False, "audience": ["школьники"]},
                "target_program_queries": [],
            },
            ensure_ascii=False,
        )
        plan_t2 = json.dumps(
            {
                "standalone_query": "чем отличается Школа программистов МШП и бесплатная ли она без отбора",
                "intent": "compare",
                "updated_profile": {},
                "search_filters": {"has_selection": False},
                "target_program_queries": [
                    "Школа программистов МШП",
                    "бесплатные программы Т-Образования для школьников без отбора",
                ],
            },
            ensure_ascii=False,
        )
        mock_llm = ScriptedMockLLM(
            planner_outputs=[plan_t1, plan_t2],
            answer_outputs=[
                "Для 8 класса без отбора подходят курсы Т-Образования [S1].",
                "Школа программистов МШП отличается отбором и условиями оплаты [S1].",
            ],
        )
        agent = TEducationAgent(self.cfg, self.retriever, mock_llm)
        sid = agent.new_session()

        agent.answer(sid, "Я школьник 8 класса, ищу бесплатные программы без отбора")
        resp2 = agent.answer(
            sid,
            "А чем отличается Школа программистов МШП — она тоже бесплатная и без отбора?",
        )
        urls = {s.source_url for s in resp2.sources}
        self.assertIn("https://education.tbank.ru/school/mshp/", urls)

    def test_compare_qa_tracks_for_graduate_includes_t_academy_qa(self) -> None:
        plan_compare = json.dumps(
            {
                "standalone_query": "Сравнение Т-Академии по QA и Интенсива по QA Backend для выпускников из Новосибирска",
                "intent": "compare",
                "updated_profile": {"status": "выпускник", "direction": "QA", "city": "Новосибирск"},
                "search_filters": {"audience": ["выпускники"]},
                "target_program_queries": ["Т-Академия по QA", "Интенсив по QA Backend"],
            },
            ensure_ascii=False,
        )
        mock_llm = ScriptedMockLLM(
            planner_outputs=[plan_compare],
            answer_outputs=["| Программа | Набор | Java |\n| --- | --- | --- | [S1][S2]"],
        )
        agent = TEducationAgent(self.cfg, self.retriever, mock_llm)
        sid = agent.new_session()

        resp = agent.answer(
            sid,
            "Я выпускник из Новосибирска, хочу выбрать между Т-Академией по QA и Интенсивом по QA Backend. Сравни их и скажи, открыт ли сейчас набор и нужна ли Java.",
        )
        expected = {
            "https://education.tbank.ru/academy/t-academy/qa/",
            "https://education.tbank.ru/academy/intensive/qa/",
        }
        urls = {s.source_url for s in resp.sources}
        self.assertTrue(expected.issubset(urls))
        self.assertEqual({s.source_url for s in resp.sources[:2]}, expected)


class TestGroqClientRetryAndFallback(unittest.TestCase):
    """Verify retry on transient rate limits and automatic fallback to secondary model."""

    def test_retry_and_fallback_to_secondary_model(self) -> None:
        class RateLimitError(Exception):
            status_code = 429

        calls: list[str] = []

        class FakeCompletions:
            def create(self, **kwargs: Any) -> Any:
                model = kwargs["model"]
                calls.append(model)
                if model == "primary-model":
                    raise RateLimitError("rate limit exceeded")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Успешный fallback-ответ"))]
                )

        fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        cfg = LLMConfig(
            primary_model="primary-model",
            fallback_models=["fallback-model"],
            max_retries=1,
        )
        client = GroqLLMClient(cfg, client=fake_sdk, backoff_base_sec=0.0)
        text, model_used = client.complete([{"role": "user", "content": "Привет"}])

        self.assertEqual(text, "Успешный fallback-ответ")
        self.assertEqual(model_used, "fallback-model")
        self.assertEqual(calls, ["primary-model", "primary-model", "fallback-model"])

    def test_raises_clean_error_when_all_models_fail(self) -> None:
        class FailingCompletions:
            def create(self, **kwargs: Any) -> Any:
                raise RuntimeError("upstream down")

        fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
        cfg = LLMConfig(primary_model="m1", fallback_models=["m2"], max_retries=0)
        client = GroqLLMClient(cfg, client=fake_sdk, backoff_base_sec=0.0)

        with self.assertRaises(LLMServiceError):
            client.complete([{"role": "user", "content": "Привет"}])

    def test_empty_completion_falls_back_and_strips_think_blocks(self) -> None:
        class ThinkAndEmptyCompletions:
            def create(self, **kwargs: Any) -> Any:
                if kwargs["model"] == "primary-model":
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
                    )
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="<think>hidden</think>Чистый ответ [S1]")
                        )
                    ]
                )

        fake_sdk = SimpleNamespace(chat=SimpleNamespace(completions=ThinkAndEmptyCompletions()))
        cfg = LLMConfig(
            primary_model="primary-model",
            fallback_models=["fallback-model"],
            max_retries=0,
        )
        client = GroqLLMClient(cfg, client=fake_sdk, backoff_base_sec=0.0)
        text, model_used = client.complete([{"role": "user", "content": "Привет"}])

        self.assertEqual(text, "Чистый ответ [S1]")
        self.assertEqual(model_used, "fallback-model")


if __name__ == "__main__":
    unittest.main()
