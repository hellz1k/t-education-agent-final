"""Offline unit tests for SQLite chat persistence, F5 session restoration, and owner isolation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t_edu_agent.agent import TEducationAgent
from t_edu_agent.agent.llm_client import BaseLLMClient
from t_edu_agent.agent.state import SourceRef
from t_edu_agent.config import load_config
from t_edu_agent.retrieval.hybrid import HybridRetriever
from t_edu_agent.storage import SQLiteChatStore


class ScriptedMockLLM(BaseLLMClient):
    """Minimal mock LLM for testing SQLite-backed F5 session restoration."""

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
        self.recorded_calls.append({"messages": messages, "json_mode": json_mode})
        if json_mode:
            text = self.planner_outputs.pop(0) if self.planner_outputs else '{"standalone_query":"курсы","intent":"recommend"}'
        else:
            text = self.answer_outputs.pop(0) if self.answer_outputs else "Ответ [S1]"
        return text, "mock-groq-model"


class TestSQLiteChatStore(unittest.TestCase):
    """Verify SQLite persistence, F5 restoration of TEducationAgent state, and privacy boundaries."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()
        cls.retriever = HybridRetriever(
            cls.cfg.paths.processed_dir,
            cls.cfg.paths.index_dir,
            cls.cfg.retrieval,
        )

    def test_save_and_reload_messages_with_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "chats.db"
            store1 = SQLiteChatStore(db_path)
            owner = store1.generate_token()
            chat_id = store1.create_chat(owner)

            plan1 = json.dumps(
                {
                    "standalone_query": "программы для первокурсника с Python",
                    "intent": "recommend",
                    "updated_profile": {"status": "студент", "grade_or_course": "1 курс", "known_skills": ["python"]},
                    "search_filters": {},
                    "target_program_queries": [],
                },
                ensure_ascii=False,
            )
            mock_llm = ScriptedMockLLM([plan1], ["Подойдут Т-Старт и Алгоритмы [S1]."])
            agent = TEducationAgent(self.cfg, self.retriever, mock_llm)
            resp = agent.answer(chat_id, "Я первокурсник, знаю Python. Что подойдёт?")

            store1.save_exchange(
                owner,
                chat_id,
                "Я первокурсник, знаю Python. Что подойдёт?",
                resp.answer,
                resp.sources,
                agent.get_session(chat_id),
            )

            # Reopen fresh store instance (simulating process restart / F5)
            store2 = SQLiteChatStore(db_path)
            chats = store2.list_chats(owner)
            self.assertEqual(len(chats), 1)
            self.assertEqual(chats[0].chat_id, chat_id)
            self.assertIn("Я первокурсник", chats[0].title)

            restored = store2.load_chat(owner, chat_id)
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(len(restored.messages), 2)
            self.assertEqual(restored.messages[0].role, "user")
            self.assertEqual(restored.messages[1].role, "assistant")
            self.assertEqual(
                [s.source_url for s in restored.messages[1].sources],
                [s.source_url for s in resp.sources],
            )

    def test_f5_recovery_restores_agent_profile_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "chats.db"
            store = SQLiteChatStore(db_path)
            owner = store.generate_token()
            chat_id = store.create_chat(owner)

            # Turn 1 before F5 reload
            plan_t1 = json.dumps(
                {
                    "standalone_query": "программы для первокурсника с базовым Python",
                    "intent": "recommend",
                    "updated_profile": {"status": "студент", "grade_or_course": "1 курс", "known_skills": ["python"]},
                    "search_filters": {},
                    "target_program_queries": [],
                },
                ensure_ascii=False,
            )
            agent_before_f5 = TEducationAgent(
                self.cfg,
                self.retriever,
                ScriptedMockLLM([plan_t1], ["Подойдут Алгоритмы [S1]."]),
            )
            resp1 = agent_before_f5.answer(chat_id, "Я первокурсник, знаю Python на базовом уровне.")
            store.save_exchange(
                owner,
                chat_id,
                "Я первокурсник, знаю Python на базовом уровне.",
                resp1.answer,
                resp1.sources,
                agent_before_f5.get_session(chat_id),
            )

            # Simulated F5: brand-new agent and brand-new SQLite store wrapper with empty RAM
            fresh_store = SQLiteChatStore(db_path)
            restored = fresh_store.load_chat(owner, chat_id)
            self.assertIsNotNone(restored)
            assert restored is not None

            plan_t2 = json.dumps(
                {
                    "standalone_query": "программы для первокурсника с Python без SQL",
                    "intent": "recommend",
                    "updated_profile": {},
                    "search_filters": {},
                    "target_program_queries": [],
                },
                ensure_ascii=False,
            )
            llm_after_f5 = ScriptedMockLLM([plan_t2], ["Без SQL рекомендации не меняются [S1]."])
            agent_after_f5 = TEducationAgent(self.cfg, self.retriever, llm_after_f5)
            agent_after_f5.restore_session(restored.to_session_state())

            resp2 = agent_after_f5.answer(chat_id, "А если я пока не знаю SQL?")
            self.assertEqual(resp2.profile.status, "студент")
            self.assertEqual(resp2.profile.grade_or_course, "1 курс")
            self.assertEqual(resp2.profile.known_skills, ["python"])
            self.assertIn("sql", resp2.profile.missing_skills)

            planner_prompt_t2 = llm_after_f5.recorded_calls[0]["messages"][-1]["content"]
            self.assertIn("1 курс", planner_prompt_t2)
            self.assertIn("Подойдут Алгоритмы", planner_prompt_t2)

    def test_strict_owner_isolation_blocks_other_users(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteChatStore(Path(tmpdir) / "chats.db")
            owner_a = store.generate_token()
            owner_b = store.generate_token()
            chat_a = store.create_chat(owner_a, title="Чат пользователя A")

            self.assertEqual(store.list_chats(owner_b), [])
            self.assertIsNone(store.load_chat(owner_b, chat_a))
            with self.assertRaises(PermissionError):
                store.save_exchange(
                    owner_b,
                    chat_a,
                    "Привет",
                    "Ответ",
                    [],
                    TEducationAgent(self.cfg, self.retriever, ScriptedMockLLM([], [])).get_session(chat_a),
                )

    def test_secrets_and_raw_owner_token_never_stored_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "chats.db"
            store = SQLiteChatStore(db_path)
            owner = "secret-owner-token-123456"
            chat_id = store.create_chat(owner)

            plan1 = json.dumps(
                {
                    "standalone_query": "курсы Python",
                    "intent": "recommend",
                    "updated_profile": {},
                    "search_filters": {},
                    "target_program_queries": [],
                },
                ensure_ascii=False,
            )
            agent = TEducationAgent(self.cfg, self.retriever, ScriptedMockLLM([plan1], ["Курсы по Python [S1]."]))
            agent.answer(chat_id, "Подбери курсы по Python")
            store.save_exchange(
                owner,
                chat_id,
                "Мой ключ gsk_very_secret_abc1234567890, подскажи курсы",
                "Не используйте ключ GROQ_API_KEY=real_key_value в коде [S1].",
                [
                    SourceRef(
                        source_id="S1",
                        doc_id="doc1",
                        title="Информатика решает",
                        section="Описание",
                        source_url="https://education.tbank.ru/school/basic/programming/",
                        verified_at="2026-11-21",
                    )
                ],
                agent.get_session(chat_id),
            )

            restored = store.load_chat(owner, chat_id)
            assert restored is not None
            self.assertNotIn("gsk_very_secret_abc1234567890", restored.messages[0].content)
            self.assertIn("[REDACTED_SECRET]", restored.messages[0].content)
            self.assertNotIn("real_key_value", restored.messages[1].content)

            raw_bytes = db_path.read_bytes()
            self.assertNotIn(owner.encode("utf-8"), raw_bytes)
            self.assertNotIn(b"gsk_very_secret_abc1234567890", raw_bytes)
            self.assertNotIn(b"real_key_value", raw_bytes)

    def test_clear_chat_resets_messages_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SQLiteChatStore(Path(tmpdir) / "chats.db")
            owner = store.generate_token()
            chat_id = store.create_chat(owner)

            plan1 = json.dumps(
                {
                    "standalone_query": "курсы",
                    "intent": "recommend",
                    "updated_profile": {"status": "студент", "city": "Казань"},
                    "search_filters": {},
                    "target_program_queries": [],
                },
                ensure_ascii=False,
            )
            agent = TEducationAgent(self.cfg, self.retriever, ScriptedMockLLM([plan1], ["Ответ [S1]."]))
            resp = agent.answer(chat_id, "Я студент из Казани")
            store.save_exchange(owner, chat_id, "Я студент из Казани", resp.answer, resp.sources, agent.get_session(chat_id))

            store.clear_chat(owner, chat_id)
            restored = store.load_chat(owner, chat_id)
            assert restored is not None
            self.assertEqual(restored.messages, [])
            self.assertEqual(restored.title, "Новый чат")
            self.assertIsNone(restored.profile.status)
            self.assertIsNone(restored.profile.city)


if __name__ == "__main__":
    unittest.main()
