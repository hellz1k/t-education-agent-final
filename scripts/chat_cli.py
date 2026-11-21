"""Interactive terminal chat runner for testing TEducationAgent with Groq."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t_edu_agent.agent import TEducationAgent


def main() -> int:
    agent = TEducationAgent.from_config()
    session_id = agent.new_session()
    print("=== T-Education Agent CLI ===")
    print("Команды: /new (новая сессия), /profile (текущий профиль), /quit (выход)\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return 0

        if not user_input:
            continue
        if user_input == "/quit":
            return 0
        if user_input == "/new":
            session_id = agent.new_session()
            print(f"[Создана новая изолированная сессия: {session_id[:8]}]\n")
            continue
        if user_input == "/profile":
            state = agent.get_session(session_id)
            print(f"[Профиль]: {state.profile.model_dump_json(ensure_ascii=False)}\n")
            continue

        try:
            resp = agent.answer(session_id, user_input)
        except Exception as exc:
            print(f"[Ошибка]: {exc}\n")
            continue

        print(f"\nАгент ({resp.model_used}, {resp.latency_ms:.0f} ms):\n{resp.answer}\n")
        if resp.sources:
            print("Источники:")
            for src in resp.sources:
                print(
                    f"  [{src.source_id}] {src.title} — {src.section} | "
                    f"{src.source_url or 'нет URL'} (проверено: {src.verified_at or 'н/д'})"
                )
        print()


if __name__ == "__main__":
    raise SystemExit(main())
