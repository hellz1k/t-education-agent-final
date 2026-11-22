"""Простой Streamlit-интерфейс для ИИ-консультанта «Т-Образование»."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Добавляем папку src в путь импорта, чтобы `streamlit run app.py` работал без настройки PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t_edu_agent.agent import LLMServiceError, TEducationAgent  # noqa: E402
from t_edu_agent.agent.state import SourceRef  # noqa: E402
from t_edu_agent.storage import SQLiteChatStore  # noqa: E402

# Подгружаем переменные окружения из локального .env (ключ Groq никогда не выводится на экран)
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

st.set_page_config(
    page_title="Т-Образование — ИИ-консультант",
    page_icon="🎓",
    layout="centered",
)


@st.cache_resource
def get_agent() -> TEducationAgent:
    """Создаёт и кэширует агента (RAG-индекс и клиент Groq) один раз на запуск сервера."""
    return TEducationAgent.from_config(PROJECT_ROOT / "configs" / "settings.yaml")


@st.cache_resource
def get_chat_store() -> SQLiteChatStore:
    """Создаёт локальное SQLite-хранилище чатов для сохранения истории после F5."""
    return SQLiteChatStore(PROJECT_ROOT / "data" / "chat_history.db")


def get_owner_token() -> str:
    """Получает или создаёт личный токен браузера и хранит его в URL (?u=...) для переживания F5."""
    if "owner_token" in st.session_state:
        return str(st.session_state["owner_token"])

    url_token = (st.query_params.get("u") or "").strip()
    owner_token = url_token or SQLiteChatStore.generate_token()
    st.session_state["owner_token"] = owner_token
    if url_token != owner_token:
        st.query_params["u"] = owner_token
    return owner_token


def switch_chat(store: SQLiteChatStore, agent: TEducationAgent, owner_token: str, chat_id: str) -> bool:
    """Загружает выбранный чат из SQLite и восстанавливает память агента."""
    restored = store.load_chat(owner_token, chat_id)
    if restored is None:
        return False

    st.session_state["active_chat_id"] = restored.chat_id
    st.session_state["restored_chat_id"] = restored.chat_id
    st.session_state["messages"] = [
        {"role": msg.role, "content": msg.content, "sources": msg.sources}
        for msg in restored.messages
    ]
    st.query_params["chat"] = restored.chat_id
    # Передаём агенту профиль, обсуждавшиеся программы и последние 10 реплик из SQLite
    agent.restore_session(restored.to_session_state())
    return True


def ensure_active_chat(store: SQLiteChatStore, agent: TEducationAgent, owner_token: str) -> str:
    """Определяет текущий чат при первом входе или после нажатия F5."""
    current_id = st.session_state.get("active_chat_id")
    if current_id and st.session_state.get("restored_chat_id") == current_id:
        return str(current_id)

    url_chat_id = (st.query_params.get("chat") or "").strip()
    if url_chat_id and switch_chat(store, agent, owner_token, url_chat_id):
        return url_chat_id

    existing = store.list_chats(owner_token, limit=1)
    if existing and switch_chat(store, agent, owner_token, existing[0].chat_id):
        return existing[0].chat_id

    new_chat_id = store.create_chat(owner_token)
    switch_chat(store, agent, owner_token, new_chat_id)
    return new_chat_id


def render_sources(sources: list[SourceRef]) -> None:
    """Показывает компактный раскрывающийся список подтверждённых источников под ответом."""
    if not sources:
        return

    with st.expander(f"Источники ({len(sources)})"):
        seen: set[tuple[str | None, str]] = set()
        for src in sources:
            key = (src.source_url, src.section)
            if key in seen:
                continue
            seen.add(key)

            checked = f" (проверено: {src.verified_at})" if src.verified_at else ""
            if src.source_url:
                st.markdown(f"- **[{src.source_id}]** [{src.title} — {src.section}]({src.source_url}){checked}")
            else:
                st.markdown(f"- **[{src.source_id}]** {src.title} — {src.section}{checked}")


# Инициализация агента и БД с понятным сообщением при сбое
try:
    agent = get_agent()
    store = get_chat_store()
except Exception:
    st.error("Не удалось загрузить базу знаний или настройки агента. Проверьте файлы конфигурации и индекса.")
    st.stop()

owner_token = get_owner_token()
active_chat_id = ensure_active_chat(store, agent, owner_token)

# Боковая панель: управление диалогами и список прошлых чатов пользователя
with st.sidebar:
    st.header("Диалоги")
    if st.button("➕ Новый чат", use_container_width=True):
        new_id = store.create_chat(owner_token)
        switch_chat(store, agent, owner_token, new_id)
        st.rerun()

    if st.button("🧹 Очистить текущий чат", use_container_width=True):
        store.clear_chat(owner_token, active_chat_id)
        agent.reset_session(active_chat_id)
        switch_chat(store, agent, owner_token, active_chat_id)
        st.rerun()

    st.divider()
    st.caption("Ваши чаты")
    for chat in store.list_chats(owner_token):
        label = f"💬 {chat.title}" if chat.chat_id == active_chat_id else chat.title
        if st.button(label, key=f"chat_btn_{chat.chat_id}", use_container_width=True):
            if switch_chat(store, agent, owner_token, chat.chat_id):
                st.rerun()

# Основная область чата
st.title("🎓 Т-Образование")
st.caption("ИИ-консультант по курсам, интенсивам, школам и стажировкам Т-Банка.")

# Отрисовываем историю текущего диалога (сохраняется в SQLite после F5)
for saved in st.session_state.get("messages", []):
    with st.chat_message(saved["role"]):
        st.markdown(saved["content"])
        if saved["role"] == "assistant":
            render_sources(saved.get("sources", []))

# Обработка нового сообщения пользователя
prompt = st.chat_input("Напишите ваш запрос, например: «Я первокурсник, знаю Python, что подойдёт?»")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Ищу программы и готовлю ответ..."):
            response = agent.answer(active_chat_id, prompt)
    except ValueError:
        st.warning("Пожалуйста, введите непустой вопрос.")
    except LLMServiceError:
        st.error("Сервис языковой модели временно недоступен или сработал лимит запросов. Попробуйте ещё раз чуть позже.")
    except Exception:
        st.error("Не удалось обработать запрос. Попробуйте повторить или открыть новый чат.")
    else:
        with st.chat_message("assistant"):
            st.markdown(response.answer)
            render_sources(response.sources)

        # Сохраняем в историю и SQLite только штатные диалоги (блокировки защиты от атак в память не пишутся)
        if response.model_used != "security_guard":
            store.save_exchange(
                owner_token=owner_token,
                chat_id=active_chat_id,
                user_message=prompt,
                assistant_answer=response.answer,
                sources=response.sources,
                session_state=agent.get_session(active_chat_id),
            )
            st.session_state["messages"].append({"role": "user", "content": prompt})
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                }
            )
            # Перерисовываем интерфейс, чтобы заголовок нового чата сразу обновился в боковой панели
            st.rerun()
