"""Streamlit-интерфейс для AI-ассистента «Т-Образование»."""

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

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/school:"
CHAT_AVATARS = {"user": USER_AVATAR, "assistant": ASSISTANT_AVATAR}
EXAMPLE_PROMPTS = (
    "Какие образовательные программы сейчас доступны?",
    "Какие требования для поступления на программу?",
    "Какие навыки я получу после обучения?",
    "Сравни несколько образовательных программ.",
)

st.set_page_config(
    page_title="Т-Образование — AI-ассистент",
    page_icon=ASSISTANT_AVATAR,
    layout="centered",
    initial_sidebar_state="expanded",
)


def apply_custom_styles() -> None:
    """Добавляет минималистичное косметическое оформление без вмешательства в логику Streamlit."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 48rem;
            padding-top: 2.25rem;
            padding-bottom: 2.5rem;
        }
        .te-header {
            border-left: 4px solid #FFDD2D;
            padding: 0.15rem 0 0.2rem 1rem;
            margin-bottom: 1.5rem;
        }
        .te-brand-title {
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.2;
            letter-spacing: -0.02em;
            color: #1C1F26;
            margin: 0;
        }
        .te-brand-subtitle {
            font-size: 0.95rem;
            font-weight: 600;
            color: #4B5563;
            margin: 0.25rem 0 0.45rem 0;
        }
        .te-brand-desc {
            font-size: 0.92rem;
            line-height: 1.5;
            color: #5C6470;
            margin: 0;
        }
        .te-sidebar-brand {
            border-left: 3px solid #FFDD2D;
            padding-left: 0.65rem;
            margin-bottom: 1rem;
        }
        .te-sidebar-brand-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #1C1F26;
            line-height: 1.2;
        }
        .te-sidebar-brand-sub {
            font-size: 0.8rem;
            color: #6B7280;
            margin-top: 0.15rem;
        }
        div[data-testid="stChatMessage"] {
            padding: 0.95rem 1.1rem;
            border-radius: 0.75rem;
            margin-bottom: 0.6rem;
            border: 1px solid #ECEEF2;
            max-width: 100%;
            box-sizing: border-box;
        }
        div[data-testid="stChatMessageContent"],
        div[data-testid="stChatMessage"] div[data-testid="stVerticalBlock"],
        div[data-testid="stChatMessage"] div[data-testid="stElementContainer"],
        div[data-testid="stChatMessage"] div[data-testid="stMarkdown"],
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
            min-width: 0;
            max-width: 100%;
        }
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li {
            line-height: 1.58;
        }
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] table {
            display: block;
            width: max-content;
            max-width: 100%;
            overflow-x: auto;
            border-collapse: collapse;
            margin: 0.75rem 0;
            font-size: 0.9rem;
        }
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] th,
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] td {
            white-space: normal;
            overflow-wrap: break-word;
            word-break: normal;
            vertical-align: top;
            padding: 0.5rem 0.7rem;
            max-width: 16rem;
        }
        div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] th {
            font-weight: 600;
            background-color: #F7F8FA;
        }
        section[data-testid="stSidebar"] button[kind="tertiary"] {
            justify-content: flex-start;
            text-align: left;
            font-weight: 400;
        }
        section[data-testid="stSidebar"] button[kind="secondary"] {
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Отображает лаконичную брендированную шапку Т-Образования."""
    st.markdown(
        """
        <div class="te-header">
            <div class="te-brand-title">Т-Образование</div>
            <div class="te-brand-subtitle">AI-ассистент образовательных программ</div>
            <p class="te-brand-desc">
                Помогаем разобраться в образовательных программах, требованиях к поступлению,
                сроках и возможностях обучения.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    """Показывает приветственный блок и текстовые примеры запросов для пустого диалога."""
    with st.container(border=True):
        st.subheader("Здравствуйте! Я AI-ассистент Т-Образования")
        st.write(
            "Помогу разобраться в образовательных программах, требованиях к поступлению, "
            "сроках и возможностях обучения."
        )
        st.caption("Примеры запросов:")
        st.markdown("\n".join(f"- {example}" for example in EXAMPLE_PROMPTS))


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

    unique_sources: list[SourceRef] = []
    seen: set[tuple[str | None, str]] = set()
    for src in sources:
        key = (src.source_url, src.section)
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(src)

    with st.expander(f"Источники ({len(unique_sources)})", icon=":material/menu_book:"):
        for src in unique_sources:
            checked = f" · проверено {src.verified_at}" if src.verified_at else ""
            if src.source_url:
                st.markdown(f"- **[{src.source_id}]** [{src.title}]({src.source_url}) — {src.section}{checked}")
            else:
                st.markdown(f"- **[{src.source_id}]** {src.title} — {src.section}{checked}")


apply_custom_styles()

# Инициализация агента и БД с понятным сообщением при сбое
try:
    agent = get_agent()
    store = get_chat_store()
except Exception:
    st.error(
        "Не удалось загрузить базу знаний или настройки агента. Проверьте файлы конфигурации и индекса.",
        icon=":material/error_outline:",
    )
    st.stop()

owner_token = get_owner_token()
active_chat_id = ensure_active_chat(store, agent, owner_token)

# Боковая панель: управление диалогами и список прошлых чатов пользователя
with st.sidebar:
    st.markdown(
        """
        <div class="te-sidebar-brand">
            <div class="te-sidebar-brand-title">Т-Образование</div>
            <div class="te-sidebar-brand-sub">AI-ассистент программ</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Новый чат", icon=":material/add:", type="primary", use_container_width=True):
        new_id = store.create_chat(owner_token)
        switch_chat(store, agent, owner_token, new_id)
        st.rerun()

    if st.button(
        "Очистить текущий чат",
        icon=":material/delete_sweep:",
        type="secondary",
        use_container_width=True,
    ):
        store.clear_chat(owner_token, active_chat_id)
        agent.reset_session(active_chat_id)
        switch_chat(store, agent, owner_token, active_chat_id)
        st.rerun()

    st.divider()
    st.caption("История диалогов")
    for chat in store.list_chats(owner_token):
        is_active = chat.chat_id == active_chat_id
        if st.button(
            chat.title,
            key=f"chat_btn_{chat.chat_id}",
            icon=":material/chat:" if is_active else ":material/chat_bubble_outline:",
            type="secondary" if is_active else "tertiary",
            use_container_width=True,
        ):
            if switch_chat(store, agent, owner_token, chat.chat_id):
                st.rerun()

# Основная область чата
render_header()

messages = st.session_state.get("messages", [])
if not messages:
    render_welcome()

# Отрисовываем историю текущего диалога (сохраняется в SQLite после F5)
for saved in messages:
    role = saved["role"]
    with st.chat_message(role, avatar=CHAT_AVATARS.get(role, ASSISTANT_AVATAR)):
        st.markdown(saved["content"])
        if role == "assistant":
            render_sources(saved.get("sources", []))

# Обработка нового сообщения пользователя
prompt = st.chat_input("Задайте вопрос о программах, требованиях или сроках обучения...")
if prompt:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    try:
        with st.spinner("Подбираю программы и проверяю источники..."):
            response = agent.answer(active_chat_id, prompt)
    except ValueError:
        st.warning("Пожалуйста, введите непустой вопрос.", icon=":material/info:")
    except LLMServiceError:
        st.error(
            "Сервис языковой модели временно недоступен или сработал лимит запросов. Попробуйте ещё раз чуть позже.",
            icon=":material/error_outline:",
        )
    except Exception:
        st.error(
            "Не удалось обработать запрос. Попробуйте повторить или открыть новый чат.",
            icon=":material/error_outline:",
        )
    else:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
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
