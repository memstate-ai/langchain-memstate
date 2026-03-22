"""
MemstateChatMessageHistory — LangChain BaseChatMessageHistory backed by Memstate AI.

Stores conversation messages at structured keypaths, enabling:
- Cross-session conversation persistence
- Semantic search over past conversations
- Version history of how conversations evolved
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence

import httpx
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)


class MemstateChatMessageHistory(BaseChatMessageHistory):
    """Persistent chat message history stored in Memstate AI.

    Messages are stored at a structured keypath so they can be retrieved,
    searched, and versioned across sessions.

    Args:
        api_key: Your Memstate API key.
        session_id: Unique identifier for this conversation session.
        project_id: Memstate project to store messages in.
        base_url: Memstate API base URL. Defaults to https://api.memstate.ai.
        keypath_prefix: Keypath prefix for messages. Defaults to "conversations".

    Example::

        from langchain_memstate import MemstateChatMessageHistory
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from langchain_openai import ChatOpenAI

        history = MemstateChatMessageHistory(
            api_key="mst_...",
            session_id="user-123-session-1",
            project_id="my-chatbot",
        )

        chain = RunnableWithMessageHistory(
            ChatOpenAI(model="gpt-4o-mini") | ...,
            lambda session_id: MemstateChatMessageHistory(
                api_key="mst_...",
                session_id=session_id,
                project_id="my-chatbot",
            ),
        )
    """

    def __init__(
        self,
        api_key: str,
        session_id: str,
        project_id: str,
        base_url: str = "https://api.memstate.ai",
        keypath_prefix: str = "conversations",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.session_id = session_id
        self.project_id = project_id
        self.base_url = base_url.rstrip("/")
        self.keypath_prefix = keypath_prefix
        self._keypath = f"{keypath_prefix}.{session_id}.messages"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
                "User-Agent": "langchain-memstate/0.2.0",
            },
            timeout=timeout,
        )
        self._messages: list[BaseMessage] = []
        self._loaded = False

    def _load(self) -> None:
        """Load messages from Memstate."""
        if self._loaded:
            return
        resp = self._client.get(
            f"/api/v1/memories/keypath/{self._keypath}",
            params={"project_id": self.project_id},
        )
        if resp.status_code == 404:
            self._messages = []
            self._loaded = True
            return
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("content", "[]")
        try:
            msg_dicts = json.loads(raw)
            self._messages = messages_from_dict(msg_dicts)
        except (json.JSONDecodeError, TypeError, KeyError):
            self._messages = []
        self._loaded = True

    def _save(self) -> None:
        """Persist messages to Memstate."""
        msg_dicts = messages_to_dict(self._messages)
        content = json.dumps(msg_dicts)
        self._client.post(
            "/api/v1/memories/remember",
            json={
                "content": content,
                "keypath": self._keypath,
                "project_id": self.project_id,
            },
        ).raise_for_status()

    @property
    def messages(self) -> list[BaseMessage]:
        """Return all messages in this session."""
        self._load()
        return list(self._messages)

    def add_message(self, message: BaseMessage) -> None:
        """Append a message and persist."""
        self._load()
        self._messages.append(message)
        self._save()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append multiple messages and persist once."""
        self._load()
        self._messages.extend(messages)
        self._save()

    def clear(self) -> None:
        """Clear all messages for this session."""
        self._messages = []
        self._loaded = True
        self._save()

    def get_session_summary(self) -> str:
        """Return the AI-generated summary of this conversation from Memstate."""
        resp = self._client.get(
            f"/api/v1/memories/keypath/{self._keypath}",
            params={"project_id": self.project_id},
        )
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        return resp.json().get("summary", "")
