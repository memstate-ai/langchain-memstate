"""
langchain-memstate — LangChain & LangGraph integration for Memstate AI.

Memstate AI gives your agents a structured, versioned knowledge graph
organized by keypath hierarchies. Every fact is automatically versioned,
semantically searchable, and time-travelable.

Quick start::

    pip install langchain-memstate

Usage::

    from langchain_memstate import MemstateStore, get_memstate_tools

    # LangGraph BaseStore (recommended for LangGraph agents)
    store = MemstateStore(api_key="mst_...", project_id="my-agent")

    # LangChain agent tools
    tools = get_memstate_tools(api_key="mst_...", project_id="my-agent")

    # LangChain conversation history
    from langchain_memstate import MemstateChatMessageHistory
    history = MemstateChatMessageHistory(
        api_key="mst_...", session_id="user-123", project_id="my-chatbot"
    )

    # LangChain RAG retriever
    from langchain_memstate import MemstateRetriever
    retriever = MemstateRetriever(api_key="mst_...", project_id="my-agent")
"""

from langchain_memstate.chat_history import MemstateChatMessageHistory
from langchain_memstate.retriever import MemstateRetriever
from langchain_memstate.store import MemstateStore
from langchain_memstate.tools import (
    MemstateBrowseTool,
    MemstateGetHistoryTool,
    MemstateRecallTool,
    MemstateRememberTool,
    MemstateTimeTravelTool,
    get_memstate_tools,
)

__version__ = "0.2.1"

__all__ = [
    # Core store (LangGraph BaseStore)
    "MemstateStore",
    # Chat history (LangChain BaseChatMessageHistory)
    "MemstateChatMessageHistory",
    # Retriever (LangChain BaseRetriever)
    "MemstateRetriever",
    # Agent tools
    "get_memstate_tools",
    "MemstateRememberTool",
    "MemstateRecallTool",
    "MemstateBrowseTool",
    "MemstateGetHistoryTool",
    "MemstateTimeTravelTool",
    # Version
    "__version__",
]
