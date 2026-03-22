"""
Memstate AI tools for LangChain agents.

Provides a set of structured tools that give agents direct access to
Memstate's keypath-based memory system, including remember, recall,
browse, and time-travel capabilities.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _make_client(api_key: str, base_url: str) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        headers={
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "langchain-memstate/0.2.0",
        },
        timeout=30.0,
    )


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class RememberInput(BaseModel):
    content: str = Field(
        description=(
            "The fact or information to remember. Be specific and precise. "
            "Example: 'Alice prefers Python over JavaScript for backend work.'"
        )
    )
    keypath: str = Field(
        description=(
            "Dot-separated path to organize this memory in a hierarchy. "
            "Use descriptive, lowercase segments. "
            "Examples: 'users.alice.preferences.language', "
            "'project.myapp.auth.provider', 'team.engineering.decisions.database'"
        )
    )


class RecallInput(BaseModel):
    query: str = Field(
        description="Natural language query to search memories semantically."
    )
    keypath_prefix: Optional[str] = Field(
        default=None,
        description=(
            "Optional keypath prefix to scope the search. "
            "Example: 'users.alice' to only search Alice's memories."
        ),
    )
    limit: int = Field(default=5, description="Maximum number of results to return.")


class BrowseInput(BaseModel):
    keypath: str = Field(
        description=(
            "Keypath prefix to browse. Returns all memories under this path. "
            "Example: 'project.myapp' returns all memories about myapp."
        )
    )


class GetHistoryInput(BaseModel):
    keypath: str = Field(
        description="Full keypath to get version history for."
    )


class TimeTravelInput(BaseModel):
    keypath: str = Field(
        description="Keypath prefix to query at a past revision."
    )
    revision: int = Field(
        description=(
            "Revision number to time-travel to. "
            "Use get_history to find valid revision numbers."
        )
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class MemstateRememberTool(BaseTool):
    """Store a structured fact in Memstate AI at a specific keypath.

    Memstate automatically versions the memory if it already exists —
    the previous value is preserved in history and can be retrieved later.
    """

    name: str = "memstate_remember"
    description: str = (
        "Store a fact or piece of information in long-term memory at a structured "
        "keypath. Use dot-notation for the keypath to organize memories hierarchically. "
        "Memstate automatically versions memories — if a fact changes, the old version "
        "is preserved. Use this when you learn something important that should persist "
        "across sessions."
    )
    args_schema: Type[BaseModel] = RememberInput

    api_key: str
    project_id: str
    base_url: str = "https://api.memstate.ai"

    def _run(self, content: str, keypath: str) -> str:
        with _make_client(self.api_key, self.base_url) as client:
            resp = client.post(
                "/api/v1/memories/remember",
                json={
                    "content": content,
                    "keypath": keypath,
                    "project_id": self.project_id,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        memory_id = data.get("id", "unknown")
        version = data.get("version", 1)
        return (
            f"Stored memory at keypath '{keypath}' "
            f"(id={memory_id}, version={version}). "
            f"Previous versions are preserved in history."
        )


class MemstateRecallTool(BaseTool):
    """Semantically search Memstate AI memories."""

    name: str = "memstate_recall"
    description: str = (
        "Search long-term memory using natural language. Returns the most relevant "
        "memories with their keypaths and relevance scores. Use this to answer "
        "questions about what was previously learned or stored. You can scope the "
        "search to a specific keypath prefix for more targeted results."
    )
    args_schema: Type[BaseModel] = RecallInput

    api_key: str
    project_id: str
    base_url: str = "https://api.memstate.ai"

    def _run(
        self,
        query: str,
        keypath_prefix: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        payload: dict[str, Any] = {
            "query": query,
            "project_id": self.project_id,
            "limit": limit,
        }
        if keypath_prefix:
            payload["keypath_prefix"] = keypath_prefix

        with _make_client(self.api_key, self.base_url) as client:
            resp = client.post("/api/v1/memories/search", json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return "No memories found matching that query."

        lines = [f"Found {len(results)} relevant memories:\n"]
        for i, r in enumerate(results, 1):
            score = r.get("score") or 0.0
            keypath = r.get("keypath", "unknown")
            summary = r.get("summary") or r.get("content", "")
            lines.append(
                f"{i}. [{keypath}] (score={score:.3f})\n   {summary}"
            )
        return "\n".join(lines)


class MemstateBrowseTool(BaseTool):
    """Browse the memory knowledge graph by keypath prefix."""

    name: str = "memstate_browse"
    description: str = (
        "Browse all memories under a keypath prefix as a structured map. "
        "Returns a hierarchical overview of what is known about a topic. "
        "Use this to get a bird's-eye view before diving into details, or to "
        "discover what keypaths exist under a given prefix. "
        "Example: browsing 'project.myapp.auth' shows all auth-related memories."
    )
    args_schema: Type[BaseModel] = BrowseInput

    api_key: str
    project_id: str
    base_url: str = "https://api.memstate.ai"

    def _run(self, keypath: str) -> str:
        with _make_client(self.api_key, self.base_url) as client:
            resp = client.post(
                "/api/v1/keypaths",
                json={
                    "keypath": keypath,
                    "project_id": self.project_id,
                    "recursive": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        memories = data.get("memories", {})
        keypaths = data.get("keypaths", [])

        if not memories and not keypaths:
            return f"No memories found under keypath '{keypath}'."

        if memories:
            lines = [f"Knowledge tree under '{keypath}':\n"]
            for kp, summary in memories.items():
                lines.append(f"  {kp}: {summary}")
            return "\n".join(lines)
        else:
            lines = [f"Keypaths under '{keypath}':\n"]
            for kp in keypaths:
                lines.append(f"  {kp}")
            return "\n".join(lines)


class MemstateGetHistoryTool(BaseTool):
    """Get the version history for a specific memory keypath."""

    name: str = "memstate_get_history"
    description: str = (
        "Get the full version history for a memory at a specific keypath. "
        "Memstate preserves every version of a memory when it is updated. "
        "Use this to see how a fact has changed over time, or to find the "
        "revision number for time-travel queries."
    )
    args_schema: Type[BaseModel] = GetHistoryInput

    api_key: str
    project_id: str
    base_url: str = "https://api.memstate.ai"

    def _run(self, keypath: str) -> str:
        with _make_client(self.api_key, self.base_url) as client:
            resp = client.post(
                "/api/v1/memories/history",
                json={"keypath": keypath, "project_id": self.project_id},
            )
            resp.raise_for_status()
            data = resp.json()

        versions = data.get("versions", [])
        if not versions:
            return f"No history found for keypath '{keypath}'."

        lines = [f"Version history for '{keypath}' ({len(versions)} versions):\n"]
        for v in versions:
            ver_num = v.get("version", "?")
            summary = v.get("summary", v.get("content", ""))[:120]
            created = v.get("created_at", "unknown")
            superseded = v.get("superseded_by")
            status = f"→ superseded by v{superseded}" if superseded else "(current)"
            lines.append(f"  v{ver_num} [{created}] {status}\n    {summary}")
        return "\n".join(lines)


class MemstateTimeTravelTool(BaseTool):
    """Time-travel to see the state of memories at a past revision."""

    name: str = "memstate_time_travel"
    description: str = (
        "Retrieve the state of all memories under a keypath at a specific past revision. "
        "This is Memstate's time-travel feature — every ingestion creates a new revision, "
        "and you can reconstruct exactly what was known at any point in history. "
        "Use memstate_get_history first to find valid revision numbers."
    )
    args_schema: Type[BaseModel] = TimeTravelInput

    api_key: str
    project_id: str
    base_url: str = "https://api.memstate.ai"

    def _run(self, keypath: str, revision: int) -> str:
        with _make_client(self.api_key, self.base_url) as client:
            resp = client.post(
                "/api/v1/keypaths",
                json={
                    "keypath": keypath,
                    "project_id": self.project_id,
                    "recursive": True,
                    "at_revision": revision,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        memories = data.get("memories", {})
        if not memories:
            return (
                f"No memories found under '{keypath}' at revision {revision}. "
                f"The keypath may not have existed yet at that revision."
            )

        lines = [f"State of '{keypath}' at revision {revision}:\n"]
        for kp, summary in memories.items():
            lines.append(f"  {kp}: {summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_memstate_tools(
    api_key: str,
    project_id: str,
    base_url: str = "https://api.memstate.ai",
    include_tools: Optional[list[str]] = None,
) -> list[BaseTool]:
    """Get a list of Memstate tools for use with LangChain agents.

    Args:
        api_key: Your Memstate API key.
        project_id: Memstate project to scope all operations to.
        base_url: Memstate API base URL. Defaults to https://api.memstate.ai.
        include_tools: Optional list of tool names to include. If None,
            all tools are returned. Options: "remember", "recall", "browse",
            "history", "time_travel".

    Returns:
        List of LangChain BaseTool instances ready to pass to an agent.

    Example::

        from langchain_memstate import get_memstate_tools
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        tools = get_memstate_tools(api_key="mst_...", project_id="my-agent")
        agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools)
    """
    all_tools: dict[str, BaseTool] = {
        "remember": MemstateRememberTool(
            api_key=api_key, project_id=project_id, base_url=base_url
        ),
        "recall": MemstateRecallTool(
            api_key=api_key, project_id=project_id, base_url=base_url
        ),
        "browse": MemstateBrowseTool(
            api_key=api_key, project_id=project_id, base_url=base_url
        ),
        "history": MemstateGetHistoryTool(
            api_key=api_key, project_id=project_id, base_url=base_url
        ),
        "time_travel": MemstateTimeTravelTool(
            api_key=api_key, project_id=project_id, base_url=base_url
        ),
    }
    if include_tools is None:
        return list(all_tools.values())
    return [all_tools[name] for name in include_tools if name in all_tools]
