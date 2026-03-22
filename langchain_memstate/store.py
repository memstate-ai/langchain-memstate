"""
MemstateStore — LangGraph BaseStore implementation backed by Memstate AI.

Memstate organizes memories as a keypath hierarchy (e.g. "project.auth.provider")
with automatic versioning, semantic search, and time-travel queries.

The LangGraph namespace tuple maps to Memstate keypaths:
  namespace=("users", "alice", "preferences") + key="theme"
  → keypath: "users.alice.preferences.theme"
  → project_id: derived from namespace[0] (or explicit project_id param)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Optional, Sequence

import httpx
from langgraph.store.base import BaseStore, Item, SearchItem

logger = logging.getLogger(__name__)

_NOT_GIVEN = object()


class MemstateStore(BaseStore):
    """LangGraph-compatible memory store backed by Memstate AI.

    Memstate gives your agents a structured, versioned knowledge graph they
    can navigate by keypath — like a filesystem for facts. Every write is
    automatically versioned, conflicts are resolved, and you can time-travel
    to any prior state.

    Args:
        api_key: Your Memstate API key (from https://memstate.ai/dashboard).
        base_url: Memstate API base URL. Defaults to https://api.memstate.ai.
        project_id: Default project to scope all operations to. If not set,
            the first element of the LangGraph namespace tuple is used.
        timeout: HTTP request timeout in seconds. Defaults to 30.

    Example::

        from langchain_memstate import MemstateStore

        store = MemstateStore(api_key="mst_...", project_id="my-agent")

        # Store a fact at a structured keypath
        store.put(("users", "alice"), "preferred_language", {"value": "Python"})

        # Retrieve it
        item = store.get(("users", "alice"), "preferred_language")
        print(item.value)  # {"value": "Python"}

        # Semantic search across all memories
        results = store.search(("users",), query="what language does alice prefer")
        for r in results:
            print(r.key, r.value, r.score)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.memstate.ai",
        project_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_project_id = project_id
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
                "User-Agent": "langchain-memstate/0.2.0",
            },
            timeout=timeout,
        )
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                    "User-Agent": "langchain-memstate/0.2.0",
                },
                timeout=self.timeout,
            )
        return self._async_client

    # -------------------------------------------------------------------------
    # Helpers: namespace/keypath conversion
    # -------------------------------------------------------------------------

    def _namespace_to_keypath(self, namespace: tuple[str, ...], key: str) -> str:
        """Convert (namespace, key) → dot-separated keypath.

        LangGraph namespace ("users", "alice") + key "theme"
        → keypath "users.alice.theme"
        """
        parts = list(namespace) + [key]
        return ".".join(p.replace(".", "_") for p in parts)

    def _namespace_to_project(self, namespace: tuple[str, ...]) -> str:
        """Derive project_id from namespace or use the default."""
        if self.default_project_id:
            return self.default_project_id
        if namespace:
            return namespace[0]
        return "default"

    def _namespace_prefix_to_keypath(self, namespace_prefix: tuple[str, ...]) -> str:
        """Convert namespace prefix tuple to dot-separated keypath prefix."""
        return ".".join(p.replace(".", "_") for p in namespace_prefix)

    def _keypath_to_namespace_key(
        self, keypath: str, project_id: str
    ) -> tuple[tuple[str, ...], str]:
        """Convert a full keypath back to (namespace, key).

        If keypath = "users.alice.theme" and project_id = "users"
        → namespace = ("users", "alice"), key = "theme"
        """
        parts = keypath.split(".")
        if len(parts) >= 2:
            namespace = tuple(parts[:-1])
            key = parts[-1]
        else:
            namespace = (project_id,)
            key = keypath
        return namespace, key

    def _make_item(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        score: Optional[float] = None,
    ) -> Item:
        now = datetime.now(timezone.utc)
        created = (
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created_at
            else now
        )
        updated = (
            datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if updated_at
            else now
        )
        if score is not None:
            return SearchItem(
                namespace=namespace,
                key=key,
                value=value,
                created_at=created,
                updated_at=updated,
                score=score,
            )
        return Item(
            namespace=namespace,
            key=key,
            value=value,
            created_at=created,
            updated_at=updated,
        )

    # -------------------------------------------------------------------------
    # Sync API
    # -------------------------------------------------------------------------

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None = _NOT_GIVEN,  # type: ignore[assignment]
    ) -> None:
        """Store a value at (namespace, key).

        Memstate automatically versions the memory if a value already exists
        at this keypath — the previous version is preserved in history.
        """
        keypath = self._namespace_to_keypath(namespace, key)
        project_id = self._namespace_to_project(namespace)
        content = json.dumps(value)
        payload: dict[str, Any] = {
            "content": content,
            "keypath": keypath,
            "project_id": project_id,
        }
        resp = self._client.post("/api/v1/memories/remember", json=payload)
        resp.raise_for_status()
        logger.debug(
            "put: stored %s/%s → keypath=%s (project=%s)",
            namespace,
            key,
            keypath,
            project_id,
        )

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Retrieve the latest value at (namespace, key)."""
        keypath = self._namespace_to_keypath(namespace, key)
        project_id = self._namespace_to_project(namespace)
        resp = self._client.get(
            f"/api/v1/memories/keypath/{keypath}",
            params={"project_id": project_id},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        # Parse the content back from JSON string
        raw_content = data.get("content", "{}")
        try:
            value = json.loads(raw_content)
            if not isinstance(value, dict):
                value = {"value": value}
        except (json.JSONDecodeError, TypeError):
            value = {"value": raw_content}
        return self._make_item(
            namespace=namespace,
            key=key,
            value=value,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Soft-delete a memory. The version history is preserved."""
        keypath = self._namespace_to_keypath(namespace, key)
        project_id = self._namespace_to_project(namespace)
        payload = {"keypath": keypath, "project_id": project_id, "recursive": False}
        resp = self._client.post("/api/v1/memories/delete", json=payload)
        resp.raise_for_status()

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """Search memories by semantic query or browse by keypath prefix.

        If query is None or empty, returns all memories under the namespace
        prefix ordered by keypath (browse mode).
        """
        project_id = self._namespace_to_project(namespace_prefix)
        keypath_prefix = self._namespace_prefix_to_keypath(namespace_prefix)

        payload: dict[str, Any] = {
            "query": query or "",
            "project_id": project_id,
            "limit": limit,
        }
        if keypath_prefix:
            payload["keypath_prefix"] = keypath_prefix

        resp = self._client.post("/api/v1/memories/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        items: list[SearchItem] = []
        for r in results[offset:]:
            kp = r.get("keypath", "")
            ns, key = self._keypath_to_namespace_key(kp, project_id)
            raw = r.get("content") or r.get("summary", "")
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    value = {"value": raw, "summary": r.get("summary", "")}
            except (json.JSONDecodeError, TypeError):
                value = {"value": raw, "summary": r.get("summary", "")}
            items.append(
                SearchItem(
                    namespace=ns,
                    key=key,
                    value=value,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    score=r.get("score"),
                )
            )
        return items

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List all namespaces (keypath prefixes) in the store."""
        project_id = self._namespace_to_project(prefix or ())
        params: dict[str, Any] = {"project_id": project_id}
        if prefix:
            params["prefix"] = self._namespace_prefix_to_keypath(prefix)
        resp = self._client.get("/api/v1/keypaths", params=params)
        resp.raise_for_status()
        data = resp.json()
        keypaths = data.get("keypaths", [])
        namespaces: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for kp in keypaths[offset : offset + limit]:
            parts = tuple(kp.split("."))
            # Return namespace (all but last part)
            ns = parts[:-1] if len(parts) > 1 else parts
            if max_depth is not None:
                ns = ns[:max_depth]
            if ns not in seen:
                seen.add(ns)
                namespaces.append(ns)
        return namespaces

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute a batch of operations."""
        return asyncio.get_event_loop().run_until_complete(self.abatch(ops))

    # -------------------------------------------------------------------------
    # Async API
    # -------------------------------------------------------------------------

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None = _NOT_GIVEN,  # type: ignore[assignment]
    ) -> None:
        """Async version of put."""
        keypath = self._namespace_to_keypath(namespace, key)
        project_id = self._namespace_to_project(namespace)
        content = json.dumps(value)
        payload: dict[str, Any] = {
            "content": content,
            "keypath": keypath,
            "project_id": project_id,
        }
        client = self._get_async_client()
        resp = await client.post("/api/v1/memories/remember", json=payload)
        resp.raise_for_status()

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Async version of get."""
        keypath = self._namespace_to_keypath(namespace, key)
        project_id = self._namespace_to_project(namespace)
        client = self._get_async_client()
        resp = await client.get(
            f"/api/v1/memories/keypath/{keypath}",
            params={"project_id": project_id},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        raw_content = data.get("content", "{}")
        try:
            value = json.loads(raw_content)
            if not isinstance(value, dict):
                value = {"value": value}
        except (json.JSONDecodeError, TypeError):
            value = {"value": raw_content}
        return self._make_item(
            namespace=namespace,
            key=key,
            value=value,
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Async version of delete."""
        keypath = self._namespace_to_keypath(namespace, key)
        project_id = self._namespace_to_project(namespace)
        payload = {"keypath": keypath, "project_id": project_id, "recursive": False}
        client = self._get_async_client()
        resp = await client.post("/api/v1/memories/delete", json=payload)
        resp.raise_for_status()

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """Async version of search."""
        project_id = self._namespace_to_project(namespace_prefix)
        keypath_prefix = self._namespace_prefix_to_keypath(namespace_prefix)
        payload: dict[str, Any] = {
            "query": query or "",
            "project_id": project_id,
            "limit": limit,
        }
        if keypath_prefix:
            payload["keypath_prefix"] = keypath_prefix
        client = self._get_async_client()
        resp = await client.post("/api/v1/memories/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        items: list[SearchItem] = []
        for r in results[offset:]:
            kp = r.get("keypath", "")
            ns, key = self._keypath_to_namespace_key(kp, project_id)
            raw = r.get("content") or r.get("summary", "")
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    value = {"value": raw, "summary": r.get("summary", "")}
            except (json.JSONDecodeError, TypeError):
                value = {"value": raw, "summary": r.get("summary", "")}
            items.append(
                SearchItem(
                    namespace=ns,
                    key=key,
                    value=value,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    score=r.get("score"),
                )
            )
        return items

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """Async version of list_namespaces."""
        project_id = self._namespace_to_project(prefix or ())
        params: dict[str, Any] = {"project_id": project_id}
        if prefix:
            params["prefix"] = self._namespace_prefix_to_keypath(prefix)
        client = self._get_async_client()
        resp = await client.get("/api/v1/keypaths", params=params)
        resp.raise_for_status()
        data = resp.json()
        keypaths = data.get("keypaths", [])
        namespaces: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for kp in keypaths[offset : offset + limit]:
            parts = tuple(kp.split("."))
            ns = parts[:-1] if len(parts) > 1 else parts
            if max_depth is not None:
                ns = ns[:max_depth]
            if ns not in seen:
                seen.add(ns)
                namespaces.append(ns)
        return namespaces

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute a batch of operations asynchronously."""
        from langgraph.store.base import GetOp, PutOp, SearchOp, ListNamespacesOp

        results: list[Any] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self.aget(op.namespace, op.key))
            elif isinstance(op, PutOp):
                await self.aput(op.namespace, op.key, op.value or {})
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(
                    await self.asearch(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                results.append(
                    await self.alist_namespaces(
                        prefix=op.match_conditions[0].path
                        if op.match_conditions
                        else None,
                        max_depth=op.max_depth,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            else:
                results.append(None)
        return results

    # -------------------------------------------------------------------------
    # Context manager support
    # -------------------------------------------------------------------------

    def __enter__(self) -> "MemstateStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()
        if self._async_client is not None:
            asyncio.get_event_loop().run_until_complete(self._async_client.aclose())

    async def __aenter__(self) -> "MemstateStore":
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._client.close()
        if self._async_client is not None:
            await self._async_client.aclose()

    # -------------------------------------------------------------------------
    # Memstate-specific extensions (beyond BaseStore)
    # -------------------------------------------------------------------------

    def get_history(
        self, namespace: tuple[str, ...], key: str
    ) -> list[dict[str, Any]]:
        """Get the full version history for a keypath.

        This is a Memstate-specific extension that exposes the versioning
        system. Every time a value is updated, the previous version is
        preserved and accessible here.

        Returns:
            List of version dicts with keys: id, summary, version,
            created_at, superseded_by.

        Example::

            history = store.get_history(("users", "alice"), "preferred_language")
            for version in history:
                print(f"v{version['version']}: {version['summary']}")
        """
        keypath = self._namespace_to_keypath(namespace, key)
        project_id = self._namespace_to_project(namespace)
        resp = self._client.post(
            "/api/v1/memories/history",
            json={"keypath": keypath, "project_id": project_id},
        )
        resp.raise_for_status()
        return resp.json().get("versions", [])

    def get_at_revision(
        self,
        namespace_prefix: tuple[str, ...],
        at_revision: int,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """Time-travel: retrieve the entire namespace state at a past revision.

        This is one of Memstate's most powerful features. Every ingestion
        creates a new revision number, and you can reconstruct exactly what
        your agent knew at any point in time.

        Args:
            namespace_prefix: The namespace to query (maps to a keypath prefix).
            at_revision: The revision number to time-travel to.
            recursive: Include all child keypaths. Defaults to True.

        Returns:
            Dict mapping keypath → summary for all memories at that revision.

        Example::

            # What did the agent know at revision 5?
            snapshot = store.get_at_revision(("project",), at_revision=5)
            for keypath, summary in snapshot.items():
                print(f"{keypath}: {summary}")
        """
        project_id = self._namespace_to_project(namespace_prefix)
        keypath = self._namespace_prefix_to_keypath(namespace_prefix)
        payload: dict[str, Any] = {
            "project_id": project_id,
            "recursive": recursive,
            "at_revision": at_revision,
        }
        if keypath:
            payload["keypath"] = keypath
        resp = self._client.post("/api/v1/keypaths", json=payload)
        resp.raise_for_status()
        return resp.json().get("memories", {})

    def browse(
        self,
        namespace_prefix: tuple[str, ...],
        include_content: bool = False,
        limit: int = 100,
    ) -> dict[str, str]:
        """Browse all memories under a namespace prefix as keypath → summary.

        Returns a compact map of the entire knowledge subtree — ideal for
        giving an agent a structured overview before diving into details.

        Args:
            namespace_prefix: Namespace to browse.
            include_content: If True, returns full content instead of summaries.
            limit: Maximum memories to return.

        Returns:
            Dict mapping keypath → summary (or content if include_content=True).

        Example::

            # Get a bird's-eye view of everything known about "auth"
            overview = store.browse(("project", "myapp", "auth"))
            for keypath, summary in overview.items():
                print(f"  {keypath}: {summary}")
        """
        project_id = self._namespace_to_project(namespace_prefix)
        keypath = self._namespace_prefix_to_keypath(namespace_prefix)
        payload: dict[str, Any] = {
            "project_id": project_id,
            "recursive": True,
        }
        if keypath:
            payload["keypath"] = keypath
        resp = self._client.post("/api/v1/keypaths", json=payload)
        resp.raise_for_status()
        return resp.json().get("memories", {})
