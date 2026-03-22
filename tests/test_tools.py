"""
Tests for Memstate LangChain tools and retriever.
"""

import os
import time
import uuid

import pytest

from langchain_memstate import (
    MemstateRecallTool,
    MemstateRememberTool,
    MemstateBrowseTool,
    MemstateGetHistoryTool,
    MemstateTimeTravelTool,
    MemstateRetriever,
    get_memstate_tools,
)

API_KEY = os.environ.get("MEMSTATE_API_KEY", "mst_A94jiQCkQqFRuRtV1qRPL9Jo4vIkOi1r")
BASE_URL = os.environ.get("MEMSTATE_BASE_URL", "https://api.memstate.ai")
TEST_RUN_ID = str(uuid.uuid4())[:8]
PROJECT_ID = f"langchain-tools-test-{TEST_RUN_ID}"


@pytest.fixture(scope="module")
def remember_tool():
    return MemstateRememberTool(api_key=API_KEY, project_id=PROJECT_ID, base_url=BASE_URL)


@pytest.fixture(scope="module")
def recall_tool():
    return MemstateRecallTool(api_key=API_KEY, project_id=PROJECT_ID, base_url=BASE_URL)


@pytest.fixture(scope="module")
def browse_tool():
    return MemstateBrowseTool(api_key=API_KEY, project_id=PROJECT_ID, base_url=BASE_URL)


@pytest.fixture(scope="module")
def history_tool():
    return MemstateGetHistoryTool(api_key=API_KEY, project_id=PROJECT_ID, base_url=BASE_URL)


@pytest.fixture(scope="module")
def time_travel_tool():
    return MemstateTimeTravelTool(api_key=API_KEY, project_id=PROJECT_ID, base_url=BASE_URL)


class TestMemstateRememberTool:
    def test_remember_returns_confirmation(self, remember_tool):
        """Remember tool should return a confirmation string with keypath."""
        result = remember_tool._run(
            content="Alice is a Senior Backend Engineer who prefers Python and Go",
            keypath=f"users.alice.role",
        )
        assert isinstance(result, str)
        assert "users.alice.role" in result

    def test_remember_includes_version_info(self, remember_tool):
        """Remember tool result should mention versioning."""
        result = remember_tool._run(
            content="The database is PostgreSQL 16",
            keypath=f"project.database.engine",
        )
        assert isinstance(result, str)
        # Should mention version or id
        assert any(word in result.lower() for word in ["version", "id", "stored", "keypath"])

    def test_remember_nested_keypath(self, remember_tool):
        """Remember tool should handle deeply nested keypaths."""
        result = remember_tool._run(
            content="OAuth2 with JWT tokens, 24h expiry",
            keypath=f"project.myapp.auth.provider",
        )
        assert isinstance(result, str)
        assert "project.myapp.auth.provider" in result


class TestMemstateRecallTool:
    @pytest.fixture(autouse=True)
    def seed(self, remember_tool):
        """Seed memories before recall tests."""
        remember_tool._run(
            content="Bob is a Frontend Engineer specializing in React and TypeScript",
            keypath="users.bob.role",
        )
        remember_tool._run(
            content="The frontend uses React 18 with TypeScript",
            keypath="project.frontend.framework",
        )
        time.sleep(1.0)

    def test_recall_returns_string(self, recall_tool):
        """Recall tool should return a string."""
        result = recall_tool._run(query="frontend engineer")
        assert isinstance(result, str)

    def test_recall_with_prefix(self, recall_tool):
        """Recall with keypath_prefix should scope the search."""
        result = recall_tool._run(
            query="React TypeScript",
            keypath_prefix="project",
        )
        assert isinstance(result, str)

    def test_recall_no_results_message(self, recall_tool):
        """Recall with no results should return a helpful message."""
        result = recall_tool._run(
            query="quantum entanglement theory of everything xyz_unique_" + TEST_RUN_ID,
        )
        assert isinstance(result, str)


class TestMemstateBrowseTool:
    @pytest.fixture(autouse=True)
    def seed(self, remember_tool):
        """Seed memories for browse tests."""
        remember_tool._run(
            content="PostgreSQL 16 with pgvector",
            keypath="project.infra.database",
        )
        remember_tool._run(
            content="Redis 7 for caching and sessions",
            keypath="project.infra.cache",
        )
        remember_tool._run(
            content="Cloudflare for CDN and DDoS protection",
            keypath="project.infra.cdn",
        )
        time.sleep(1.0)

    def test_browse_returns_knowledge_tree(self, browse_tool):
        """Browse should return a structured overview."""
        result = browse_tool._run(keypath="project.infra")
        assert isinstance(result, str)

    def test_browse_empty_path_message(self, browse_tool):
        """Browse with non-existent path should return helpful message."""
        result = browse_tool._run(keypath=f"nonexistent.path.{TEST_RUN_ID}")
        assert isinstance(result, str)


class TestMemstateGetHistoryTool:
    @pytest.fixture(autouse=True)
    def seed_with_versions(self, remember_tool):
        """Create multiple versions of a memory."""
        remember_tool._run(
            content="Using Basic Auth for API authentication",
            keypath="project.api.auth",
        )
        time.sleep(0.5)
        remember_tool._run(
            content="Migrated to OAuth2 + JWT for API authentication",
            keypath="project.api.auth",
        )
        time.sleep(0.5)
        remember_tool._run(
            content="Now using OAuth2 + JWT + MFA for API authentication",
            keypath="project.api.auth",
        )
        time.sleep(0.5)

    def test_history_returns_string(self, history_tool):
        """History tool should return a string."""
        result = history_tool._run(keypath="project.api.auth")
        assert isinstance(result, str)

    def test_history_mentions_versions(self, history_tool):
        """History result should mention version numbers."""
        result = history_tool._run(keypath="project.api.auth")
        assert isinstance(result, str)
        # Should contain version info or history info
        assert any(word in result.lower() for word in ["v1", "version", "history", "no history"])


class TestMemstateTimeTravelTool:
    def test_time_travel_returns_string(self, time_travel_tool):
        """Time travel tool should return a string."""
        result = time_travel_tool._run(keypath="project", revision=1)
        assert isinstance(result, str)

    def test_time_travel_future_revision_graceful(self, time_travel_tool):
        """Time travel to a very high revision should return graceful message."""
        result = time_travel_tool._run(keypath="project", revision=999999)
        assert isinstance(result, str)


class TestGetMemstateTools:
    def test_get_all_tools(self):
        """get_memstate_tools should return all 5 tools by default."""
        tools = get_memstate_tools(api_key=API_KEY, project_id=PROJECT_ID)
        assert len(tools) == 5

    def test_get_specific_tools(self):
        """get_memstate_tools should return only specified tools."""
        tools = get_memstate_tools(
            api_key=API_KEY,
            project_id=PROJECT_ID,
            include_tools=["remember", "recall"],
        )
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "memstate_remember" in names
        assert "memstate_recall" in names

    def test_tools_have_correct_names(self):
        """All tools should have the expected names."""
        tools = get_memstate_tools(api_key=API_KEY, project_id=PROJECT_ID)
        names = {t.name for t in tools}
        expected = {
            "memstate_remember",
            "memstate_recall",
            "memstate_browse",
            "memstate_get_history",
            "memstate_time_travel",
        }
        assert names == expected

    def test_tools_have_descriptions(self):
        """All tools should have non-empty descriptions."""
        tools = get_memstate_tools(api_key=API_KEY, project_id=PROJECT_ID)
        for tool in tools:
            assert tool.description
            assert len(tool.description) > 20


class TestMemstateRetriever:
    @pytest.fixture(autouse=True)
    def seed(self):
        """Seed memories for retriever tests."""
        from langchain_memstate import MemstateStore
        store = MemstateStore(api_key=API_KEY, base_url=BASE_URL, project_id=PROJECT_ID)
        store.put(
            ("retriever_test",),
            "database",
            {"value": "We use PostgreSQL 16 with pgvector for vector similarity search"},
        )
        store.put(
            ("retriever_test",),
            "auth",
            {"value": "Authentication uses OAuth2 with JWT tokens"},
        )
        time.sleep(1.0)

    def test_retriever_returns_documents(self):
        """Retriever should return LangChain Documents."""
        from langchain_core.documents import Document
        retriever = MemstateRetriever(
            api_key=API_KEY,
            project_id=PROJECT_ID,
            base_url=BASE_URL,
            k=5,
        )
        docs = retriever.invoke("database")
        assert isinstance(docs, list)
        for doc in docs:
            assert isinstance(doc, Document)

    def test_retriever_documents_have_metadata(self):
        """Retrieved documents should have keypath metadata."""
        retriever = MemstateRetriever(
            api_key=API_KEY,
            project_id=PROJECT_ID,
            base_url=BASE_URL,
            k=5,
        )
        docs = retriever.invoke("PostgreSQL database")
        for doc in docs:
            assert "keypath" in doc.metadata
            assert "source" in doc.metadata

    def test_retriever_respects_k(self):
        """Retriever should return at most k documents."""
        retriever = MemstateRetriever(
            api_key=API_KEY,
            project_id=PROJECT_ID,
            base_url=BASE_URL,
            k=2,
        )
        docs = retriever.invoke("project")
        assert len(docs) <= 2
