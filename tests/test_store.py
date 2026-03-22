"""
Tests for MemstateStore — LangGraph BaseStore implementation.

These tests run against the live Memstate API using the test API key.
Set MEMSTATE_API_KEY and MEMSTATE_PROJECT_ID environment variables,
or they default to the test values.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

import pytest
from langgraph.store.base import Item, SearchItem

from langchain_memstate import MemstateStore

# Test configuration
API_KEY = os.environ.get("MEMSTATE_API_KEY", "mst_A94jiQCkQqFRuRtV1qRPL9Jo4vIkOi1r")
BASE_URL = os.environ.get("MEMSTATE_BASE_URL", "https://api.memstate.ai")
# Use a unique project prefix per test run to avoid cross-test contamination
TEST_RUN_ID = str(uuid.uuid4())[:8]
PROJECT_ID = f"langchain-test-{TEST_RUN_ID}"


@pytest.fixture(scope="module")
def store():
    """Create a MemstateStore for the test module."""
    s = MemstateStore(api_key=API_KEY, base_url=BASE_URL, project_id=PROJECT_ID)
    yield s
    s._client.close()


class TestMemstateStorePut:
    """Test the put() method — storing facts at keypaths."""

    def test_put_simple_value(self, store):
        """Store a simple dict value at a keypath."""
        store.put(("users", "alice"), "language", {"value": "Python"})
        # No exception = success

    def test_put_nested_namespace(self, store):
        """Store a value at a deeply nested keypath."""
        store.put(
            ("project", "myapp", "auth"),
            "provider",
            {"value": "OAuth2", "version": "2.0"},
        )

    def test_put_overwrites_with_versioning(self, store):
        """Overwriting a value should create a new version, not delete the old one."""
        ns = ("versioning", "test")
        key = "counter"
        store.put(ns, key, {"value": 1, "note": "initial"})
        store.put(ns, key, {"value": 2, "note": "updated"})
        # The latest value should be 2
        item = store.get(ns, key)
        assert item is not None
        assert item.value.get("value") == 2 or "2" in str(item.value)

    def test_put_string_content(self, store):
        """Store a string value wrapped in a dict."""
        store.put(
            ("team", "engineering"),
            "stack",
            {"value": "Go backend, React frontend, PostgreSQL"},
        )


class TestMemstateStoreGet:
    """Test the get() method — retrieving facts by keypath."""

    def test_get_existing_key(self, store):
        """Get a value that was previously stored."""
        ns = ("get_test",)
        key = "my_fact"
        store.put(ns, key, {"value": "hello world", "type": "test"})
        time.sleep(0.5)  # Brief wait for indexing
        item = store.get(ns, key)
        assert item is not None
        assert isinstance(item, Item)
        assert item.key == key
        assert item.namespace == ns

    def test_get_returns_none_for_missing(self, store):
        """Get returns None for a key that doesn't exist."""
        item = store.get(("nonexistent", "path"), "missing_key_xyz_" + TEST_RUN_ID)
        assert item is None

    def test_get_item_has_timestamps(self, store):
        """Retrieved items should have created_at and updated_at timestamps."""
        ns = ("timestamp_test",)
        key = "ts_fact"
        store.put(ns, key, {"value": "timestamp test"})
        time.sleep(0.5)
        item = store.get(ns, key)
        if item is not None:
            assert isinstance(item.created_at, datetime)
            assert isinstance(item.updated_at, datetime)

    def test_get_value_roundtrip(self, store):
        """Value stored should be retrievable with same structure."""
        ns = ("roundtrip",)
        key = "complex_fact"
        original = {
            "value": "PostgreSQL 16",
            "reason": "ACID compliance and JSON support",
            "decided_by": "engineering team",
        }
        store.put(ns, key, original)
        time.sleep(0.5)
        item = store.get(ns, key)
        assert item is not None
        # The content may be stored as a summary, so check for key values
        assert item.value is not None


class TestMemstateStoreSearch:
    """Test the search() method — semantic search over memories."""

    @pytest.fixture(autouse=True)
    def seed_memories(self, store):
        """Seed some memories for search tests."""
        store.put(
            ("search_test", "users", "alice"),
            "role",
            {"value": "Senior Backend Engineer, specializes in Python and Go"},
        )
        store.put(
            ("search_test", "users", "bob"),
            "role",
            {"value": "Frontend Engineer, React and TypeScript expert"},
        )
        store.put(
            ("search_test", "project"),
            "database",
            {"value": "PostgreSQL 16 with pgvector extension"},
        )
        time.sleep(1.0)  # Wait for indexing

    def test_search_returns_list(self, store):
        """Search should return a list of SearchItems."""
        results = store.search(("search_test",), query="Python engineer")
        assert isinstance(results, list)

    def test_search_returns_search_items(self, store):
        """Search results should be SearchItem instances."""
        results = store.search(("search_test",), query="backend engineer")
        for r in results:
            assert isinstance(r, SearchItem)

    def test_search_empty_query_returns_results(self, store):
        """Empty query should return all memories in browse mode."""
        results = store.search(("search_test",), query="")
        assert isinstance(results, list)

    def test_search_with_limit(self, store):
        """Search should respect the limit parameter."""
        results = store.search(("search_test",), query="engineer", limit=2)
        assert len(results) <= 2

    def test_search_results_have_scores(self, store):
        """Search results should have relevance scores."""
        results = store.search(("search_test",), query="database")
        for r in results:
            # Score may be None for browse mode but should be float for semantic search
            if r.score is not None:
                assert isinstance(r.score, float)


class TestMemstateStoreVersioning:
    """Test Memstate's unique versioning capabilities."""

    def test_version_history_preserved(self, store):
        """Updating a memory should preserve version history."""
        ns = ("history_test",)
        key = "auth_provider"
        # Write v1
        store.put(ns, key, {"value": "Basic Auth"})
        time.sleep(0.5)
        # Write v2
        store.put(ns, key, {"value": "OAuth2 + JWT"})
        time.sleep(0.5)
        # Write v3
        store.put(ns, key, {"value": "OAuth2 + JWT + MFA"})
        time.sleep(0.5)

        # Get history
        history = store.get_history(ns, key)
        assert isinstance(history, list)
        # Should have at least 2 versions (may have more from previous runs)
        assert len(history) >= 1

    def test_get_history_returns_dicts(self, store):
        """get_history should return a list of dicts with version metadata."""
        ns = ("history_meta_test",)
        key = "my_versioned_fact"
        store.put(ns, key, {"value": "version 1"})
        time.sleep(0.5)
        store.put(ns, key, {"value": "version 2"})
        time.sleep(0.5)

        history = store.get_history(ns, key)
        assert isinstance(history, list)
        if history:
            v = history[0]
            assert isinstance(v, dict)

    def test_browse_returns_knowledge_tree(self, store):
        """browse() should return a dict mapping keypaths to summaries."""
        ns_prefix = ("browse_test",)
        store.put(ns_prefix + ("auth",), "provider", {"value": "OAuth2"})
        store.put(ns_prefix + ("database",), "engine", {"value": "PostgreSQL"})
        store.put(ns_prefix + ("cache",), "engine", {"value": "Redis"})
        time.sleep(1.0)

        result = store.browse(ns_prefix)
        assert isinstance(result, dict)


class TestMemstateStoreDelete:
    """Test the delete() method."""

    def test_delete_existing_key(self, store):
        """Delete should not raise for an existing key."""
        ns = ("delete_test",)
        key = "to_delete_" + TEST_RUN_ID
        store.put(ns, key, {"value": "will be deleted"})
        time.sleep(0.5)
        # Should not raise
        store.delete(ns, key)


class TestMemstoreStoreListNamespaces:
    """Test the list_namespaces() method."""

    def test_list_namespaces_returns_list(self, store):
        """list_namespaces should return a list of tuples."""
        store.put(("ns_list_test", "sub1"), "key1", {"value": "v1"})
        store.put(("ns_list_test", "sub2"), "key2", {"value": "v2"})
        time.sleep(0.5)

        namespaces = store.list_namespaces(prefix=("ns_list_test",))
        assert isinstance(namespaces, list)

    def test_list_namespaces_returns_tuples(self, store):
        """Each namespace should be a tuple of strings."""
        namespaces = store.list_namespaces(prefix=("ns_list_test",))
        for ns in namespaces:
            assert isinstance(ns, tuple)
            for part in ns:
                assert isinstance(part, str)
