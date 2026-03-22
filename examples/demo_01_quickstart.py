"""
Demo 1: Quick Start
===================
Shows the most basic usage of MemstateStore with LangGraph.
Store a few facts, retrieve them, and search by semantic similarity.
"""
import os
import uuid
from langchain_memstate import MemstateStore

API_KEY = os.environ.get("MEMSTATE_API_KEY", "mst_A94jiQCkQqFRuRtV1qRPL9Jo4vIkOi1r")
# Use a unique project per demo run so tests don't bleed into each other
PROJECT_ID = f"demo-quickstart-{uuid.uuid4().hex[:8]}"

def main():
    print("=== Demo 1: Quick Start ===\n")

    store = MemstateStore(api_key=API_KEY, project_id=PROJECT_ID)

    # --- Store some facts ---
    print("Storing facts...")
    store.put(("users", "alice"), "preferred_language", {"value": "Python"})
    store.put(("users", "alice"), "favorite_editor", {"value": "Neovim"})
    store.put(("users", "bob"), "preferred_language", {"value": "TypeScript"})
    print("  stored: users/alice/preferred_language = Python")
    print("  stored: users/alice/favorite_editor = Neovim")
    print("  stored: users/bob/preferred_language = TypeScript")

    # --- Retrieve a specific fact ---
    print("\nRetrieving users/alice/preferred_language...")
    item = store.get(("users", "alice"), "preferred_language")
    assert item is not None, "Expected item but got None"
    print(f"  got: {item.value}")
    assert "Python" in str(item.value), f"Expected Python in value, got {item.value}"

    # --- Semantic search ---
    print("\nSearching for 'what coding language does alice use'...")
    results = store.search(("users", "alice"), query="what coding language does alice use", limit=5)
    assert len(results) > 0, "Expected search results but got none"
    print(f"  found {len(results)} result(s):")
    for r in results:
        print(f"    [{r.key}] score={r.score:.3f} value={r.value}")
    # At least one result should be about language
    found_language = any(
        "language" in r.key.lower() or
        "python" in str(r.value).lower() or
        "language" in str(r.value).lower()
        for r in results
    )
    assert found_language, f"Expected a language-related result, got: {[r.value for r in results]}"

    # --- Browse all memories under a namespace ---
    print("\nBrowsing all memories under 'users'...")
    overview = store.browse(("users",))
    assert len(overview) >= 3, f"Expected at least 3 memories, got {len(overview)}"
    print(f"  found {len(overview)} memories:")
    for keypath, summary in overview.items():
        print(f"    {keypath}: {summary[:80]}")

    print("\nDemo 1 PASSED!")

if __name__ == "__main__":
    main()
