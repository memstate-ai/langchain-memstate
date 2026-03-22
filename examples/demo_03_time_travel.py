"""
Demo 3: Time Travel
===================
Shows how to retrieve the state of your knowledge base at any past revision.
This is one of Memstate's most powerful and unique features.

Every time you call put(), Memstate creates a new revision. You can
reconstruct exactly what your agent knew at any point in history.
"""
import os
import uuid
from langchain_memstate import MemstateStore

API_KEY = os.environ.get("MEMSTATE_API_KEY", "mst_A94jiQCkQqFRuRtV1qRPL9Jo4vIkOi1r")
PROJECT_ID = f"demo-timetravel-{uuid.uuid4().hex[:8]}"

def main():
    print("=== Demo 3: Time Travel ===\n")

    store = MemstateStore(api_key=API_KEY, project_id=PROJECT_ID)

    # --- Build up a history of facts ---
    print("Building up facts over time...")

    store.put(("project",), "status", {"value": "planning"})
    print("  revision 1: project/status = planning")

    store.put(("project",), "team_size", {"value": "3 engineers"})
    print("  revision 2: project/team_size = 3 engineers")

    store.put(("project",), "status", {"value": "in progress"})
    print("  revision 3: project/status = in progress")

    store.put(("project",), "team_size", {"value": "5 engineers"})
    print("  revision 4: project/team_size = 5 engineers")

    store.put(("project",), "status", {"value": "shipped"})
    print("  revision 5: project/status = shipped")

    # --- Get version history to find revision numbers ---
    print("\nFetching version history for project/status...")
    history = store.get_history(("project",), "status")
    print(f"  found {len(history)} version(s):")
    for v in history:
        ver = v.get("version", "?")
        summary = v.get("summary", v.get("content", ""))[:100]
        superseded = v.get("superseded_by")
        status_label = f"-> superseded by v{superseded}" if superseded else "(current)"
        print(f"    v{ver} {status_label}: {summary}")

    assert len(history) >= 2, f"Expected at least 2 versions, got {len(history)}"

    # --- Time-travel to an early revision ---
    # Find the earliest revision number from history
    earliest_revision = min(v.get("version", 999) for v in history)
    print(f"\nTime-traveling to revision {earliest_revision} (when project was 'planning')...")
    snapshot = store.get_at_revision(("project",), at_revision=earliest_revision)
    print(f"  snapshot at revision {earliest_revision}:")
    for keypath, summary in snapshot.items():
        print(f"    {keypath}: {summary[:80]}")

    # Current state should show "shipped"
    print("\nChecking current state (should show 'shipped')...")
    current = store.get(("project",), "status")
    assert current is not None
    print(f"  current project/status: {current.value}")

    print("\nDemo 3 PASSED!")

if __name__ == "__main__":
    main()
