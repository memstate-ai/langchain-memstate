"""
Demo 2: Automatic Versioning
============================
Shows how Memstate automatically versions every fact when it changes.
Every update preserves the previous version -- nothing is ever deleted.
"""
import os
import uuid
from langchain_memstate import MemstateStore

API_KEY = os.environ.get("MEMSTATE_API_KEY", "mst_A94jiQCkQqFRuRtV1qRPL9Jo4vIkOi1r")
PROJECT_ID = f"demo-versioning-{uuid.uuid4().hex[:8]}"

def main():
    print("=== Demo 2: Automatic Versioning ===\n")

    store = MemstateStore(api_key=API_KEY, project_id=PROJECT_ID)

    # --- Store initial fact ---
    print("Storing initial fact: alice uses Python...")
    store.put(("users", "alice"), "preferred_language", {"value": "Python"})

    # --- Update it (Memstate auto-versions the old value) ---
    print("Updating: alice switched to Rust...")
    store.put(("users", "alice"), "preferred_language", {"value": "Rust"})

    print("Updating again: alice switched back to Python + Rust...")
    store.put(("users", "alice"), "preferred_language", {"value": "Python and Rust"})

    # --- Current value should be the latest ---
    print("\nChecking current value...")
    item = store.get(("users", "alice"), "preferred_language")
    assert item is not None, "Expected item but got None"
    print(f"  current: {item.value}")

    # --- Version history should have all 3 versions ---
    print("\nFetching version history...")
    history = store.get_history(("users", "alice"), "preferred_language")
    print(f"  found {len(history)} version(s):")
    for v in history:
        ver = v.get("version", "?")
        summary = v.get("summary", v.get("content", ""))[:100]
        superseded = v.get("superseded_by")
        status = f"-> superseded by v{superseded}" if superseded else "(current)"
        print(f"    v{ver} {status}: {summary}")

    assert len(history) >= 2, f"Expected at least 2 versions, got {len(history)}"
    print("\nDemo 2 PASSED!")

if __name__ == "__main__":
    main()
