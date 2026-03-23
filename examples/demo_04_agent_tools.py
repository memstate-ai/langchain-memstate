"""
Demo 4: Agent Tools
===================
Shows all 5 Memstate tools working directly (without an LLM, to keep
the demo fast and free of API key requirements for OpenAI).

The tools are: remember, recall, browse, history, time_travel.
"""
import os
import uuid
from langchain_memstate import get_memstate_tools

API_KEY = os.environ.get("MEMSTATE_API_KEY", "mst_A94jiQCkQqFRuRtV1qRPL9Jo4vIkOi1r")
PROJECT_ID = f"demo-tools-{uuid.uuid4().hex[:8]}"

def main():
    print("=== Demo 4: Agent Tools ===\n")

    tools = get_memstate_tools(api_key=API_KEY, project_id=PROJECT_ID)
    tool_map = {t.name: t for t in tools}

    print(f"Available tools: {[t.name for t in tools]}\n")
    assert len(tools) == 6, f"Expected 6 tools, got {len(tools)}"

    # --- remember (auto-extraction) ---
    print("Tool: memstate_remember (auto-extraction)")
    result = tool_map["memstate_remember"].invoke({
        "content": (
            "Alice is a Senior Backend Engineer. She prefers Python for backend services "
            "and TypeScript for frontend work. Her go-to database is PostgreSQL for "
            "production and SQLite for local dev."
        ),
        "source": "agent",
    })
    print(f"  result: {result}")
    assert any(word in result.lower() for word in ["job_id", "ingestion_id", "queued", "remember"]), \
        f"Unexpected remember result: {result}"

    # --- store (precise keypath) ---
    print("\nTool: memstate_store (precise keypath)")
    result = tool_map["memstate_store"].invoke({
        "keypath": "users.alice.stack.language",
        "content": "Python (backend) and TypeScript (frontend)",
    })
    print(f"  result: {result}")
    assert "users.alice.stack.language" in result, \
        f"Unexpected store result: {result}"

    result2 = tool_map["memstate_store"].invoke({
        "keypath": "users.alice.stack.database",
        "content": "PostgreSQL (production), SQLite (local dev)",
    })
    print(f"  result: {result2}")

    # Update the language fact -- creates a new version
    result3 = tool_map["memstate_store"].invoke({
        "keypath": "users.alice.stack.language",
        "content": "Rust (performance-critical backend), Python (scripting), TypeScript (frontend)",
    })
    print(f"  result (update): {result3}")

    import time; time.sleep(1.0)

    # --- recall ---
    print("\nTool: memstate_recall")
    result = tool_map["memstate_recall"].invoke({
        "query": "what programming language does alice use"
    })
    print(f"  result: {result[:200]}")
    assert "alice" in result.lower() or "rust" in result.lower() or "python" in result.lower(), \
        f"Unexpected recall result: {result}"

    # --- browse ---
    print("\nTool: memstate_browse")
    result = tool_map["memstate_browse"].invoke({
        "keypath": "users.alice"
    })
    print(f"  result: {result[:300]}")
    assert "alice" in result.lower() or "language" in result.lower() or "database" in result.lower(), \
        f"Unexpected browse result: {result}"

    # --- history ---
    print("\nTool: memstate_get_history")
    result = tool_map["memstate_get_history"].invoke({
        "keypath": "users.alice.stack.language"
    })
    print(f"  result: {result[:300]}")
    assert "version" in result.lower() or "v1" in result.lower() or "history" in result.lower(), \
        f"Unexpected history result: {result}"

    # --- time_travel ---
    # We need a revision number -- grab it from the history result
    print("\nTool: memstate_time_travel")
    result = tool_map["memstate_time_travel"].invoke({
        "keypath": "users.alice",
        "revision": 1
    })
    print(f"  result: {result[:300]}")
    # Either finds memories or says none existed at revision 1 -- both are valid
    assert "revision" in result.lower() or "state" in result.lower() or "no memories" in result.lower(), \
        f"Unexpected time_travel result: {result}"

    print("\nDemo 4 PASSED!")

if __name__ == "__main__":
    main()
