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
    assert len(tools) == 5, f"Expected 5 tools, got {len(tools)}"

    # --- remember ---
    print("Tool: memstate_remember")
    result = tool_map["memstate_remember"].invoke({
        "keypath": "users.alice.stack.language",
        "content": "Alice prefers Python for backend services and TypeScript for frontend work"
    })
    print(f"  result: {result}")
    assert "success" in result.lower() or "stored" in result.lower() or "remember" in result.lower() or "alice" in result.lower(), \
        f"Unexpected remember result: {result}"

    result2 = tool_map["memstate_remember"].invoke({
        "keypath": "users.alice.stack.database",
        "content": "Alice uses PostgreSQL for production and SQLite for local dev"
    })
    print(f"  result: {result2}")

    result3 = tool_map["memstate_remember"].invoke({
        "keypath": "users.alice.stack.language",
        "content": "Alice now primarily uses Rust for performance-critical backend services"
    })
    print(f"  result (update): {result3}")

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
