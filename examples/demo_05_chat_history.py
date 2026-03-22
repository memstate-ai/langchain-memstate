"""
Demo 5: Persistent Chat History
================================
Shows MemstateChatMessageHistory storing conversation turns in Memstate.
The history persists across sessions and is organized by keypath.
"""
import os
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_memstate import MemstateChatMessageHistory

API_KEY = os.environ.get("MEMSTATE_API_KEY", "mst_A94jiQCkQqFRuRtV1qRPL9Jo4vIkOi1r")
PROJECT_ID = f"demo-chat-{uuid.uuid4().hex[:8]}"
SESSION_ID = f"session-{uuid.uuid4().hex[:8]}"

def main():
    print("=== Demo 5: Persistent Chat History ===\n")

    # Create a chat history for a specific user + session
    history = MemstateChatMessageHistory(
        api_key=API_KEY,
        project_id=PROJECT_ID,
        session_id=SESSION_ID,
    )

    print(f"Session ID: {SESSION_ID}")
    print(f"Keypath: users/{SESSION_ID}/messages\n")

    # --- Add messages ---
    print("Adding messages to history...")
    history.add_message(HumanMessage(content="Hey, what's the best way to learn Rust?"))
    history.add_message(AIMessage(content="Start with the Rust Book at doc.rust-lang.org/book -- it's free and excellent."))
    history.add_message(HumanMessage(content="Any good projects to build as a beginner?"))
    history.add_message(AIMessage(content="Try building a CLI tool with clap, or a small web server with axum. Both are great for learning ownership."))
    print("  added 4 messages")

    # --- Retrieve messages ---
    print("\nRetrieving messages...")
    messages = history.messages
    print(f"  found {len(messages)} message(s):")
    for msg in messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"    [{role}] {msg.content[:80]}")

    assert len(messages) >= 2, f"Expected at least 2 messages, got {len(messages)}"

    # --- Simulate a new session loading the same history ---
    print("\nSimulating a new session loading the same history...")
    history2 = MemstateChatMessageHistory(
        api_key=API_KEY,
        project_id=PROJECT_ID,
        session_id=SESSION_ID,
    )
    messages2 = history2.messages
    print(f"  loaded {len(messages2)} message(s) from persistent storage")
    assert len(messages2) >= 2, f"Expected at least 2 messages in new session, got {len(messages2)}"

    print("\nDemo 5 PASSED!")

if __name__ == "__main__":
    main()
