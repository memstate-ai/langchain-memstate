# langchain-memstate

[![PyPI version](https://badge.fury.io/py/langchain-memstate.svg)](https://pypi.org/project/langchain-memstate/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LangChain & LangGraph integration for [Memstate AI](https://memstate.ai)** — structured, versioned, semantic memory for AI agents.

Memstate gives your agents a knowledge graph organized by **keypath hierarchies** (like `users.alice.preferences.language`). Every fact is automatically **versioned**, **semantically searchable**, and **time-travelable**. Your agents can build up structured knowledge over time and query it with precision.

## Why Memstate?

| Feature | Memstate | Vector DB | SQL |
|---|---|---|---|
| Semantic search | Yes | Yes | No |
| Keypath hierarchy | Yes | No | Partial |
| Automatic versioning | Yes | No | No |
| Time-travel queries | Yes | No | No |
| Conflict resolution | Automatic | Manual | Manual |
| Agent-native API | Yes | No | No |

## Installation

```bash
pip install langchain-memstate
```

Get your API key at [memstate.ai/dashboard](https://memstate.ai/dashboard).

## Quick Start

### LangGraph Store (Recommended)

The `MemstateStore` implements the LangGraph `BaseStore` interface, making it a drop-in replacement for `InMemoryStore` in any LangGraph agent:

```python
from langchain_memstate import MemstateStore
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

store = MemstateStore(api_key="mst_...", project_id="my-agent")

# Use with any LangGraph agent
agent = create_react_agent(
    ChatOpenAI(model="gpt-4o-mini"),
    tools=[...],
    store=store,
)
```

### Agent Tools

Give your agent direct access to Memstate's keypath memory system:

```python
from langchain_memstate import get_memstate_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

tools = get_memstate_tools(api_key="mst_...", project_id="my-agent")
agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools)

# The agent can now remember, recall, browse, and time-travel through memories
result = agent.invoke({
    "messages": [{"role": "user", "content": "Remember that Alice prefers Python"}]
})
```

### Keypath-Organized Memory

Memstate's keypath system lets you organize facts like a filesystem:

```python
from langchain_memstate import MemstateStore

store = MemstateStore(api_key="mst_...", project_id="my-project")

# Store structured facts at meaningful keypaths
store.put(("users", "alice", "preferences"), "language", {"value": "Python"})
store.put(("users", "alice", "preferences"), "framework", {"value": "FastAPI"})
store.put(("users", "alice"), "role", {"value": "Senior Backend Engineer"})
store.put(("project", "myapp", "auth"), "provider", {"value": "OAuth2 + JWT"})
store.put(("project", "myapp", "database"), "engine", {"value": "PostgreSQL 16"})

# Retrieve a specific fact
item = store.get(("users", "alice", "preferences"), "language")
print(item.value)  # {"value": "Python"}

# Semantic search across all memories
results = store.search(("users",), query="what does alice prefer for backend work?")
for r in results:
    print(f"[{r.namespace}.{r.key}] score={r.score:.3f}: {r.value}")

# Browse the entire knowledge tree
overview = store.browse(("project", "myapp"))
for keypath, summary in overview.items():
    print(f"  {keypath}: {summary}")
```

### Automatic Versioning

Every update to a memory is automatically versioned — the previous value is never lost:

```python
# Initial value
store.put(("project", "myapp", "auth"), "provider", {"value": "Basic Auth"})

# Update it — Memstate preserves the old version automatically
store.put(("project", "myapp", "auth"), "provider", {"value": "OAuth2 + JWT"})

# See the full version history
history = store.get_history(("project", "myapp", "auth"), "provider")
for version in history:
    print(f"v{version['version']}: {version['summary']}")
# v1: Basic Auth was used for authentication
# v2: OAuth2 + JWT is now used for authentication (current)
```

### Time-Travel Queries

Reconstruct exactly what your agent knew at any point in history:

```python
# What did the agent know about the project at revision 3?
snapshot = store.get_at_revision(("project", "myapp"), at_revision=3)
for keypath, summary in snapshot.items():
    print(f"{keypath}: {summary}")
```

### Chat Message History

Persist conversation history across sessions:

```python
from langchain_memstate import MemstateChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

chain = RunnableWithMessageHistory(
    ChatOpenAI(model="gpt-4o-mini"),
    lambda session_id: MemstateChatMessageHistory(
        api_key="mst_...",
        session_id=session_id,
        project_id="my-chatbot",
    ),
    input_messages_key="input",
    history_messages_key="history",
)
```

### RAG Retriever

Use Memstate as a retriever in any LangChain RAG pipeline:

```python
from langchain_memstate import MemstateRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

retriever = MemstateRetriever(
    api_key="mst_...",
    project_id="my-agent",
    k=5,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=retriever,
)

answer = qa_chain.invoke({"query": "What database does myapp use?"})
```

## Available Tools

When using `get_memstate_tools()`, your agent gets access to:

| Tool | Description |
|---|---|
| `memstate_remember` | **Primary tool.** Pass any text and Memstate's custom-trained AI automatically extracts facts and organizes them into keypaths |
| `memstate_store` | Store a specific value at an exact keypath. Use when you already know the keypath (auto-versioned) |
| `memstate_recall` | Semantic search across all memories |
| `memstate_browse` | Browse the knowledge tree by keypath prefix |
| `memstate_get_history` | Get version history for a keypath |
| `memstate_time_travel` | Retrieve memory state at a past revision |

## Documentation

Full documentation and tutorials: [memstate.ai/docs/integrations/langchain](https://memstate.ai/docs/integrations/langchain)

## License

MIT License. See [LICENSE](LICENSE) for details.
