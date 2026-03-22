"""
MemstateRetriever — LangChain BaseRetriever backed by Memstate AI.

Enables semantic search over your agent's memory store as a retriever,
making it compatible with RAG chains, RetrievalQA, and any LangChain
pipeline that accepts a retriever.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class MemstateRetriever(BaseRetriever):
    """Retrieve memories from Memstate AI as LangChain Documents.

    Each memory is returned as a Document with the content as page_content
    and rich metadata including keypath, version, and relevance score.

    Args:
        api_key: Your Memstate API key.
        project_id: Memstate project to search.
        base_url: Memstate API base URL. Defaults to https://api.memstate.ai.
        keypath_prefix: Optional keypath prefix to scope searches.
        k: Number of results to return. Defaults to 5.
        score_threshold: Minimum relevance score (0.0–1.0). Defaults to 0.0.

    Example::

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

        answer = qa_chain.invoke({"query": "What is Alice's preferred language?"})
    """

    api_key: str
    project_id: str
    base_url: str = "https://api.memstate.ai"
    keypath_prefix: Optional[str] = None
    k: int = 5
    score_threshold: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    def _get_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "langchain-memstate/0.2.0",
            },
            timeout=30.0,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Search Memstate and return results as LangChain Documents."""
        payload: dict[str, Any] = {
            "query": query,
            "project_id": self.project_id,
            "limit": self.k,
        }
        if self.keypath_prefix:
            payload["keypath_prefix"] = self.keypath_prefix

        with self._get_client() as client:
            resp = client.post("/api/v1/memories/search", json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        docs: list[Document] = []
        for r in results:
            score = r.get("score") or 0.0
            if score < self.score_threshold:
                continue
            # Use the AI summary as the primary content for RAG
            content = r.get("summary") or r.get("content", "")
            keypath = r.get("keypath", "")
            metadata = {
                "keypath": keypath,
                "score": score,
                "memory_id": r.get("id", ""),
                "version": r.get("version", 1),
                "source": f"memstate://{self.project_id}/{keypath}",
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs
