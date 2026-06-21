"""Thin HTTP clients used by the evaluation harness.

``BackendClient`` talks to the running PaiSmart Spring Boot service:
  - POST /api/v1/users/login        -> JWT
  - GET  /api/v1/search/hybrid      -> ranked SearchResult list

``LLMClient`` talks to any OpenAI-compatible chat-completions endpoint
(DeepSeek and DashScope/Doubao are all OpenAI-compatible). It is used both to
generate answers (mirroring the production prompt) and to run the LLM-as-judge.

Both are optional: the offline path in ``evaluate.py`` never touches them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class RetrievedChunk:
    file_name: str
    chunk_id: int | None
    text: str
    score: float | None

    @property
    def doc_id(self) -> str:
        """Relevance is judged at document granularity -> use the file name."""
        return self.file_name or "unknown"


class BackendClient:
    def __init__(self, base_url: str, username: str, password: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self._token: str | None = None

    def login(self) -> str:
        resp = requests.post(
            f"{self.base_url}/api/v1/users/login",
            json={"username": self.username, "password": self.password},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        token = (body.get("data") or {}).get("token")
        if not token:
            raise RuntimeError(f"login returned no token: {body}")
        self._token = token
        return token

    def _headers(self) -> dict[str, str]:
        if not self._token:
            self.login()
        return {"Authorization": f"Bearer {self._token}"}

    def search(self, query: str, top_k: int) -> tuple[list[RetrievedChunk], float]:
        """Return (ranked chunks, latency_ms) for one query."""
        start = time.perf_counter()
        resp = requests.get(
            f"{self.base_url}/api/v1/search/hybrid",
            params={"query": query, "topK": top_k},
            headers=self._headers(),
            timeout=self.timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data") or []
        chunks = [
            RetrievedChunk(
                file_name=item.get("fileName") or item.get("fileMd5") or "unknown",
                chunk_id=item.get("chunkId"),
                text=item.get("textContent") or "",
                score=item.get("score"),
            )
            for item in data
        ]
        return chunks, latency_ms


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        # base_url should be the OpenAI-compatible root, e.g. https://api.deepseek.com
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3,
             max_tokens: int = 2000, top_p: float = 0.9) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()
        return body["choices"][0]["message"]["content"]
