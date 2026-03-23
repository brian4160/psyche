"""Shared Ollama LLM client — serializes all requests through a single queue."""

from __future__ import annotations

import asyncio
import httpx


class LLMClient:
    """Async wrapper around the Ollama API with a request queue.

    Only one request runs at a time so we don't exceed GPU memory.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(1)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 300,
    ) -> str:
        """Send a prompt to the LLM. Waits for the semaphore so only one
        request is in-flight at a time."""
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "system": system_prompt,
                        "prompt": user_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                )
                resp.raise_for_status()
                return resp.json()["response"]
