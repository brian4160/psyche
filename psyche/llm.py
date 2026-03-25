"""Shared Ollama LLM client — serializes all requests through a single queue."""

from __future__ import annotations

import asyncio
import logging
import threading
import httpx

log = logging.getLogger(__name__)


class LLMClient:
    """Async wrapper around the Ollama API with a request queue.

    Only one request runs at a time so we don't exceed GPU memory.
    Uses a threading.Semaphore so it works across multiple event loops
    (needed for combined architectures that run in separate threads).
    Retries on failure with exponential backoff.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral-nemo",
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._semaphore = threading.Semaphore(1)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 150,
    ) -> str:
        """Send a prompt to the LLM. Waits for the semaphore so only one
        request is in-flight at a time. Retries on failure."""
        # acquire in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._semaphore.acquire)
        try:
            return await self._generate_inner(system_prompt, user_prompt,
                                              temperature, max_tokens)
        finally:
            self._semaphore.release()

    async def _generate_inner(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
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
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    log.warning(f"LLM request failed (attempt {attempt + 1}), "
                                f"retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    log.error(f"LLM request failed after {self.max_retries + 1} "
                              f"attempts: {e}")
        return ""
