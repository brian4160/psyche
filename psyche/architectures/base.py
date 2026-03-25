"""Base class for consciousness architectures.

Each architecture (GWT, HOT, Freudian) implements a different computational
theory of consciousness. They share a common interface for the test harness
but have fundamentally different internal structures.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod

from psyche.board import Board, Post
from psyche.llm import LLMClient

log = logging.getLogger(__name__)


class Architecture(ABC):
    """Base class for all consciousness architectures."""

    name: str
    description: str

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.board = Board()  # each architecture gets its own board for logging
        self._reply_callbacks: list[callable] = []
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def on_reply(self, callback: callable) -> None:
        self._reply_callbacks.append(callback)

    def _emit_reply(self, reply: str) -> None:
        log.info(f"[{self.name}] REPLY: {reply}")
        for cb in self._reply_callbacks:
            try:
                cb(reply)
            except Exception:
                pass

    def _log_thought(self, source: str, content: str) -> None:
        """Log an internal thought for the transcript."""
        log.info(f"[{self.name}/{source}] THOUGHT: {content}")
        self.board.post_sync(Post(
            author=source, content=content,
            urgency=0.5, importance=0.5,
            decay_seconds=120.0, tags=[source],
        ))

    @abstractmethod
    def inject_user_message(self, text: str) -> None:
        """Process a user message and eventually produce a reply."""
        ...

    @abstractmethod
    def start_background(self) -> None:
        """Start any background processes."""
        ...

    def stop(self) -> None:
        self._running = False
