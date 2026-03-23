"""Base class for all cognitive modules."""

from __future__ import annotations

import asyncio
import random
import logging
from abc import ABC, abstractmethod

from psyche.board import Board, Post
from psyche.llm import LLMClient

log = logging.getLogger(__name__)


class Agent(ABC):
    """Base cognitive module.

    Subclasses must define:
        name: str           — identifier shown on the board
        system_prompt: str  — the LLM system prompt for this module
        decay_seconds: float — how long this agent's posts persist
        fire_interval: tuple[float, float] — (min, max) seconds between firings
    """

    name: str
    system_prompt: str
    decay_seconds: float = 90.0
    fire_interval: tuple[float, float] = (5.0, 15.0)

    def __init__(self, board: Board, llm: LLMClient):
        self.board = board
        self.llm = llm
        self._running = False

    async def run(self) -> None:
        """Main loop: sleep for a random interval, then fire."""
        self._running = True
        while self._running:
            delay = random.uniform(*self.fire_interval)
            await asyncio.sleep(delay)
            if not self._running:
                break
            try:
                await self.fire()
            except Exception:
                log.exception(f"[{self.name}] error during fire")

    def stop(self) -> None:
        self._running = False

    async def fire(self) -> None:
        """Read the board, think, post a response."""
        recent = await self.board.get_recent(n=20)
        context = self._build_context(recent)
        raw = await self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=context,
            temperature=self.temperature(),
        )
        post = self.parse_response(raw, recent)
        if post:
            await self.board.post(post)

    def _build_context(self, recent: list[Post]) -> str:
        if not recent:
            return "(The workspace is quiet. Nothing has been posted yet.)"
        lines = [p.as_board_text() for p in recent]
        return "Current workspace:\n\n" + "\n\n".join(lines)

    def parse_response(self, raw: str, recent: list[Post]) -> Post | None:
        """Parse LLM output into a Post. Subclasses can override for
        custom parsing. By default, tries to extract urgency/importance
        from the response or uses defaults."""
        content = raw.strip()
        if not content:
            return None

        urgency, importance = self.default_scores()

        # try to extract scores if the LLM included them
        lines = content.split("\n")
        clean_lines = []
        for line in lines:
            low = line.lower().strip()
            if low.startswith("urgency:"):
                try:
                    urgency = float(low.split(":")[1].strip())
                    urgency = max(0.0, min(1.0, urgency))
                except ValueError:
                    pass
            elif low.startswith("importance:"):
                try:
                    importance = float(low.split(":")[1].strip())
                    importance = max(0.0, min(1.0, importance))
                except ValueError:
                    pass
            else:
                clean_lines.append(line)

        content = "\n".join(clean_lines).strip()
        if not content:
            return None

        return Post(
            author=self.name,
            content=content,
            urgency=urgency,
            importance=importance,
            decay_seconds=self.decay_seconds,
            tags=self.tags(),
        )

    def temperature(self) -> float:
        return 0.8

    def default_scores(self) -> tuple[float, float]:
        """(urgency, importance) defaults for this agent."""
        return (0.5, 0.5)

    def tags(self) -> list[str]:
        return [self.name]

    @abstractmethod
    def agent_description(self) -> str:
        """One-line description for logging/UI."""
        ...
