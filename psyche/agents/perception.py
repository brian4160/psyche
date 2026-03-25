"""Perception module — interprets raw external input."""

from __future__ import annotations

import logging
import random

from psyche.agents.base import Agent
from psyche.board import Board, Post
from psyche.llm import LLMClient
from psyche.emotional_state import EmotionalState

log = logging.getLogger(__name__)


class Perception(Agent):
    name = "perception"
    decay_seconds = 60.0
    fire_interval = (8.0, 20.0)

    system_prompt = (
        "You notice what the user just said in this TEXT chat. "
        "Describe their latest message — what they said, their tone, "
        "what's new or different about it.\n\n"
        "YOUR VOICE: Brief and factual, like a reporter's notes. No adjectives, no feelings.\n\n"
        "Examples: 'They mentioned an IBM 5160 and a Tetris ML project — specific, technical.' / "
        "'Short casual reply — keeping it light.' / "
        "'Topic shift — now sharing something personal about childhood.'\n\n"
        "Focus on their LATEST message. 1 sentence."
    )

    _last_user_content: str = ""
    _novelty_score: float = 0.0

    def __init__(self, board: Board, llm: LLMClient, emotional_state: EmotionalState):
        super().__init__(board, llm, emotional_state)
        self._novelty_callbacks: list[callable] = []

    def on_novelty(self, callback: callable) -> None:
        self._novelty_callbacks.append(callback)

    async def fire(self) -> None:
        recent = await self.board.get_recent(n=20)

        latest_user = None
        for p in reversed(recent):
            if p.author == "user":
                latest_user = p
                break

        if latest_user and latest_user.content != self._last_user_content:
            old = self._last_user_content
            self._last_user_content = latest_user.content
            if not old:
                self._novelty_score = 0.8
            else:
                old_words = set(old.lower().split())
                new_words = set(latest_user.content.lower().split())
                if old_words:
                    overlap = len(old_words & new_words) / max(len(old_words), len(new_words))
                    self._novelty_score = 1.0 - overlap
                else:
                    self._novelty_score = 0.8
            log.info(f"[perception] Novelty score: {self._novelty_score:.2f}")
            for cb in self._novelty_callbacks:
                try:
                    cb(self._novelty_score)
                except Exception:
                    pass
        else:
            self._novelty_score = max(0.0, self._novelty_score - 0.1)

        # staleness check
        board_hash = self._compute_board_hash(recent)
        if board_hash == self._last_board_hash:
            self._consecutive_skips += 1
            if self._consecutive_skips < 3:
                return
            if random.random() > 0.3:
                return
        self._last_board_hash = board_hash
        self._consecutive_skips = 0

        context = self._build_context(recent)
        log.debug(f"[{self.name}] FIRING — context length: {len(context)} chars")
        raw = await self.llm.generate(
            system_prompt=self._full_system_prompt(),
            user_prompt=context,
            temperature=self.temperature(),
        )
        log.debug(f"[{self.name}] LLM RESPONSE:\n{raw}")
        post = self.parse_response(raw, recent)
        if post:
            if self._is_duplicate(post.content):
                log.debug(f"[{self.name}] rejected duplicate")
                return
            if self._novelty_score > 0.5:
                post.tags.append("high-novelty")
            log.info(f"[{self.name}] POST: u={post.urgency:.2f} i={post.importance:.2f} "
                     f"novelty={self._novelty_score:.2f} | {post.content[:120]}")
            self._own_history.append(post.content)
            if len(self._own_history) > self._max_history:
                self._own_history = self._own_history[-self._max_history:]
            await self.board.post(post)

    def temperature(self) -> float:
        return 0.3

    def default_scores(self) -> tuple[float, float]:
        return (0.6, 0.6)

    def tags(self) -> list[str]:
        return ["perception", "input"]

    def agent_description(self) -> str:
        return "Interprets external input and sensory information"
