"""Orchestrator — the executive module that synthesizes and responds."""

from __future__ import annotations

import asyncio
import logging
import time

from psyche.agents.base import Agent
from psyche.board import Board, Post
from psyche.llm import LLMClient
from psyche.emotional_state import EmotionalState
from psyche.shared_prompts import REPLY_PROMPT as SHARED_REPLY_PROMPT, REPLY_CONTEXT_TEMPLATE

log = logging.getLogger(__name__)


class Orchestrator(Agent):
    name = "orchestrator"
    decay_seconds = 300.0
    fire_interval = (10.0, 20.0)

    system_prompt = (
        "Summarize what the modules are saying in 1 sentence. "
        "What should we focus on right now? Be direct and plain."
    )

    # Uses SHARED_REPLY_PROMPT from shared_prompts.py for experimental consistency

    # minimum modules that should have posted since last user input
    MIN_MODULES_BEFORE_REPLY = 3

    def __init__(self, board: Board, llm: LLMClient, emotional_state: EmotionalState):
        super().__init__(board, llm, emotional_state)
        self._reply_callback: callable | None = None
        self._last_user_post_id: str | None = None
        self._replied_to: set[str] = set()
        self._last_reply: str = ""
        self._reply_history: list[str] = []
        self._max_reply_history = 5
        self._plain_mode: bool = False  # set True for control condition

    async def _fire_plain(self, recent: list[Post]) -> None:
        """Plain mode — direct reply without module synthesis."""
        latest_user = None
        for p in reversed(recent):
            if p.author == "user":
                latest_user = p
                break
        if not latest_user or latest_user.id in self._replied_to:
            return

        # build simple conversation context
        conv_lines = []
        for p in recent:
            if p.author == "user":
                conv_lines.append(f"THEM: {p.content}")
            elif p.author == "self":
                conv_lines.append(f"YOU: {p.content}")
        context = "CONVERSATION:\n" + "\n".join(conv_lines) if conv_lines else ""
        context += "\n\nWrite your reply now. ONLY the reply text."

        reply = await self.llm.generate(
            system_prompt=SHARED_REPLY_PROMPT,
            user_prompt=context,
            temperature=0.75,
        )
        reply = self._clean_response(reply)
        if reply and len(reply) > 2 and reply[0] == '"' and reply[-1] == '"':
            reply = reply[1:-1].strip()

        if reply and self._reply_callback:
            log.info(f"[orchestrator/plain] REPLY: {reply}")
            self._replied_to.add(latest_user.id)
            self._last_reply = reply
            self._reply_history.append(reply)
            self._reply_callback(reply)
            await self.board.post(Post(
                author="self", content=reply,
                urgency=0.3, importance=0.5,
                decay_seconds=300.0, tags=["self-speech", "external"],
            ))

    def on_reply(self, callback: callable) -> None:
        self._reply_callback = callback

    # Plain mode also uses the shared reply prompt for consistency

    async def fire(self) -> None:
        """Two-phase fire: synthesize thoughts, then decide whether to reply."""
        recent = await self.board.get_recent(n=20)

        # plain mode: no modules, just direct reply
        if self._plain_mode:
            await self._fire_plain(recent)
            return

        # Phase 1: synthesize internal state (always)
        context = self._build_context(recent)
        log.debug(f"[orchestrator] FIRING — {len(recent)} recent posts")
        thought = await self.llm.generate(
            system_prompt=self._full_system_prompt(),
            user_prompt=context,
            temperature=0.7,
        )
        log.debug(f"[orchestrator] THOUGHT:\n{thought}")

        # clean thought
        thought = self._clean_response(thought)
        if thought:
            self._own_history.append(thought)
            if len(self._own_history) > self._max_history:
                self._own_history = self._own_history[-self._max_history:]
            await self.board.post(Post(
                author=self.name,
                content=thought,
                urgency=0.5,
                importance=0.7,
                decay_seconds=self.decay_seconds,
                tags=["orchestrator", "synthesis"],
            ))

        # Phase 2: decide whether to reply
        latest_user = None
        for p in reversed(recent):
            if p.author == "user":
                latest_user = p
                break

        if not latest_user:
            # no user input — check if we should initiate
            await self._maybe_initiate(recent, thought)
            return

        if latest_user.id in self._replied_to:
            log.debug("[orchestrator] Already replied to latest user message")
            return

        # check if enough modules have weighed in
        modules_since_user = set()
        for p in recent:
            if p.timestamp > latest_user.timestamp and p.author not in ("user", "self", "orchestrator"):
                modules_since_user.add(p.author)

        time_since_user = latest_user.age()
        if len(modules_since_user) < self.MIN_MODULES_BEFORE_REPLY:
            # timeout fallback: if 30s passed, reply with whatever we have
            if time_since_user < 30:
                log.debug(f"[orchestrator] Waiting for more modules "
                          f"({len(modules_since_user)}/{self.MIN_MODULES_BEFORE_REPLY})")
                return
            else:
                log.info(f"[orchestrator] Timeout — replying with {len(modules_since_user)} modules")

        # compose reply using SHARED prompt for experimental consistency
        reply_context = self._build_reply_context(recent, thought)
        reply = await self.llm.generate(
            system_prompt=SHARED_REPLY_PROMPT,
            user_prompt=reply_context,
            temperature=0.75,
        )
        reply = self._clean_response(reply)

        # strip wrapping quotes
        if reply and len(reply) > 2 and reply[0] == '"' and reply[-1] == '"':
            reply = reply[1:-1].strip()

        # filter out assistant-like replies
        banned = ["how can i assist", "how can i help", "anything interesting going on"]
        if reply and any(b in reply.lower() for b in banned):
            log.debug(f"[orchestrator] Rejected assistant-like reply: {reply[:80]}")
            reply = None

        if reply and self._reply_callback:
            log.info(f"[orchestrator] REPLY: {reply}")
            self._replied_to.add(latest_user.id)
            self._last_reply = reply
            self._last_user_post_id = latest_user.id
            self._reply_history.append(reply)
            if len(self._reply_history) > self._max_reply_history:
                self._reply_history = self._reply_history[-self._max_reply_history:]
            self._reply_callback(reply)

            # post our reply to the board so all modules can see it
            await self.board.post(Post(
                author="self",
                content=reply,
                urgency=0.3,
                importance=0.5,
                decay_seconds=300.0,
                tags=["self-speech", "external"],
            ))

    async def _maybe_initiate(self, recent: list[Post], thought: str) -> None:
        """Consider initiating conversation during idle periods."""
        # only initiate if it's been quiet for a while
        last_speech = None
        for p in reversed(recent):
            if p.author in ("user", "self"):
                last_speech = p
                break

        if not last_speech:
            idle_time = time.time() - self._conversation_start
        else:
            idle_time = last_speech.age()

        if idle_time < 45:
            return  # too soon

        # check if drive is pushing for something
        drive_pushing = any(
            p.author == "drive" and p.age() < 30 and p.importance > 0.6
            for p in recent
        )

        if not drive_pushing:
            return

        log.info("[orchestrator] Initiating conversation (idle + drive push)")
        reply_context = self._build_reply_context(recent, thought)
        reply_context += (
            "\n\nThe conversation has been quiet. You want to say something — "
            "share a thought, ask a question, or bring up something interesting. "
            "Be natural, like a person who's been sitting in comfortable silence "
            "and has something on their mind."
        )
        reply = await self.llm.generate(
            system_prompt=SHARED_REPLY_PROMPT,
            user_prompt=reply_context,
            temperature=0.85,
        )
        reply = self._clean_response(reply)
        if reply and len(reply) > 2 and reply[0] == '"' and reply[-1] == '"':
            reply = reply[1:-1].strip()
        if reply and self._reply_callback:
            self._reply_callback(reply)
            await self.board.post(Post(
                author="self",
                content=reply,
                urgency=0.3,
                importance=0.5,
                decay_seconds=300.0,
                tags=["self-speech", "external", "initiated"],
            ))

    def _build_reply_context(self, recent: list[Post], thought: str) -> str:
        """Build reply context using the shared template format."""
        # conversation history
        conv_lines = []
        for p in recent:
            if p.author == "user":
                conv_lines.append(f"THEM: {p.content}")
            elif p.author == "self":
                conv_lines.append(f"YOU: {p.content}")
        conversation = "\n".join(conv_lines) if conv_lines else "(just started)"

        # internal context — unique to GWT (module synthesis)
        internal_parts = []
        internal_parts.append(f"YOUR SYNTHESIS: {thought}")

        mood = self.emotional_state.get_mood()
        internal_parts.append(f"YOUR MOOD: {mood.describe()}")

        # key module inputs — one per module, most recent
        seen = {}
        for p in recent:
            if p.author not in ("user", "self", "orchestrator") and p.age() < 60:
                seen[p.author] = f"  {p.author}: {p.content}"
        if seen:
            internal_parts.append("MODULE THOUGHTS:\n" + "\n".join(seen.values()))

        drive_posts = [p for p in recent if p.author == "drive" and p.age() < 30]
        if drive_posts:
            internal_parts.append(f"DRIVE WANTS: {drive_posts[-1].content}")

        if self._reply_history:
            avoid_lines = "\n".join(f"  - \"{r}\"" for r in self._reply_history[-3:])
            internal_parts.append(
                f"YOUR RECENT REPLIES (vary your style):\n{avoid_lines}"
            )

        internal_context = "\n".join(internal_parts)

        return REPLY_CONTEXT_TEMPLATE.format(
            conversation=conversation,
            internal_context=internal_context,
        )

    def _clean_response(self, raw: str) -> str:
        """Strip meta-formatting from LLM output."""
        lines = []
        for line in raw.strip().split("\n"):
            low = line.lower().strip()
            if any(low.startswith(p) for p in
                   ("attend:", "thought:", "reply:", "urgency:", "importance:",
                    "note:", "---", "conversation state")):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def temperature(self) -> float:
        return 0.7

    def default_scores(self) -> tuple[float, float]:
        return (0.5, 0.7)

    def agent_description(self) -> str:
        return "Executive synthesis and response generation"
