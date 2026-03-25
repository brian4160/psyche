"""Freudian Psychodynamic architecture.

Freud's structural model: Id (drives), Superego (morality), Ego (mediator).

Architecture:
  1. Id generates raw impulse/desire (internal processing)
  2. Superego generates moral/social judgment (internal processing)
  3. Shared reply prompt generates final reply using Id+Superego as context

The key insight: personality and depth emerge from TENSION between
competing psychic forces.
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading

from psyche.architectures.base import Architecture
from psyche.board import Post
from psyche.llm import LLMClient
from psyche.shared_prompts import REPLY_PROMPT, REPLY_CONTEXT_TEMPLATE

log = logging.getLogger(__name__)


class FreudianArchitecture(Architecture):
    name = "freudian"
    description = "Freudian psychodynamic — Id/Ego/Superego conflict"

    # Internal processing prompts (unique to Freudian — this IS the architecture)
    ID_PROMPT = (
        "You are the ID — the primal, instinctive part of a mind. You want "
        "what you want, immediately, without filter. You are desire, curiosity, "
        "hunger, boredom, attraction, irritation, playfulness.\n\n"
        "Read what the person said and express your RAW, unfiltered impulse. "
        "What do you WANT right now? What's your gut reaction?\n\n"
        "Be honest and primal. No social filter. 1-2 sentences.\n"
        "You are NOT talking to the person — this is internal."
    )

    SUPEREGO_PROMPT = (
        "You are the SUPEREGO — the moral conscience of a mind. You represent "
        "internalized social norms, values, and standards. You judge what is "
        "appropriate, kind, honest, and right.\n\n"
        "Read what the person said and what the Id wants. Then give your "
        "moral/social judgment. What SHOULD we do? What's the right thing?\n\n"
        "Be principled but not rigid. 1-2 sentences.\n"
        "You are NOT talking to the person — this is internal."
    )

    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self._conversation: list[dict] = []
        self._pending_user: Post | None = None
        self._reply_event = threading.Event()
        self._repressed: list[str] = []

    def inject_user_message(self, text: str) -> None:
        self._conversation.append({"role": "user", "content": text})
        post = Post(
            author="user", content=text,
            urgency=0.9, importance=0.9,
            tags=["user-input"], decay_seconds=300.0,
        )
        self.board.post_sync(post)
        self._pending_user = post
        self._reply_event.set()

    def start_background(self) -> None:
        self._running = True

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_until_complete(self._main_loop())

        t = threading.Thread(target=run, daemon=True)
        t.start()

    async def _main_loop(self) -> None:
        while self._running:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._reply_event.wait(timeout=2.0)
            )
            if not self._running:
                break
            if not self._reply_event.is_set():
                continue
            self._reply_event.clear()

            if self._pending_user is None:
                continue

            user_post = self._pending_user
            self._pending_user = None
            await self._process(user_post)

    async def _process(self, user_post: Post) -> None:
        """Three-stage: Id → Superego → shared reply prompt with conflict as context."""
        conv_text = self._format_conversation()

        # Stage 1: Id — raw impulse
        id_context = (
            f"CONVERSATION:\n{conv_text}\n\n"
            f"THEIR LATEST: \"{user_post.content}\"\n\n"
            f"What is your raw, unfiltered impulse?"
        )
        id_response = await self.llm.generate(
            system_prompt=self.ID_PROMPT,
            user_prompt=id_context,
            temperature=0.95,
        )
        id_response = id_response.strip()
        self._log_thought("id", id_response)

        # Stage 2: Superego — moral judgment
        superego_context = (
            f"CONVERSATION:\n{conv_text}\n\n"
            f"THEIR LATEST: \"{user_post.content}\"\n\n"
            f"THE ID WANTS: \"{id_response}\"\n\n"
            f"What is your moral/social judgment?"
        )
        superego_response = await self.llm.generate(
            system_prompt=self.SUPEREGO_PROMPT,
            user_prompt=superego_context,
            temperature=0.5,
        )
        superego_response = superego_response.strip()
        self._log_thought("superego", superego_response)

        # Track repression
        repression_words = ["inappropriate", "shouldn't", "wrong", "harmful",
                          "selfish", "rude", "disrespectful"]
        if any(w in superego_response.lower() for w in repression_words):
            self._repressed.append(id_response)
            self._log_thought("repression", f"Id impulse repressed: {id_response[:80]}")

        # Stage 3: Final reply using SHARED prompt with conflict as context
        internal_context = (
            f"YOUR GUT FEELING: {id_response}\n"
            f"YOUR BETTER JUDGMENT: {superego_response}\n"
            f"Let both of these inform your reply naturally. Don't mention "
            f"them explicitly — just let them shape your tone and content."
        )
        context = REPLY_CONTEXT_TEMPLATE.format(
            conversation=conv_text,
            internal_context=internal_context,
        )
        reply = await self.llm.generate(
            system_prompt=REPLY_PROMPT,
            user_prompt=context,
            temperature=0.75,
        )
        reply = reply.strip()

        if reply and len(reply) > 2 and reply[0] == '"' and reply[-1] == '"':
            reply = reply[1:-1].strip()

        if reply:
            self._conversation.append({"role": "self", "content": reply})
            self.board.post_sync(Post(
                author="self", content=reply,
                urgency=0.3, importance=0.5,
                decay_seconds=300.0, tags=["self-speech"],
            ))
            self._emit_reply(reply)

    def _format_conversation(self) -> str:
        lines = []
        for turn in self._conversation[-10:]:
            label = "THEM" if turn["role"] == "user" else "YOU"
            lines.append(f"{label}: {turn['content']}")
        return "\n".join(lines) if lines else "(conversation just started)"
