"""Higher-Order Thought (HOT) architecture.

Rosenthal's Higher-Order Theory: a mental state is conscious only when
there is a higher-order representation of it. Consciousness IS the
meta-cognitive layer, not the first-order processing.

Architecture:
  Layer 1 (unconscious): First-order processor generates a raw reaction
  Layer 2 (conscious):   Meta-cognitive monitor reflects on the reaction
  Layer 3 (output):      Shared reply prompt generates final reply using reflection as context

The key insight: the response is informed by REFLECTION on the draft,
not by the draft itself directly.
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


class HOTArchitecture(Architecture):
    name = "hot"
    description = "Higher-Order Thought — two-layer conscious reflection"

    # Internal processing prompts (unique to HOT — this IS the architecture)
    FIRST_ORDER_PROMPT = (
        "You are having a conversation. What is your immediate, unfiltered "
        "THOUGHT about what the person just said? Not a reply to them — "
        "just your raw internal reaction.\n\n"
        "Examples: 'Oh cool, that reminds me of my old Compaq.' / "
        "'They seem kind of down today.' / 'I actually disagree with that.'\n\n"
        "1-2 sentences. Internal thought only, not a message to them."
    )

    META_COGNITIVE_PROMPT = (
        "You are reflecting on your own draft reaction to someone's message. "
        "Consider:\n"
        "- Is this draft genuine or just polite filler?\n"
        "- Did I actually react to what they said, or did I dodge it?\n"
        "- Am I asking a question just to seem interested, or do I actually care?\n"
        "- What would a REAL friend say here instead?\n\n"
        "Write a 1-sentence correction. Be specific about what to fix.\n"
        "Example: 'The draft asks a question instead of sharing my own take — "
        "I should relate it to my own experience instead.'"
    )

    def __init__(self, llm: LLMClient):
        super().__init__(llm)
        self._conversation: list[dict] = []
        self._pending_user: Post | None = None
        self._reply_event = threading.Event()

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
        """Three-step: first-order draft → meta-cognitive reflection → shared reply prompt."""
        conv_text = self._format_conversation()

        # Layer 1: First-order reaction (unconscious)
        first_order_context = (
            f"CONVERSATION:\n{conv_text}\n\n"
            f"Their latest message: \"{user_post.content}\"\n\n"
            f"Write your immediate reaction."
        )
        draft = await self.llm.generate(
            system_prompt=self.FIRST_ORDER_PROMPT,
            user_prompt=first_order_context,
            temperature=0.8,
        )
        draft = draft.strip()
        self._log_thought("first-order", f"Draft: {draft}")

        # Layer 2: Meta-cognitive reflection (conscious)
        meta_context = (
            f"THEIR MESSAGE: \"{user_post.content}\"\n\n"
            f"YOUR DRAFT REACTION: \"{draft}\"\n\n"
            f"What do you notice about your own reaction?"
        )
        reflection = await self.llm.generate(
            system_prompt=self.META_COGNITIVE_PROMPT,
            user_prompt=meta_context,
            temperature=0.7,
        )
        reflection = reflection.strip()
        self._log_thought("meta-cognitive", reflection)

        # Layer 3: Final reply using SHARED prompt
        internal_context = (
            f"YOUR GUT THOUGHT: {draft}\n"
            f"ON REFLECTION: {reflection}"
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
