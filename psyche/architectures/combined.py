"""Combined architectures — layering multiple consciousness theories.

GWT modules generate internal thoughts, then HOT and/or Freudian processing
adds additional context. The FINAL reply always uses the shared reply prompt
with accumulated context from all active theories.

This ensures prompt consistency: the only variable is what CONTEXT
the theories provide, not how the reply is generated.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from psyche.architectures.base import Architecture
from psyche.board import Board, Post
from psyche.config import get_condition
from psyche.llm import LLMClient
from psyche.shared_prompts import REPLY_PROMPT, REPLY_CONTEXT_TEMPLATE

log = logging.getLogger(__name__)


class CombinedArchitecture(Architecture):
    """Combines GWT with optional HOT and/or Freudian layers.

    Pipeline:
    1. GWT modules fire and post to workspace → orchestrator synthesizes
    2. Orchestrator produces a draft reply (intercepted before delivery)
    3. If HOT: meta-cognitive reflection on the draft
    4. If Freudian: Id/Superego react to the draft
    5. All accumulated context fed to SHARED reply prompt → final reply
    """

    # Internal processing prompts (unique to each theory layer)
    HOT_REFLECTION_PROMPT = (
        "You are reflecting on a draft reply. Consider:\n"
        "- Is this genuine or just polite filler?\n"
        "- Does it react to what they actually said?\n"
        "- Is it asking questions just to seem interested?\n"
        "- What would a real friend say differently?\n\n"
        "Write 1 sentence: what should change about this draft?\n"
        "Example: 'Too generic — share a personal experience instead of asking a question.'"
    )

    ID_IMPULSE_PROMPT = (
        "You are the ID — raw desire. Given this draft reply, what does your "
        "gut ACTUALLY want to say? What's the unfiltered impulse? 1 sentence."
    )

    SUPEREGO_JUDGMENT_PROMPT = (
        "You are the SUPEREGO — moral conscience. Given this draft reply and "
        "the Id's impulse, is this appropriate? What should we do? 1 sentence."
    )

    def __init__(self, llm: LLMClient, use_hot: bool = False, use_freudian: bool = False):
        self.use_hot = use_hot
        self.use_freudian = use_freudian

        parts = ["gwt"]
        if use_hot:
            parts.append("hot")
        if use_freudian:
            parts.append("freudian")
        self._name = "+".join(parts)

        super().__init__(llm)

        # create the GWT system
        from psyche.main import Psyche
        config = get_condition("gwt")
        self._psyche = Psyche(config=config, ui=False)
        self._psyche.llm = llm
        for agent in self._psyche.agents:
            agent.llm = llm

        # intercept the orchestrator's reply for post-processing
        self._psyche.orchestrator._reply_callback = self._on_gwt_reply
        self.board = self._psyche.board

        self._pending_draft: str | None = None
        self._pending_user_content: str = ""
        self._conversation: list[dict] = []
        self._reply_ready = threading.Event()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def description(self) -> str:
        parts = ["GWT"]
        if self.use_hot:
            parts.append("Higher-Order Thought")
        if self.use_freudian:
            parts.append("Freudian psychodynamic")
        return " + ".join(parts)

    @description.setter
    def description(self, val):
        pass

    def _on_gwt_reply(self, draft: str) -> None:
        """Intercept GWT's draft reply for additional processing."""
        self._pending_draft = draft
        self._reply_ready.set()

    def inject_user_message(self, text: str) -> None:
        self._pending_user_content = text
        self._conversation.append({"role": "user", "content": text})
        self._pending_draft = None
        self._reply_ready.clear()
        self._psyche.inject_user_message(text)

    def start_background(self) -> None:
        self._running = True
        self._psyche.start_background()

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._post_process_loop())

        t = threading.Thread(target=run, daemon=True)
        t.start()

    async def _post_process_loop(self) -> None:
        """Wait for GWT draft, apply theory layers, generate final reply."""
        while self._running:
            got = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._reply_ready.wait(timeout=2.0)
            )
            if not self._running:
                break
            if not got or self._pending_draft is None:
                continue

            draft = self._pending_draft
            user_content = self._pending_user_content
            self._pending_draft = None
            self._reply_ready.clear()

            # accumulate context from theory layers
            internal_parts = [f"GWT DRAFT: {draft}"]

            # HOT layer
            if self.use_hot:
                reflection = await self._get_hot_reflection(draft, user_content)
                internal_parts.append(f"META-COGNITIVE REFLECTION: {reflection}")

            # Freudian layer
            if self.use_freudian:
                id_impulse, superego_judgment = await self._get_freudian_conflict(
                    draft, user_content
                )
                internal_parts.append(f"GUT FEELING: {id_impulse}")
                internal_parts.append(f"BETTER JUDGMENT: {superego_judgment}")
                internal_parts.append(
                    "Let both inform your reply naturally — don't mention them explicitly."
                )

            # Final reply using SHARED prompt
            conv_text = self._format_conversation()
            internal_context = "\n".join(internal_parts)
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
                    decay_seconds=300.0, tags=["self-speech", "combined"],
                ))
                self._emit_reply(reply)

    async def _get_hot_reflection(self, draft: str, user_content: str) -> str:
        """HOT meta-cognitive reflection on the draft."""
        context = (
            f"THEIR MESSAGE: \"{user_content}\"\n"
            f"DRAFT REPLY: \"{draft}\"\n\n"
            f"What do you notice about this draft?"
        )
        reflection = await self.llm.generate(
            system_prompt=self.HOT_REFLECTION_PROMPT,
            user_prompt=context,
            temperature=0.7,
        )
        reflection = reflection.strip()
        self._log_thought("hot-reflection", reflection)
        return reflection

    async def _get_freudian_conflict(self, draft: str, user_content: str) -> tuple[str, str]:
        """Get Id impulse and Superego judgment on the draft."""
        # Id
        id_context = (
            f"THEIR MESSAGE: \"{user_content}\"\n"
            f"DRAFT REPLY: \"{draft}\"\n\n"
            f"What does your gut actually want to say?"
        )
        id_response = await self.llm.generate(
            system_prompt=self.ID_IMPULSE_PROMPT,
            user_prompt=id_context,
            temperature=0.95,
        )
        id_response = id_response.strip()
        self._log_thought("id", id_response)

        # Superego
        superego_context = (
            f"THEIR MESSAGE: \"{user_content}\"\n"
            f"DRAFT REPLY: \"{draft}\"\n"
            f"ID WANTS: \"{id_response}\"\n\n"
            f"Is this appropriate? What should we do?"
        )
        superego_response = await self.llm.generate(
            system_prompt=self.SUPEREGO_JUDGMENT_PROMPT,
            user_prompt=superego_context,
            temperature=0.5,
        )
        superego_response = superego_response.strip()
        self._log_thought("superego", superego_response)

        return (id_response, superego_response)

    def _format_conversation(self) -> str:
        lines = []
        for turn in self._conversation[-10:]:
            label = "THEM" if turn["role"] == "user" else "YOU"
            lines.append(f"{label}: {turn['content']}")
        return "\n".join(lines) if lines else "(conversation just started)"

    def stop(self) -> None:
        self._running = False
        self._reply_ready.set()
        self._psyche.stop()
