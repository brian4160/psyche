"""Plain single-agent architecture — control condition.

No consciousness theory. Just a single LLM responding to conversation.
This is the ChatGPT-style baseline.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from psyche.architectures.base import Architecture
from psyche.board import Post
from psyche.llm import LLMClient
from psyche.shared_prompts import REPLY_PROMPT, REPLY_CONTEXT_TEMPLATE

log = logging.getLogger(__name__)


class PlainArchitecture(Architecture):
    name = "plain"
    description = "Plain single-agent LLM — no consciousness architecture"

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

            conv_text = self._format_conversation()
            context = REPLY_CONTEXT_TEMPLATE.format(
                conversation=conv_text,
                internal_context="(No additional context.)",
            )

            reply = await self.llm.generate(
                system_prompt=REPLY_PROMPT,
                user_prompt=context,
                temperature=0.75,
            )
            reply = reply.strip()
            if len(reply) > 2 and reply[0] == '"' and reply[-1] == '"':
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
