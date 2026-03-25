"""Plain-Multi architecture — compute-controlled baseline.

Makes the same number of LLM calls as GWT (~8) but WITHOUT any architectural
structure. Generates 8 independent candidate replies and picks the best one
via a selection pass. This controls for the confound that more LLM calls
= more "thinking" regardless of theory.

If GWT outperforms plain-multi, it's the ARCHITECTURE that matters,
not just additional compute.
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

NUM_CANDIDATES = 8


class PlainMultiArchitecture(Architecture):
    name = "plain-multi"
    description = "Plain multi-call LLM — compute-controlled baseline (8 candidates, pick best)"

    SELECTOR_PROMPT = (
        "You are selecting the BEST reply from several candidates. "
        "Pick the one that sounds most natural, engaging, and human-like. "
        "It should directly engage with what the person said and feel authentic.\n\n"
        "Reply with ONLY the number of the best candidate (1-8)."
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
        """Generate N candidates sequentially, then pick the best."""
        conv_text = self._format_conversation()

        # generate candidates (sequentially — same serialization as GWT)
        candidates = []
        for i in range(NUM_CANDIDATES):
            context = REPLY_CONTEXT_TEMPLATE.format(
                conversation=conv_text,
                internal_context="(No additional context.)",
            )
            # vary temperature slightly for diversity
            temp = 0.6 + (i * 0.05)
            candidate = await self.llm.generate(
                system_prompt=REPLY_PROMPT,
                user_prompt=context,
                temperature=temp,
            )
            candidate = candidate.strip()
            if candidate and len(candidate) > 2:
                if candidate[0] == '"' and candidate[-1] == '"':
                    candidate = candidate[1:-1].strip()
                candidates.append(candidate)
                self._log_thought(f"candidate-{i+1}", candidate)

        if not candidates:
            return

        # if only one candidate, use it
        if len(candidates) == 1:
            reply = candidates[0]
        else:
            # selection pass — pick the best
            candidate_list = "\n".join(
                f"{i+1}. \"{c}\"" for i, c in enumerate(candidates)
            )
            selector_context = (
                f"THEIR MESSAGE: \"{user_post.content}\"\n\n"
                f"CANDIDATES:\n{candidate_list}\n\n"
                f"Which number is the best reply? Reply with ONLY the number."
            )
            selection = await self.llm.generate(
                system_prompt=self.SELECTOR_PROMPT,
                user_prompt=selector_context,
                temperature=0.1,
            )
            # parse selection
            try:
                idx = int(selection.strip().replace(".", "")) - 1
                if 0 <= idx < len(candidates):
                    reply = candidates[idx]
                else:
                    reply = candidates[0]
            except (ValueError, IndexError):
                reply = candidates[0]

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
