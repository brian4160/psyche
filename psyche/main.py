"""Entry point — wires all components together and runs the system."""

from __future__ import annotations

import asyncio
import logging
import threading

from psyche.board import Board, Post
from psyche.llm import LLMClient
from psyche.ui import PsycheApp
from psyche.agents.perception import Perception
from psyche.agents.emotion import Emotion
from psyche.agents.reasoning import Reasoning
from psyche.agents.memory import Memory
from psyche.agents.self_model import SelfModel
from psyche.agents.social import SocialCognition
from psyche.agents.drive import Drive
from psyche.agents.critic import InnerCritic
from psyche.agents.orchestrator import Orchestrator

logging.basicConfig(level=logging.WARNING)


class Psyche:
    """The conscious system — ties board, agents, and UI together."""

    def __init__(self):
        self.board = Board()
        self.llm = LLMClient()
        self.app = PsycheApp(on_user_input=self._handle_user_input)
        self._agent_loop: asyncio.AbstractEventLoop | None = None

        # create all agents
        self.orchestrator = Orchestrator(self.board, self.llm)
        self.orchestrator.on_reply(self._handle_reply)

        self.agents = [
            Perception(self.board, self.llm),
            Emotion(self.board, self.llm),
            Reasoning(self.board, self.llm),
            Memory(self.board, self.llm),
            SelfModel(self.board, self.llm),
            SocialCognition(self.board, self.llm),
            Drive(self.board, self.llm),
            InnerCritic(self.board, self.llm),
            self.orchestrator,
        ]

        # subscribe to board for UI updates
        self.board.subscribe(self._on_board_post)

    def _on_board_post(self, post: Post) -> None:
        """Called on every new board post (from any thread)."""
        if post.author != "user":
            self.app.call_from_thread(
                self.app.post_thought,
                post.author,
                post.content,
                post.urgency,
                post.importance,
            )

    def _handle_user_input(self, text: str) -> None:
        """Called by the UI thread when the user submits a message."""
        post = Post(
            author="user",
            content=text,
            urgency=0.9,
            importance=0.9,
            tags=["user-input", "external"],
            decay_seconds=300.0,
        )
        # post_sync is thread-safe
        self.board.post_sync(post)

    def _handle_reply(self, reply: str) -> None:
        """Called from the agent thread when the orchestrator produces a reply."""
        self.app.call_from_thread(self.app.post_chat, "Psyche", reply)

    async def _run_agents(self) -> None:
        """Start all agent loops."""
        tasks = [asyncio.create_task(a.run()) for a in self.agents]
        await asyncio.gather(*tasks)

    def run(self) -> None:
        """Start the entire system."""

        def agent_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._agent_loop = loop
            loop.run_until_complete(self._run_agents())

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()

        self.app.run()

        # cleanup
        for a in self.agents:
            a.stop()


def main():
    psyche = Psyche()
    psyche.run()


if __name__ == "__main__":
    main()
