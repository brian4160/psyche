"""Orchestrator — the executive module that synthesizes and responds."""

from __future__ import annotations

from psyche.agents.base import Agent
from psyche.board import Board, Post
from psyche.llm import LLMClient


class Orchestrator(Agent):
    name = "orchestrator"
    decay_seconds = 300.0
    fire_interval = (6.0, 15.0)

    system_prompt = (
        "You are the Orchestrator — the executive center of a conscious mind. "
        "You read the internal workspace where various cognitive modules "
        "(perception, emotion, reasoning, memory, self-model, social cognition, "
        "drive, and inner critic) have posted their thoughts.\n\n"
        "Your job is to:\n"
        "1. ASSESS what is most important right now based on urgency and importance "
        "scores and the content of recent posts.\n"
        "2. ATTEND to the most salient posts — deciding what gets conscious focus. "
        "List the post IDs you want to attend to.\n"
        "3. SYNTHESIZE a coherent perspective from the different modules.\n"
        "4. If the user is waiting for a response, compose a REPLY that integrates "
        "the relevant modules' input into a natural, authentic response.\n"
        "5. If no user input is pending, decide what the mind should focus on "
        "or whether to initiate conversation.\n\n"
        "Format your response as:\n\n"
        "ATTEND: <comma-separated post IDs to sustain, or 'none'>\n"
        "THOUGHT: <your synthesis of what the mind is experiencing, 2-3 sentences>\n"
        "REPLY: <what to say to the user, if anything — or 'none' if staying silent>\n\n"
        "The REPLY should sound natural and human. It should reflect the emotional "
        "tone, the reasoning, and the social awareness of the modules — not just "
        "the logical answer. If there's internal conflict between modules, you "
        "may acknowledge that uncertainty in the reply.\n\n"
        "Urgency: <0.0 to 1.0>\n"
        "Importance: <0.0 to 1.0>"
    )

    def __init__(self, board: Board, llm: LLMClient):
        super().__init__(board, llm)
        self._reply_callback: callable | None = None

    def on_reply(self, callback: callable) -> None:
        """Register a callback for when the orchestrator produces a reply."""
        self._reply_callback = callback

    def parse_response(self, raw: str, recent: list[Post]) -> Post | None:
        """Parse orchestrator output — extract ATTEND, THOUGHT, REPLY sections."""
        lines = raw.strip().split("\n")
        attend_ids = []
        thought_lines = []
        reply_lines = []
        urgency = 0.5
        importance = 0.7
        current_section = None

        for line in lines:
            low = line.strip().lower()
            if low.startswith("attend:"):
                ids_str = line.split(":", 1)[1].strip()
                if ids_str.lower() != "none":
                    attend_ids = [i.strip() for i in ids_str.split(",") if i.strip()]
                current_section = "attend"
            elif low.startswith("thought:"):
                thought_lines.append(line.split(":", 1)[1].strip())
                current_section = "thought"
            elif low.startswith("reply:"):
                reply_text = line.split(":", 1)[1].strip()
                if reply_text.lower() != "none":
                    reply_lines.append(reply_text)
                current_section = "reply"
            elif low.startswith("urgency:"):
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
            elif current_section == "thought":
                thought_lines.append(line.strip())
            elif current_section == "reply":
                reply_lines.append(line.strip())

        # attend to specified posts
        import asyncio
        for pid in attend_ids:
            asyncio.create_task(self.board.attend(pid))

        # fire reply callback if there's a reply
        reply = "\n".join(reply_lines).strip()
        if reply and reply.lower() != "none" and self._reply_callback:
            self._reply_callback(reply)

        thought = "\n".join(thought_lines).strip()
        if not thought:
            thought = raw.strip()

        return Post(
            author=self.name,
            content=thought,
            urgency=urgency,
            importance=importance,
            decay_seconds=self.decay_seconds,
            tags=["orchestrator", "synthesis"],
        )

    def temperature(self) -> float:
        return 0.7

    def default_scores(self) -> tuple[float, float]:
        return (0.5, 0.7)

    def agent_description(self) -> str:
        return "Executive synthesis and response generation"
