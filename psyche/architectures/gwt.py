"""GWT architecture wrapper — delegates to the existing Psyche system.

This wraps the full GWT multi-agent system (board, 8 modules, attention gate,
orchestrator) into the Architecture interface for the test harness.
"""

from __future__ import annotations

import logging

from psyche.architectures.base import Architecture
from psyche.config import get_condition
from psyche.llm import LLMClient

log = logging.getLogger(__name__)


class GWTArchitecture(Architecture):
    name = "gwt"
    description = "Global Workspace Theory — multi-agent broadcast workspace"

    def __init__(self, llm: LLMClient, agents_override: list[str] | None = None):
        super().__init__(llm)
        # import here to avoid circular imports
        from psyche.main import Psyche
        from psyche.config import TheoryConfig
        if agents_override:
            config = TheoryConfig(
                name="gwt-ablation",
                description=f"GWT with {len(agents_override)} modules",
                agents=agents_override,
                attention_gate=True,
                prediction_engine=True,
                emotional_state=True,
                conversation_boundaries=True,
            )
        else:
            config = get_condition("gwt")
        self._psyche = Psyche(config=config, ui=False)
        # share the LLM instance
        self._psyche.llm = llm
        for agent in self._psyche.agents:
            agent.llm = llm
        # wire reply callbacks through
        self._psyche.on_reply(self._emit_reply)
        self.board = self._psyche.board

    def inject_user_message(self, text: str) -> None:
        self._psyche.inject_user_message(text)

    def start_background(self) -> None:
        self._running = True
        self._psyche.start_background()

    def stop(self) -> None:
        self._running = False
        self._psyche.stop()
