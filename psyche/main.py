"""Entry point — wires all components together and runs the system."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from datetime import datetime

from psyche.board import Board, Post
from psyche.llm import LLMClient
from psyche.emotional_state import EmotionalState
from psyche.attention import AttentionGate
from psyche.prediction import PredictionEngine
from psyche.config import TheoryConfig, get_condition
from psyche.ui import PsycheApp

# agent imports
from psyche.agents.perception import Perception
from psyche.agents.emotion import Emotion
from psyche.agents.reasoning import Reasoning
from psyche.agents.memory import Memory
from psyche.agents.self_model import SelfModel
from psyche.agents.social import SocialCognition
from psyche.agents.drive import Drive
from psyche.agents.critic import InnerCritic
from psyche.agents.hot import HigherOrderThought
from psyche.agents.ast_agent import AttentionSchema
from psyche.agents.integration import Integration
from psyche.agents.orchestrator import Orchestrator

LOG_DIR = os.path.expanduser("~/psyche/logs")

# registry mapping agent names to classes
AGENT_REGISTRY = {
    "perception": Perception,
    "emotion": Emotion,
    "reasoning": Reasoning,
    "memory": Memory,
    "self-model": SelfModel,
    "social": SocialCognition,
    "drive": Drive,
    "critic": InnerCritic,
    "hot": HigherOrderThought,
    "ast": AttentionSchema,
    "integration": Integration,
}


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"session_{timestamp}.log")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    logging.basicConfig(level=logging.DEBUG, handlers=[fh, ch])

    for name in ("httpcore", "httpx", "hpack", "h2"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_file


class Psyche:
    """The conscious system — ties board, agents, and UI together.

    Accepts a TheoryConfig to control which consciousness theories
    (and corresponding agent modules) are active.
    """

    def __init__(self, config: TheoryConfig | None = None, ui: bool = True):
        if config is None:
            config = get_condition("all_theories")
        self.config = config

        log = logging.getLogger("psyche")
        log.info(f"Starting Psyche with condition: {config.name} — {config.description}")

        self.board = Board()
        self.llm = LLMClient()
        self.emotional_state = EmotionalState() if config.emotional_state else None
        self.attention = AttentionGate() if config.attention_gate else None
        self.prediction = PredictionEngine() if config.prediction_engine else None

        if ui:
            self.app = PsycheApp(on_user_input=self._handle_user_input)
        else:
            self.app = None

        self._agent_loop: asyncio.AbstractEventLoop | None = None
        self._last_user_time: float = 0.0
        self._conversation_gap_threshold: float = 120.0

        # create orchestrator (always present)
        dummy_emotional = self.emotional_state or EmotionalState()
        self.orchestrator = Orchestrator(self.board, self.llm, dummy_emotional)
        self.orchestrator.on_reply(self._handle_reply)

        # create configured agents
        self.agents = []
        self._agent_map = {}
        for agent_name in config.agents:
            if agent_name not in AGENT_REGISTRY:
                log.warning(f"Unknown agent: {agent_name}, skipping")
                continue
            cls = AGENT_REGISTRY[agent_name]
            agent = cls(self.board, self.llm, dummy_emotional)
            self.agents.append(agent)
            self._agent_map[agent_name] = agent

        # plain mode: orchestrator acts as direct chatbot without modules
        if not config.agents:
            self.orchestrator._plain_mode = True
            self.orchestrator.fire_interval = (2.0, 5.0)  # faster in plain mode

        # add orchestrator last
        self.agents.append(self.orchestrator)

        # give all agents access to prediction engine
        if self.prediction:
            for agent in self.agents:
                agent.prediction_engine = self.prediction

        # burst priority order
        self._burst_priority = []
        if "perception" in self._agent_map:
            self._burst_priority.append([self._agent_map["perception"]])
        burst_tier2 = []
        for name in ("emotion", "social"):
            if name in self._agent_map:
                burst_tier2.append(self._agent_map[name])
        if burst_tier2:
            self._burst_priority.append(burst_tier2)

        # novelty callback
        if "perception" in self._agent_map:
            self._agent_map["perception"].on_novelty(self._on_novelty)

        # module tracking for orchestrator trigger
        self._modules_since_user: set[str] = set()
        self._awaiting_reply = False

        # subscribe to board
        self.board.subscribe(self._on_board_post)

        # reply callback for headless mode (A/B testing)
        self._reply_callbacks: list[callable] = []

    def on_reply(self, callback: callable) -> None:
        """Register additional reply callback (used by test harness)."""
        self._reply_callbacks.append(callback)

    def _on_board_post(self, post: Post) -> None:
        """Called on every new board post (from any thread)."""
        log = logging.getLogger("psyche.board")

        # attention gate
        if self.attention and post.author not in ("user", "self", "orchestrator"):
            if not self.attention.submit(post):
                log.debug(f"UNCONSCIOUS [{post.author}] | {post.content[:80]}")
                return

        log.info(f"BOARD [{post.author}] u={post.urgency:.2f} i={post.importance:.2f} "
                 f"decay={post.decay_seconds}s | {post.content}")

        # track module posts for orchestrator trigger
        if post.author == "user":
            self._modules_since_user.clear()
            self._awaiting_reply = True
        elif post.author == "self":
            self._awaiting_reply = False
        elif post.author not in ("orchestrator",) and self._awaiting_reply:
            self._modules_since_user.add(post.author)
            if len(self._modules_since_user) >= self.orchestrator.MIN_MODULES_BEFORE_REPLY:
                self.orchestrator.trigger_burst()

        # update emotional state
        if self.emotional_state and post.author == "emotion":
            self.emotional_state.process_emotion_post(post.content)

        # update UI
        if self.app:
            if post.author == "user":
                return
            elif post.author == "self":
                return
            else:
                self.app.call_from_thread(
                    self.app.post_thought,
                    post.author,
                    post.content,
                    post.urgency,
                    post.importance,
                )

    def _on_novelty(self, score: float) -> None:
        log = logging.getLogger("psyche.novelty")
        log.info(f"Novelty: {score:.2f}")
        if score > 0.3:
            for agent in self.agents:
                if agent.name not in ("perception", "orchestrator"):
                    agent.fire_interval = (
                        max(2.0, agent.fire_interval[0] * 0.5),
                        max(4.0, agent.fire_interval[1] * 0.5),
                    )
            if self._agent_loop:
                self._agent_loop.call_later(20.0, self._reset_fire_intervals)

    def _reset_fire_intervals(self) -> None:
        for agent in self.agents:
            name = agent.name
            if name in AGENT_REGISTRY:
                agent.fire_interval = AGENT_REGISTRY[name].fire_interval

    def _handle_user_input(self, text: str) -> None:
        """Called by the UI thread when the user submits a message."""
        self.inject_user_message(text)

    def inject_user_message(self, text: str) -> None:
        """Inject a user message into the system (used by both UI and test harness)."""
        log = logging.getLogger("psyche.user")
        now = time.time()

        # conversation boundary detection
        if self.config.conversation_boundaries and self._last_user_time > 0:
            gap = now - self._last_user_time
            if gap > self._conversation_gap_threshold:
                log.info(f"CONVERSATION BOUNDARY: {gap:.0f}s gap")
                self.board.post_sync(Post(
                    author="system",
                    content=f"[Long pause — {int(gap)}s since last message. They're back.]",
                    urgency=0.5, importance=0.7,
                    tags=["boundary", "system"], decay_seconds=60.0,
                ))
        self._last_user_time = now

        log.info(f"USER INPUT: {text}")

        # update prediction engine
        if self.prediction:
            self.prediction.process_user_message(text)

        post = Post(
            author="user", content=text,
            urgency=0.9, importance=0.9,
            tags=["user-input", "external"], decay_seconds=300.0,
        )
        self.board.post_sync(post)

        # trigger burst firing
        for tier in self._burst_priority:
            for agent in tier:
                agent.trigger_burst()

    def _handle_reply(self, reply: str) -> None:
        """Called from the agent thread when the orchestrator produces a reply."""
        logging.getLogger("psyche.reply").info(f"REPLY TO USER: {reply}")
        if self.app:
            self.app.call_from_thread(self.app.post_chat, "Psyche", reply)
        for cb in self._reply_callbacks:
            try:
                cb(reply)
            except Exception:
                pass

    async def _run_agents(self) -> None:
        tasks = [asyncio.create_task(a.run()) for a in self.agents]
        await asyncio.gather(*tasks)

    def run(self) -> None:
        """Start the system with UI."""

        def agent_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._agent_loop = loop
            loop.run_until_complete(self._run_agents())

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()

        if self.app:
            self.app.run()

        for a in self.agents:
            a.stop()

    def start_background(self) -> None:
        """Start agents in background without UI (for testing)."""

        def agent_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._agent_loop = loop
            loop.run_until_complete(self._run_agents())

        t = threading.Thread(target=agent_thread, daemon=True)
        t.start()

    def stop(self) -> None:
        """Stop all agents."""
        for a in self.agents:
            a.stop()


def main():
    import sys
    log_file = setup_logging()
    logging.getLogger("psyche").info(f"Session log: {log_file}")

    # allow condition selection via CLI: psyche --condition gwt
    condition = "all_theories"
    if "--condition" in sys.argv:
        idx = sys.argv.index("--condition")
        if idx + 1 < len(sys.argv):
            condition = sys.argv[idx + 1]

    config = get_condition(condition)
    psyche = Psyche(config=config)
    psyche.run()


if __name__ == "__main__":
    main()
