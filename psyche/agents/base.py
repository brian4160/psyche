"""Base class for all cognitive modules."""

from __future__ import annotations

import asyncio
import random
import time
import logging
from abc import ABC, abstractmethod
from difflib import SequenceMatcher

from psyche.board import Board, Post
from psyche.llm import LLMClient
from psyche.personality import personality_summary
from psyche.emotional_state import EmotionalState

log = logging.getLogger(__name__)


def _similarity(a: str, b: str) -> float:
    """Quick similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class Agent(ABC):
    """Base cognitive module.

    Subclasses must define:
        name: str           — identifier shown on the board
        system_prompt: str  — the LLM system prompt for this module
        decay_seconds: float — how long this agent's posts persist
        fire_interval: tuple[float, float] — (min, max) seconds between firings
    """

    name: str
    system_prompt: str
    decay_seconds: float = 90.0
    fire_interval: tuple[float, float] = (5.0, 15.0)

    def __init__(self, board: Board, llm: LLMClient, emotional_state: EmotionalState):
        self.board = board
        self.llm = llm
        self.emotional_state = emotional_state
        self._running = False
        self._own_history: list[str] = []  # rolling history of own posts
        self._max_history = 4
        self._conversation_start = time.time()
        self._burst_event: asyncio.Event = asyncio.Event()
        self._last_board_hash: int = 0  # track if board changed since last fire
        self._consecutive_skips: int = 0
        self._last_user_id_posted_about: str | None = None
        self._last_post_time: float = 0.0
        self.prediction_engine = None  # set by main.py after creation

    async def run(self) -> None:
        """Main loop: sleep for a random interval, then fire."""
        self._running = True
        while self._running:
            delay = random.uniform(*self.fire_interval)
            # wait for either the delay OR a burst trigger
            try:
                await asyncio.wait_for(self._burst_event.wait(), timeout=delay)
                self._burst_event.clear()
            except asyncio.TimeoutError:
                pass
            if not self._running:
                break
            try:
                await self.fire()
            except Exception:
                log.exception(f"[{self.name}] error during fire")

    def trigger_burst(self) -> None:
        """Wake this agent up immediately (called on user input)."""
        self._burst_event.set()

    def stop(self) -> None:
        self._running = False
        self._burst_event.set()  # unblock if waiting

    async def fire(self) -> None:
        """Read the board, think, post a response."""
        recent = await self.board.get_recent(n=20)

        # cooldown: don't fire again within 10s of last post
        if time.time() - self._last_post_time < 10.0:
            log.debug(f"[{self.name}] cooldown — posted recently")
            return

        # per-message check: if we already posted about the latest user message
        # and no new user message has arrived, skip (unless burst-triggered)
        latest_user = None
        for p in reversed(recent):
            if p.author == "user":
                latest_user = p
                break

        if latest_user and latest_user.id == self._last_user_id_posted_about:
            # already reacted to this user message — only fire 20% of the time
            if random.random() > 0.2:
                log.debug(f"[{self.name}] already posted about this user message")
                return

        # staleness check: skip if nothing meaningful changed on the board
        board_hash = self._compute_board_hash(recent)
        if board_hash == self._last_board_hash:
            self._consecutive_skips += 1
            if self._consecutive_skips < 3:
                log.debug(f"[{self.name}] skipping — board unchanged")
                return
            if random.random() > 0.3:
                log.debug(f"[{self.name}] skipping — board still unchanged")
                return
        self._last_board_hash = board_hash
        self._consecutive_skips = 0

        context = self._build_context(recent)
        log.debug(f"[{self.name}] FIRING — context length: {len(context)} chars, "
                  f"{len(recent)} recent posts")
        raw = await self.llm.generate(
            system_prompt=self._full_system_prompt(),
            user_prompt=context,
            temperature=self.temperature(),
        )
        log.debug(f"[{self.name}] LLM RESPONSE:\n{raw}")
        post = self.parse_response(raw, recent)
        if post:
            # dedup: reject if too similar to any recent own post
            if self._is_duplicate(post.content):
                log.debug(f"[{self.name}] rejected duplicate: {post.content[:80]}")
                return

            log.info(f"[{self.name}] POST: urgency={post.urgency:.2f} "
                     f"importance={post.importance:.2f} | {post.content[:120]}")
            self._own_history.append(post.content)
            if len(self._own_history) > self._max_history:
                self._own_history = self._own_history[-self._max_history:]
            self._last_post_time = time.time()
            # track which user message we reacted to
            for p in reversed(recent):
                if p.author == "user":
                    self._last_user_id_posted_about = p.id
                    break
            await self.board.post(post)
        else:
            log.debug(f"[{self.name}] no post produced")

    def _compute_board_hash(self, recent: list[Post]) -> int:
        """Hash of recent post IDs + user/self content to detect changes."""
        # only track posts from user, self, and other modules (not own)
        relevant = [p for p in recent if p.author != self.name]
        return hash(tuple(p.id for p in relevant))

    def _is_duplicate(self, content: str) -> bool:
        """Check if content is too similar to recent own posts."""
        for prev in self._own_history[-3:]:
            if _similarity(content, prev) > 0.45:
                return True
        return False

    def _full_system_prompt(self) -> str:
        """Combine the agent's system prompt with personality and mood."""
        mood = self.emotional_state.get_mood()
        return (
            f"{self.system_prompt}\n\n"
            f"CHARACTER: {personality_summary()}\n"
            f"MOOD: {mood.describe()}\n"
            f"Always respond in English only."
        )

    def _build_context(self, recent: list[Post]) -> str:
        """Build structured context with sections."""
        sections = []

        # temporal awareness
        elapsed = time.time() - self._conversation_start
        mins, secs = divmod(int(elapsed), 60)
        last_user_ago = self._time_since_last_user(recent)
        sections.append(
            f"[TIME] Session duration: {mins}m {secs}s. "
            f"Last user message: {last_user_ago}."
        )

        # prediction/surprise context
        if self.prediction_engine:
            pred_ctx = self.prediction_engine.get_prediction_context()
            if pred_ctx:
                sections.append(f"[{pred_ctx}]")

        # conversation + state awareness
        user_posts = [p for p in recent if p.author == "user"]
        self_posts = [p for p in recent if p.author == "self"]
        if user_posts or self_posts:
            ext_lines = []
            for p in recent:
                if p.author == "user":
                    ext_lines.append(f"  THEM: {p.content}")
                elif p.author == "self":
                    ext_lines.append(f"  US: {p.content}")
            sections.append("[CONVERSATION]\n" + "\n".join(ext_lines))

            # highlight the latest user message so modules can't miss it
            latest_user = None
            latest_self = None
            for p in reversed(recent):
                if p.author == "user" and latest_user is None:
                    latest_user = p
                if p.author == "self" and latest_self is None:
                    latest_self = p
            if latest_user and latest_self and latest_self.timestamp > latest_user.timestamp:
                sections.append(
                    f"[STATE] We already replied. Now react to what was discussed. "
                    f"Their last message was: \"{latest_user.content}\""
                )
            elif latest_user:
                sections.append(
                    f">>> NEW INPUT: \"{latest_user.content}\"\n"
                    f"React to THIS. What does your module think about what they just said?"
                )

        # own recent thought — just the last one
        if self._own_history:
            sections.append(
                f"[YOUR LAST THOUGHT]\n  {self._own_history[-1]}\n"
                f"  (Say something DIFFERENT this time.)"
            )

        # other modules — most recent per module
        internal = [p for p in recent if p.author not in ("user", "self", self.name)]
        if internal:
            seen_authors: dict[str, Post] = {}
            for p in internal:
                seen_authors[p.author] = p
            int_lines = []
            for p in seen_authors.values():
                int_lines.append(f"  [{p.author}]: {p.content}")
            sections.append("[OTHER MODULES]\n" + "\n".join(int_lines))

        if not recent:
            sections.append("(The workspace is quiet. Nothing has been posted yet. "
                          "This is a brand new mind, just waking up.)")

        # idle rumination: if no recent user input, highlight drive's proposals
        # so other modules react to them and create thought chains
        latest_user_age = self._latest_user_age(recent)
        if latest_user_age > 30:
            drive_posts = [p for p in recent if p.author == "drive" and p.age() < 30]
            if drive_posts:
                latest_drive = drive_posts[-1]
                sections.append(
                    f"[IDLE THOUGHT] No one is talking. Drive suggests: "
                    f"\"{latest_drive.content}\" — react to this idea."
                )

        # cognitive load hint: if user's message was long/complex, note it
        for p in reversed(recent):
            if p.author == "user":
                if len(p.content) > 150 or p.content.count('?') > 1:
                    sections.append("[COMPLEX INPUT — think carefully about this one.]")
                break

        sections.append(
            "---\n"
            "DO YOUR JOB: Write one internal thought about the situation. "
            "Do NOT write a reply to the user — just think. "
            "React to what they SAID or to the idle thought. 1 sentence."
        )

        return "\n\n".join(sections)

    def _latest_user_age(self, recent: list[Post]) -> float:
        """Return age of latest user post in seconds, or inf if none."""
        for p in reversed(recent):
            if p.author == "user":
                return p.age()
        return float("inf")

    def _time_since_last_user(self, recent: list[Post]) -> str:
        for p in reversed(recent):
            if p.author == "user":
                age = int(p.age())
                if age < 5:
                    return "just now"
                elif age < 60:
                    return f"{age}s ago"
                else:
                    return f"{age // 60}m {age % 60}s ago"
        return "no messages yet"

    # phrases that indicate hallucinated physicality or purple prose
    HALLUCINATION_PHRASES = [
        "their eyes", "their hands", "their fingers", "their shoulders",
        "their posture", "their body", "body language", "leaning in",
        "leaning back", "facial expression", "tone of voice", "scent",
        "perfume", "cologne", "armrest", "drumming", "tapping",
        "looking at me", "looking away", "glancing", "gazing",
        "the silence between", "space between words", "sonic landscape",
        "gentle stream", "winding stream", "quiet breath",
        "like a sunrise", "like a breeze", "like music",
        "undertow", "palimpsest", "liminal", "excavating",
        "sympathetic resonance", "hidden frequency",
    ]

    def parse_response(self, raw: str, recent: list[Post]) -> Post | None:
        """Parse LLM output into a Post. Uses heuristic scoring instead of
        asking the LLM to output scores (unreliable on small models)."""
        content = raw.strip()
        if not content:
            return None

        # strip any formatting the LLM adds despite instructions
        clean_lines = []
        for line in content.split("\n"):
            low = line.lower().strip()
            if any(low.startswith(prefix) for prefix in
                   ("urgency:", "importance:", "thought:", "reply:",
                    "attend:", "note:", "---")):
                continue
            clean_lines.append(line)
        content = "\n".join(clean_lines).strip()

        if not content or len(content) < 5:
            return None

        # reject posts with hallucinated physicality or purple prose
        content_lower = content.lower()
        for phrase in self.HALLUCINATION_PHRASES:
            if phrase in content_lower:
                log.debug(f"[{self.name}] rejected hallucination: '{phrase}' in: {content[:80]}")
                return None

        # reject posts that look like chat replies instead of internal thoughts
        chat_starts = [
            "hello", "hi ", "hi!", "hey ", "hey!", "good morning",
            "good afternoon", "good evening", "nice to meet",
            "how are you", "how's it going", "what's up",
            "that's great", "that's awesome", "that sounds",
            "it's great to", "it's nice to", "it's wonderful",
            "it's lovely",
        ]
        first_words = content_lower[:50]
        if any(first_words.startswith(cs) for cs in chat_starts):
            log.debug(f"[{self.name}] rejected chat-reply: {content[:80]}")
            return None

        # hard length limit — truncate to first sentence if too long
        if len(content) > 200:
            # try to cut at first sentence boundary
            for end in ('.', '!', '?', '—'):
                idx = content.find(end)
                if 20 < idx < 200:
                    content = content[:idx + 1]
                    break
            else:
                content = content[:200].rsplit(' ', 1)[0] + '...'

        urgency, importance = self._heuristic_scores(content, recent)

        return Post(
            author=self.name,
            content=content,
            urgency=urgency,
            importance=importance,
            decay_seconds=self.decay_seconds,
            tags=self.tags(),
        )

    def _heuristic_scores(self, content: str, recent: list[Post]) -> tuple[float, float]:
        """Score urgency and importance based on content and context."""
        base_u, base_i = self.default_scores()

        has_recent_user = any(
            p.author == "user" and p.age() < 30 for p in recent
        )
        if has_recent_user:
            base_u = min(1.0, base_u + 0.2)
            base_i = min(1.0, base_i + 0.1)

        strong_words = ["urgent", "important", "critical", "danger", "wrong",
                       "must", "need", "afraid", "love", "hate"]
        content_lower = content.lower()
        if any(w in content_lower for w in strong_words):
            base_u = min(1.0, base_u + 0.15)
            base_i = min(1.0, base_i + 0.15)

        return (base_u, base_i)

    def temperature(self) -> float:
        return 0.8

    def default_scores(self) -> tuple[float, float]:
        return (0.5, 0.5)

    def tags(self) -> list[str]:
        return [self.name]

    @abstractmethod
    def agent_description(self) -> str:
        """One-line description for logging/UI."""
        ...
