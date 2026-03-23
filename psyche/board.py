"""Global Workspace — the shared message board all agents read from and post to."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class PostStatus(Enum):
    ACTIVE = "active"
    DECAYED = "decayed"
    ATTENDED = "attended"  # orchestrator chose to sustain this


@dataclass
class Post:
    author: str
    content: str
    urgency: float  # 0-1
    importance: float  # 0-1
    tags: list[str] = field(default_factory=list)
    in_response_to: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    status: PostStatus = PostStatus.ACTIVE
    decay_seconds: float = 90.0  # default; overridden per agent type
    attended_at: float | None = None

    def age(self) -> float:
        return time.time() - self.timestamp

    def is_expired(self) -> bool:
        if self.status == PostStatus.ATTENDED:
            ref = self.attended_at or self.timestamp
            return (time.time() - ref) > self.decay_seconds
        return self.age() > self.decay_seconds

    def as_board_text(self) -> str:
        age = int(self.age())
        status_marker = ""
        if self.status == PostStatus.ATTENDED:
            status_marker = " [ATTENDED]"
        elif self.status == PostStatus.DECAYED:
            status_marker = " [fading]"
        return (
            f"[{self.author}] (id={self.id}, {age}s ago, "
            f"urgency={self.urgency:.1f}, importance={self.importance:.1f})"
            f"{status_marker}\n{self.content}"
        )


class Board:
    """Thread-safe global workspace. Uses threading primitives so it works
    across the UI thread and the agent thread."""

    def __init__(self, max_history: int = 200):
        self._posts: list[Post] = []
        self._lock = threading.Lock()
        self._max_history = max_history
        self._subscribers: list[Callable[[Post], None]] = []

    async def post(self, p: Post) -> None:
        self.post_sync(p)

    def post_sync(self, p: Post) -> None:
        with self._lock:
            self._posts.append(p)
            if len(self._posts) > self._max_history:
                self._posts = self._posts[-self._max_history:]
        # notify subscribers (outside lock)
        for cb in self._subscribers:
            try:
                cb(p)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[Post], None]) -> None:
        """Register a callback that fires on every new post."""
        self._subscribers.append(callback)

    async def get_recent(self, n: int = 20, exclude_decayed: bool = True) -> list[Post]:
        with self._lock:
            self._decay_pass()
            posts = self._posts
            if exclude_decayed:
                posts = [p for p in posts if p.status != PostStatus.DECAYED]
            return posts[-n:]

    async def get_by_author(self, author: str, n: int = 5) -> list[Post]:
        with self._lock:
            return [p for p in self._posts if p.author == author][-n:]

    async def attend(self, post_id: str) -> None:
        """Orchestrator attends to a post, resetting its decay timer."""
        with self._lock:
            for p in self._posts:
                if p.id == post_id:
                    p.status = PostStatus.ATTENDED
                    p.attended_at = time.time()
                    break

    def _decay_pass(self) -> None:
        """Mark expired posts as decayed."""
        for p in self._posts:
            if p.status != PostStatus.DECAYED and p.is_expired():
                p.status = PostStatus.DECAYED
