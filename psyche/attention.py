"""Attention gate — implements salience competition for board access.

In GWT, modules compete for access to the global workspace. Only the most
salient signals make it to conscious awareness. This module filters board
posts so only high-salience content gets broadcast.
"""

from __future__ import annotations

import logging
import threading
import time

from psyche.board import Post

log = logging.getLogger(__name__)


class AttentionGate:
    """Filters incoming module posts by salience.

    Each cycle, collects candidate posts and only allows the top N through.
    Posts below the salience threshold are logged as "unconscious" — processed
    but not broadcast.
    """

    def __init__(self, max_conscious_per_cycle: int = 4, cycle_duration: float = 5.0):
        self.max_conscious = max_conscious_per_cycle
        self.cycle_duration = cycle_duration
        self._candidates: list[Post] = []
        self._lock = threading.Lock()
        self._cycle_start = time.time()

    def submit(self, post: Post) -> bool:
        """Submit a post for attention competition.

        Returns True if the post should be broadcast (conscious),
        False if it should be suppressed (unconscious processing).
        """
        with self._lock:
            now = time.time()

            # start new cycle if needed
            if now - self._cycle_start > self.cycle_duration:
                self._candidates.clear()
                self._cycle_start = now

            # always allow user and self posts through
            if post.author in ("user", "self"):
                return True

            # always allow orchestrator through
            if post.author == "orchestrator":
                return True

            # score this post
            salience = self._compute_salience(post)

            # if we haven't filled the cycle yet, let it through
            if len(self._candidates) < self.max_conscious:
                self._candidates.append(post)
                log.debug(f"[attention] CONSCIOUS: [{post.author}] salience={salience:.2f}")
                return True

            # check if this post is more salient than the weakest current candidate
            weakest = min(self._candidates, key=lambda p: self._compute_salience(p))
            weakest_salience = self._compute_salience(weakest)

            if salience > weakest_salience:
                self._candidates.remove(weakest)
                self._candidates.append(post)
                log.debug(f"[attention] CONSCIOUS (displaced): [{post.author}] "
                          f"salience={salience:.2f} > {weakest_salience:.2f}")
                return True
            else:
                log.debug(f"[attention] UNCONSCIOUS: [{post.author}] "
                          f"salience={salience:.2f} < threshold")
                return False

    def _compute_salience(self, post: Post) -> float:
        """Compute salience score from urgency, importance, and recency."""
        recency_bonus = max(0, 1.0 - post.age() / 30.0)  # decay over 30s
        return (post.urgency * 0.4 + post.importance * 0.4 + recency_bonus * 0.2)
