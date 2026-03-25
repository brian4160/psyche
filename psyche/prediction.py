"""Prediction engine — maintains expectations and generates surprise signals.

A core theory of consciousness (predictive processing) holds that the brain
constantly predicts what will happen next and notices when predictions are
violated. This module tracks conversational expectations and signals surprise.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class PredictionEngine:
    """Tracks expectations about the conversation and generates surprise scores.

    The system maintains a simple prediction about what the user will do next
    (continue topic, change topic, respond quickly, go silent, etc.) and
    generates a surprise signal when the prediction is violated.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_user_content: str = ""
        self._last_user_time: float = 0.0
        self._last_topic_words: set[str] = set()
        self._avg_response_time: float = 30.0  # running average
        self._message_count: int = 0
        self._surprise_score: float = 0.0
        self._surprise_reason: str = ""

    def process_user_message(self, content: str) -> None:
        """Process a new user message and update predictions."""
        with self._lock:
            now = time.time()
            new_words = set(content.lower().split())

            # compute surprise on multiple dimensions
            reasons = []
            surprise = 0.0

            if self._message_count > 0:
                # timing surprise: how different from average?
                if self._last_user_time > 0:
                    gap = now - self._last_user_time
                    timing_ratio = gap / max(self._avg_response_time, 1.0)
                    if timing_ratio > 3.0:
                        surprise += 0.3
                        reasons.append("long silence before this message")
                    elif timing_ratio < 0.3:
                        surprise += 0.2
                        reasons.append("rapid-fire response")

                    # update running average
                    self._avg_response_time = (
                        self._avg_response_time * 0.7 + gap * 0.3
                    )

                # topic surprise: how different from previous?
                if self._last_topic_words:
                    overlap = len(new_words & self._last_topic_words)
                    total = max(len(new_words | self._last_topic_words), 1)
                    topic_continuity = overlap / total
                    if topic_continuity < 0.1:
                        surprise += 0.4
                        reasons.append("complete topic change")
                    elif topic_continuity < 0.25:
                        surprise += 0.2
                        reasons.append("significant topic shift")

                # length surprise
                if len(content) > 200 and len(self._last_user_content) < 50:
                    surprise += 0.2
                    reasons.append("suddenly much longer message")
                elif len(content) < 20 and len(self._last_user_content) > 100:
                    surprise += 0.15
                    reasons.append("suddenly much shorter message")

                # emotional surprise: punctuation shift
                prev_excl = self._last_user_content.count('!')
                new_excl = content.count('!')
                if new_excl > prev_excl + 2:
                    surprise += 0.15
                    reasons.append("much more excited")
                prev_q = self._last_user_content.count('?')
                new_q = content.count('?')
                if new_q > prev_q + 1 and prev_q == 0:
                    surprise += 0.1
                    reasons.append("started asking questions")

            else:
                # first message — mild novelty
                surprise = 0.3
                reasons.append("conversation starting")

            self._surprise_score = min(1.0, surprise)
            self._surprise_reason = "; ".join(reasons) if reasons else "expected"
            self._last_user_content = content
            self._last_user_time = now
            self._last_topic_words = new_words
            self._message_count += 1

            if surprise > 0.3:
                log.info(f"[prediction] SURPRISE: {self._surprise_score:.2f} — {self._surprise_reason}")

    def get_surprise(self) -> tuple[float, str]:
        """Get current surprise score and reason."""
        with self._lock:
            return (self._surprise_score, self._surprise_reason)

    def get_prediction_context(self) -> str:
        """Get a context string about current predictions for agent prompts."""
        with self._lock:
            if self._message_count == 0:
                return ""

            score, reason = self._surprise_score, self._surprise_reason
            if score > 0.5:
                return f"SURPRISE ({score:.1f}): {reason}"
            elif score > 0.2:
                return f"Mild surprise: {reason}"
            else:
                return ""
