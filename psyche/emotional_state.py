"""Persistent emotional state — tracks mood as a running vector."""

from __future__ import annotations

import threading
import time
import re
from dataclasses import dataclass

from psyche.personality import PERSONALITY

# keyword → (valence_shift, arousal_shift, dominance_shift)
EMOTION_KEYWORDS: dict[str, tuple[float, float, float]] = {
    # positive
    "joy": (0.3, 0.2, 0.1),
    "happy": (0.3, 0.1, 0.1),
    "excited": (0.3, 0.4, 0.2),
    "curious": (0.2, 0.2, 0.1),
    "amused": (0.2, 0.1, 0.1),
    "playful": (0.2, 0.2, 0.1),
    "warm": (0.2, 0.0, 0.0),
    "proud": (0.2, 0.1, 0.3),
    "content": (0.2, -0.1, 0.1),
    "fascinated": (0.2, 0.3, 0.0),
    "delighted": (0.3, 0.2, 0.1),
    # negative
    "anxious": (-0.2, 0.3, -0.2),
    "worried": (-0.2, 0.2, -0.1),
    "sad": (-0.3, -0.2, -0.1),
    "frustrated": (-0.2, 0.3, 0.1),
    "irritated": (-0.2, 0.2, 0.1),
    "bored": (-0.1, -0.3, 0.0),
    "confused": (-0.1, 0.1, -0.2),
    "uneasy": (-0.2, 0.2, -0.1),
    "lonely": (-0.3, -0.1, -0.2),
    "overwhelmed": (-0.2, 0.4, -0.3),
    # neutral-ish
    "calm": (0.1, -0.2, 0.1),
    "thoughtful": (0.1, 0.0, 0.1),
    "contemplative": (0.1, -0.1, 0.1),
    "nostalgic": (0.0, 0.0, -0.1),
}


@dataclass
class MoodVector:
    valence: float     # negative ←→ positive (0-1)
    arousal: float     # calm ←→ activated (0-1)
    dominance: float   # submissive ←→ dominant (0-1)

    def describe(self) -> str:
        """Human-readable mood description."""
        parts = []

        # valence
        if self.valence > 0.7:
            parts.append("feeling quite positive")
        elif self.valence > 0.55:
            parts.append("in a good mood")
        elif self.valence > 0.45:
            parts.append("feeling neutral")
        elif self.valence > 0.3:
            parts.append("feeling a bit low")
        else:
            parts.append("feeling down")

        # arousal
        if self.arousal > 0.7:
            parts.append("highly energized")
        elif self.arousal > 0.55:
            parts.append("alert and engaged")
        elif self.arousal > 0.4:
            parts.append("calm but present")
        elif self.arousal > 0.25:
            parts.append("relaxed and quiet")
        else:
            parts.append("drowsy and low-energy")

        # dominance
        if self.dominance > 0.7:
            parts.append("feeling confident and in control")
        elif self.dominance > 0.55:
            parts.append("feeling capable")
        elif self.dominance > 0.4:
            parts.append("feeling balanced")
        elif self.dominance > 0.25:
            parts.append("feeling a bit uncertain")
        else:
            parts.append("feeling small and unsure")

        return "; ".join(parts)


class EmotionalState:
    """Tracks a running mood vector that all agents can read.

    The mood drifts back toward the personality baseline over time
    and gets nudged by emotion module posts.
    """

    def __init__(self):
        t = PERSONALITY["temperament"]
        self._baseline = MoodVector(
            valence=t["baseline_valence"],
            arousal=t["baseline_arousal"],
            dominance=t["baseline_dominance"],
        )
        self._current = MoodVector(
            valence=t["baseline_valence"],
            arousal=t["baseline_arousal"],
            dominance=t["baseline_dominance"],
        )
        self._reactivity = t["emotional_reactivity"]
        self._recovery = t["emotional_recovery"]
        self._last_update = time.time()
        self._lock = threading.Lock()

    def get_mood(self) -> MoodVector:
        """Get current mood, applying time-based decay toward baseline."""
        with self._lock:
            self._decay_toward_baseline()
            return MoodVector(
                valence=self._current.valence,
                arousal=self._current.arousal,
                dominance=self._current.dominance,
            )

    def process_emotion_post(self, content: str) -> None:
        """Update mood based on emotion tags and keywords in a post."""
        content_lower = content.lower()
        with self._lock:
            self._decay_toward_baseline()

            # first try to parse explicit [emotion] tag
            tag_match = re.match(r'\[(\w+)\]', content_lower)
            if tag_match:
                tag = tag_match.group(1)
                if tag in EMOTION_KEYWORDS:
                    dv, da, dd = EMOTION_KEYWORDS[tag]
                    scale = self._reactivity * 1.5  # explicit tags get stronger weight
                    self._current.valence = self._clamp(self._current.valence + dv * scale)
                    self._current.arousal = self._clamp(self._current.arousal + da * scale)
                    self._current.dominance = self._clamp(self._current.dominance + dd * scale)
                    return  # tag is authoritative, skip keyword scan

            # fallback: keyword scan
            for keyword, (dv, da, dd) in EMOTION_KEYWORDS.items():
                if re.search(r'\b' + keyword + r'\b', content_lower):
                    scale = self._reactivity
                    self._current.valence = self._clamp(
                        self._current.valence + dv * scale
                    )
                    self._current.arousal = self._clamp(
                        self._current.arousal + da * scale
                    )
                    self._current.dominance = self._clamp(
                        self._current.dominance + dd * scale
                    )

    def _decay_toward_baseline(self) -> None:
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        # recovery rate: how much we move toward baseline per second
        rate = self._recovery * 0.02  # ~2% per second at max recovery
        factor = min(1.0, rate * elapsed)

        self._current.valence += (self._baseline.valence - self._current.valence) * factor
        self._current.arousal += (self._baseline.arousal - self._current.arousal) * factor
        self._current.dominance += (self._baseline.dominance - self._current.dominance) * factor

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))
