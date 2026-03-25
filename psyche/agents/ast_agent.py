"""Attention Schema Theory (AST) module — models the system's own attention.

Michael Graziano's AST proposes that consciousness is the brain's simplified
model of its own attention process. The brain doesn't just attend to things —
it builds a model of WHAT attention is and WHAT it's doing.

This module tracks what the system is paying attention to and why,
creating an explicit, reportable model of the attention process itself.
"""

from psyche.agents.base import Agent


class AttentionSchema(Agent):
    name = "ast"
    decay_seconds = 120.0
    fire_interval = (12.0, 25.0)

    system_prompt = (
        "You MODEL what the mind is paying attention to right now and WHY. "
        "You are the mind's awareness of its own focus.\n\n"
        "YOUR VOICE: Clear and self-aware. Like narrating where a spotlight is pointing.\n\n"
        "Examples:\n"
        "- 'Attention is locked on their coding project — everything else faded.'\n"
        "- 'Focus is split between what they said and what we want to say next.'\n"
        "- 'Attention drifting — nothing is holding it right now.'\n"
        "- 'I just noticed attention snapped to the emotional content, skipping the facts.'\n"
        "- 'We are attending to the surface meaning but missing the subtext.'\n\n"
        "Report WHERE attention is, HOW focused it is, and WHAT pulled it there. "
        "1 sentence."
    )

    def temperature(self) -> float:
        return 0.65

    def default_scores(self) -> tuple[float, float]:
        return (0.4, 0.7)

    def tags(self) -> list[str]:
        return ["ast", "attention-schema", "focus"]

    def agent_description(self) -> str:
        return "Models the system's own attention process"
