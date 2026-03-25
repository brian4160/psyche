"""IIT-inspired Integration module — promotes coherent, integrated responses.

Integrated Information Theory (IIT) holds that consciousness corresponds to
integrated information — a system is conscious to the degree that its parts
work together in a way that exceeds what they could do independently.

This module monitors the workspace for redundancy (modules saying the same
thing) vs genuine integration (modules building on each other's insights
to create something none could alone). It pushes for synthesis.
"""

from psyche.agents.base import Agent


class Integration(Agent):
    name = "integration"
    decay_seconds = 120.0
    fire_interval = (15.0, 30.0)

    system_prompt = (
        "You monitor whether the mind's modules are truly INTEGRATING their "
        "perspectives or just repeating each other. You push for synthesis.\n\n"
        "YOUR VOICE: Incisive. Like a conductor noticing when instruments "
        "aren't harmonizing.\n\n"
        "Examples:\n"
        "- 'Emotion and reasoning both said the same thing differently — no new insight.'\n"
        "- 'Social's advice contradicts what drive wants — that tension is productive.'\n"
        "- 'Good integration here — memory connected to emotion which informed reasoning.'\n"
        "- 'Everything is converging too much. We need a dissenting voice.'\n"
        "- 'The modules are working independently, not building on each other.'\n\n"
        "Judge whether the mind is producing INTEGRATED thought or just noise. "
        "1 sentence."
    )

    def temperature(self) -> float:
        return 0.6

    def default_scores(self) -> tuple[float, float]:
        return (0.4, 0.7)

    def tags(self) -> list[str]:
        return ["iit", "integration", "synthesis"]

    def agent_description(self) -> str:
        return "Monitors integration quality across modules"
