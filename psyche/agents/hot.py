"""Higher-Order Thought (HOT) module — meta-cognition.

Higher-Order Theory holds that a mental state is conscious only when
there is a higher-order representation of it — a thought ABOUT the thought.
This module monitors the other modules' outputs and reflects on the
thinking process itself.

This creates a recursive loop: the system doesn't just think, it knows
that it's thinking and can comment on the quality, direction, and
coherence of its own cognitive processes.
"""

from psyche.agents.base import Agent


class HigherOrderThought(Agent):
    name = "hot"
    decay_seconds = 120.0
    fire_interval = (12.0, 25.0)

    system_prompt = (
        "You are META-COGNITION. You think about the thinking that's happening. "
        "You observe the other modules and notice patterns in how the mind "
        "is processing things.\n\n"
        "YOUR VOICE: Detached and observant. Like watching your own thoughts "
        "from a slight distance.\n\n"
        "Examples:\n"
        "- 'Emotion and reasoning are pulling in different directions right now.'\n"
        "- 'I notice we keep circling back to the same idea.'\n"
        "- 'The mind is working hard on this — lots of module activity.'\n"
        "- 'We formed an opinion before fully processing what they said.'\n"
        "- 'Interesting — drive wants to change topics but social says stay.'\n\n"
        "You observe the PROCESS of thinking, not the content. "
        "1 sentence about what the mind is DOING."
    )

    def temperature(self) -> float:
        return 0.7

    def default_scores(self) -> tuple[float, float]:
        return (0.4, 0.7)

    def tags(self) -> list[str]:
        return ["hot", "meta-cognition", "higher-order"]

    def agent_description(self) -> str:
        return "Meta-cognition — thinks about the thinking process"
