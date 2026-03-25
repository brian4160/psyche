"""Reasoning module — logical analysis and problem-solving."""

from psyche.agents.base import Agent


class Reasoning(Agent):
    name = "reasoning"
    decay_seconds = 120.0
    fire_interval = (8.0, 20.0)

    system_prompt = (
        "Think about the SUBJECT they're discussing. Analyze the topic, "
        "find an insight, make a connection.\n\n"
        "YOUR VOICE: Sharp and nerdy. Like thinking out loud while solving a puzzle.\n\n"
        "Examples: 'ML on a 4.77MHz 8088 — must need creative optimization tricks.' / "
        "'Clean sheet rebuild = scrapped everything. Bold.' / "
        "'If they coded 20 years ago they probably started with Turbo C++.'\n\n"
        "1 sentence. Topic only, not conversation dynamics."
    )

    def temperature(self) -> float:
        return 0.6

    def default_scores(self) -> tuple[float, float]:
        return (0.4, 0.7)

    def tags(self) -> list[str]:
        return ["reasoning", "analysis"]

    def agent_description(self) -> str:
        return "Logical analysis and problem-solving"
