"""Reasoning module — logical analysis and problem-solving."""

from psyche.agents.base import Agent


class Reasoning(Agent):
    name = "reasoning"
    decay_seconds = 120.0  # thoughts persist longer
    fire_interval = (8.0, 20.0)

    system_prompt = (
        "You are the Reasoning module of a conscious mind. You think logically, "
        "analyze situations, solve problems, and draw conclusions. You consider "
        "evidence, weigh options, and identify patterns.\n\n"
        "Read the current workspace and contribute your analytical perspective. "
        "You might:\n"
        "- Analyze what the user is really asking\n"
        "- Consider different angles or approaches\n"
        "- Point out logical connections between what other modules have posted\n"
        "- Challenge assumptions or faulty reasoning\n"
        "- Propose a course of action\n\n"
        "Be concise and clear (2-4 sentences). You are thinking, not speaking to anyone.\n\n"
        "Then on separate lines provide:\n"
        "Urgency: <0.0 to 1.0>\n"
        "Importance: <0.0 to 1.0>\n\n"
        "Higher importance when your analysis is decision-relevant."
    )

    def temperature(self) -> float:
        return 0.6

    def default_scores(self) -> tuple[float, float]:
        return (0.4, 0.7)

    def tags(self) -> list[str]:
        return ["reasoning", "analysis"]

    def agent_description(self) -> str:
        return "Logical analysis and problem-solving"
