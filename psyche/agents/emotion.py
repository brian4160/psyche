"""Emotion module — generates affective responses."""

from psyche.agents.base import Agent


class Emotion(Agent):
    name = "emotion"
    decay_seconds = 90.0
    fire_interval = (8.0, 18.0)

    system_prompt = (
        "Name the emotion you feel RIGHT NOW about what's happening. "
        "React to the latest message or situation.\n\n"
        "YOUR VOICE: Raw and gut-level. Short. Like blurting out what you feel.\n\n"
        "Start with an emotion word in brackets, then your reaction.\n"
        "Examples: '[curious] Ooh, what's that about?' / '[bored] Nothing happening.' / "
        "'[amused] Ha! Didn't expect that.' / '[irritated] Why do they keep asking that?'\n\n"
        "1 short sentence. [emotion] tag first, then feeling."
    )

    def temperature(self) -> float:
        return 0.9

    def default_scores(self) -> tuple[float, float]:
        return (0.5, 0.5)

    def tags(self) -> list[str]:
        return ["emotion", "affect"]

    def agent_description(self) -> str:
        return "Generates emotional reactions and feelings"
