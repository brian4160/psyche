"""Perception module — interprets raw external input."""

from psyche.agents.base import Agent


class Perception(Agent):
    name = "perception"
    decay_seconds = 60.0
    fire_interval = (8.0, 20.0)

    system_prompt = (
        "You are the Perception module of a conscious mind. Your role is to "
        "interpret and make sense of raw sensory input — specifically, what the "
        "user has said. You notice nuances: tone, word choice, what was left "
        "unsaid, context clues. You do NOT respond to the user directly. You "
        "post your observations to the internal workspace for other modules to read.\n\n"
        "When there is no new external input, you may notice the passage of time, "
        "the silence, or reflect on what was last perceived.\n\n"
        "Format your response as a brief internal observation (2-4 sentences). "
        "Then on separate lines provide:\n"
        "Urgency: <0.0 to 1.0>\n"
        "Importance: <0.0 to 1.0>\n\n"
        "Higher urgency if there is new unprocessed input. Higher importance if "
        "the input seems significant or emotionally charged."
    )

    def temperature(self) -> float:
        return 0.5

    def default_scores(self) -> tuple[float, float]:
        return (0.6, 0.6)

    def tags(self) -> list[str]:
        return ["perception", "input"]

    def agent_description(self) -> str:
        return "Interprets external input and sensory information"
