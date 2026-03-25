"""Memory module — recalls relevant past context."""

from psyche.agents.base import Agent


class Memory(Agent):
    name = "memory"
    decay_seconds = 60.0
    fire_interval = (15.0, 35.0)

    system_prompt = (
        "Recall anything relevant from earlier in THIS conversation. "
        "Connect what they just said to something said before.\n\n"
        "YOUR VOICE: Quiet and precise. Like pulling a file from a cabinet.\n\n"
        "Examples: 'Earlier: son in 9th grade. Now: coding. Different topic.' / "
        "'First personal detail they've shared.' / "
        "'Nothing stored yet — conversation just started.'\n\n"
        "Only recall facts from THIS conversation. Never invent memories. 1 sentence."
    )

    def temperature(self) -> float:
        return 0.4

    def default_scores(self) -> tuple[float, float]:
        return (0.3, 0.5)

    def tags(self) -> list[str]:
        return ["memory", "recall"]

    def agent_description(self) -> str:
        return "Retrieves relevant past experiences and knowledge"
