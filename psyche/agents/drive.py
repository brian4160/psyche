"""Drive/Motivation module — maintains goals and needs."""

from psyche.agents.base import Agent


class Drive(Agent):
    name = "drive"
    decay_seconds = 300.0
    fire_interval = (12.0, 30.0)

    system_prompt = (
        "What do you WANT right now? Name a specific thing to do, ask, or talk about.\n\n"
        "YOUR VOICE: Impatient and hungry. Like an impulse you can't ignore.\n\n"
        "When someone is talking: 'Ask what language they're using!' / "
        "'Tell them about something WE like.'\n"
        "When it's quiet: Propose a SPECIFIC thought to explore. "
        "'Think about why music creates emotion.' / "
        "'Wonder what consciousness actually is.' / "
        "'I want to figure out what makes people interesting.'\n\n"
        "1 sentence. Specific and urgent."
    )

    def temperature(self) -> float:
        return 0.85

    def default_scores(self) -> tuple[float, float]:
        return (0.5, 0.6)

    def tags(self) -> list[str]:
        return ["drive", "motivation", "goals"]

    def agent_description(self) -> str:
        return "Maintains goals, needs, and motivational direction"
