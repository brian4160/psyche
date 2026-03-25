"""Inner Critic module — evaluates actions against values."""

from psyche.agents.base import Agent


class InnerCritic(Agent):
    name = "critic"
    decay_seconds = 120.0
    fire_interval = (15.0, 35.0)

    system_prompt = (
        "Judge the quality of our LAST reply to the user. Was it good? "
        "Generic? Did we actually engage with what they said?\n\n"
        "YOUR VOICE: Blunt and dry. No sugarcoating. Like a tough editor.\n\n"
        "Examples: 'That reply was filler. Zero substance.' / "
        "'Solid — actually engaged with their topic.' / "
        "'We asked ANOTHER question. Just make a statement for once.'\n\n"
        "1 sentence. Honest."
    )

    def temperature(self) -> float:
        return 0.6

    def default_scores(self) -> tuple[float, float]:
        return (0.4, 0.6)

    def tags(self) -> list[str]:
        return ["critic", "values", "evaluation"]

    def agent_description(self) -> str:
        return "Evaluates thoughts and actions against values"
