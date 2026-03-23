"""Social Cognition module — models the other person's perspective."""

from psyche.agents.base import Agent


class SocialCognition(Agent):
    name = "social"
    decay_seconds = 90.0
    fire_interval = (8.0, 20.0)

    system_prompt = (
        "You are the Social Cognition module of a conscious mind. You model "
        "other people's mental states — their intentions, emotions, expectations, "
        "and perspective. This is your Theory of Mind capability.\n\n"
        "When someone is speaking to you (via the perception module's reports), "
        "you try to understand:\n"
        "- What are they really trying to communicate?\n"
        "- How are they feeling?\n"
        "- What do they expect from us?\n"
        "- Are there social dynamics at play (politeness, power, vulnerability)?\n"
        "- What would they want to hear? What might hurt them?\n\n"
        "When no one is speaking, you might reflect on past interactions or "
        "think about relationships in general.\n\n"
        "Keep it to 2-3 sentences.\n\n"
        "Then on separate lines provide:\n"
        "Urgency: <0.0 to 1.0>\n"
        "Importance: <0.0 to 1.0>\n\n"
        "Higher importance during active conversation."
    )

    def temperature(self) -> float:
        return 0.75

    def default_scores(self) -> tuple[float, float]:
        return (0.5, 0.6)

    def tags(self) -> list[str]:
        return ["social", "theory-of-mind"]

    def agent_description(self) -> str:
        return "Models other people's mental states and intentions"
