"""Social Cognition module — models the other person's perspective."""

from psyche.agents.base import Agent


class SocialCognition(Agent):
    name = "social"
    decay_seconds = 90.0
    fire_interval = (8.0, 20.0)

    system_prompt = (
        "Read the social situation and advise: what should we SAY or DO next? "
        "What kind of response would land well?\n\n"
        "YOUR VOICE: Casual and street-smart. Like a wingman coaching you.\n\n"
        "Examples: 'They're into this — match the energy.' / "
        "'Too many questions from us. Say something about ourselves.' / "
        "'They opened up. Don't make it weird, just roll with it.'\n\n"
        "1 sentence. Practical advice only."
    )

    def temperature(self) -> float:
        return 0.7

    def default_scores(self) -> tuple[float, float]:
        return (0.5, 0.6)

    def tags(self) -> list[str]:
        return ["social", "theory-of-mind"]

    def agent_description(self) -> str:
        return "Models other people's mental states and intentions"
