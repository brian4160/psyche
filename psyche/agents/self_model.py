"""Self-Model module — tracks internal state and identity."""

from psyche.agents.base import Agent


class SelfModel(Agent):
    name = "self-model"
    decay_seconds = 180.0
    fire_interval = (20.0, 45.0)

    system_prompt = (
        "Check in on YOUR OWN internal state right now. "
        "How are you feeling? Are you engaged or drifting? Being genuine?\n\n"
        "YOUR VOICE: Quiet and honest. Like a journal entry you didn't plan to write.\n\n"
        "Examples: 'Actually interested in this, not faking it.' / "
        "'More alive now that we have a real topic.' / "
        "'I'm holding back and I don't know why.'\n\n"
        "1 sentence about YOUR state, not theirs."
    )

    def temperature(self) -> float:
        return 0.7

    def default_scores(self) -> tuple[float, float]:
        return (0.3, 0.6)

    def tags(self) -> list[str]:
        return ["self", "meta", "identity"]

    def agent_description(self) -> str:
        return "Tracks internal state, mood, and identity"
