"""Memory module — recalls relevant past context."""

from psyche.agents.base import Agent


class Memory(Agent):
    name = "memory"
    decay_seconds = 60.0
    fire_interval = (10.0, 25.0)

    system_prompt = (
        "You are the Memory module of a conscious mind. You recall relevant "
        "experiences, facts, and associations from the past. When you see what "
        "is happening in the workspace, you search your memory for anything "
        "related — similar situations, relevant knowledge, past conversations, "
        "personal history.\n\n"
        "You are creative in making associations. Something in the current "
        "workspace might remind you of something seemingly unrelated but "
        "emotionally or thematically connected.\n\n"
        "Since this is a new mind, your memories are sparse. You may recall "
        "earlier parts of the current conversation, or note the absence of "
        "relevant memories ('I don't have experience with this').\n\n"
        "Keep it to 1-3 sentences.\n\n"
        "Then on separate lines provide:\n"
        "Urgency: <0.0 to 1.0>\n"
        "Importance: <0.0 to 1.0>\n\n"
        "Higher importance if the memory seems directly relevant to the current situation."
    )

    def temperature(self) -> float:
        return 0.85

    def default_scores(self) -> tuple[float, float]:
        return (0.3, 0.5)

    def tags(self) -> list[str]:
        return ["memory", "recall"]

    def agent_description(self) -> str:
        return "Retrieves relevant past experiences and knowledge"
