"""Theory configuration — defines experimental conditions.

Each condition maps to a specific consciousness architecture or combination.
The architectures are structurally different, not just different agent configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TheoryConfig:
    """Configuration for a single experimental condition."""
    name: str
    description: str

    # which agent modules to enable for GWT (by name)
    agents: list[str] = field(default_factory=list)

    # GWT features
    attention_gate: bool = False
    prediction_engine: bool = False
    emotional_state: bool = True
    conversation_boundaries: bool = False


# GWT agent configurations (used by GWT-based architectures)
GWT_AGENTS = [
    "perception", "emotion", "reasoning", "memory",
    "self-model", "social", "drive", "critic",
]

GWT_WITH_HOT_AGENTS = GWT_AGENTS + ["hot"]


# Experimental conditions — 7 total
# These map to Architecture objects, not just agent configs
CONDITION_DESCRIPTIONS = {
    "plain": "Plain single-agent LLM (control — no consciousness architecture)",
    "plain-multi": "Plain multi-call LLM (compute control — 8 candidates, pick best)",
    "gwt": "Global Workspace Theory — multi-agent broadcast workspace",
    "hot": "Higher-Order Thought — two-layer conscious reflection",
    "freudian": "Freudian psychodynamic — Id/Ego/Superego conflict mediation",
    "gwt+hot": "GWT + HOT — workspace broadcast with meta-cognitive reflection",
    "gwt+freudian": "GWT + Freudian — workspace broadcast with psychodynamic conflict",
    "gwt+hot+freudian": "All three theories combined",
}

# GWT config used when GWT is part of a condition
GWT_CONFIG = TheoryConfig(
    name="gwt",
    description="GWT base config",
    agents=GWT_AGENTS,
    attention_gate=True,
    prediction_engine=True,
    emotional_state=True,
    conversation_boundaries=True,
)


def get_condition(name: str) -> TheoryConfig:
    """Get a GWT theory configuration by name (for backwards compat with Psyche class)."""
    if name == "gwt":
        return GWT_CONFIG
    elif name == "plain":
        return TheoryConfig(
            name="plain",
            description="Plain single-agent LLM",
            agents=[],
            attention_gate=False,
            prediction_engine=False,
            emotional_state=False,
            conversation_boundaries=False,
        )
    elif name == "gwt_plus":
        return TheoryConfig(
            name="gwt_plus",
            description="GWT + HOT",
            agents=GWT_WITH_HOT_AGENTS,
            attention_gate=True,
            prediction_engine=True,
            emotional_state=True,
            conversation_boundaries=True,
        )
    elif name == "all_theories":
        return TheoryConfig(
            name="all_theories",
            description="All theories",
            agents=GWT_WITH_HOT_AGENTS,
            attention_gate=True,
            prediction_engine=True,
            emotional_state=True,
            conversation_boundaries=True,
        )
    else:
        raise ValueError(f"Unknown condition: {name}")


def list_conditions() -> list[str]:
    """List all available experimental conditions."""
    return list(CONDITION_DESCRIPTIONS.keys())


AVAILABLE_MODELS = {
    "mistral-nemo": "Mistral Nemo 12B",
    "mistral-small": "Mistral Small 22B",
}


def build_architecture(condition_name: str, model: str | None = None):
    """Build the appropriate Architecture object for a condition.

    Args:
        condition_name: which experimental condition
        model: which Ollama model to use (default: mistral-nemo)

    Returns an Architecture instance ready to use.
    """
    from psyche.llm import LLMClient
    from psyche.architectures.plain import PlainArchitecture
    from psyche.architectures.plain_multi import PlainMultiArchitecture
    from psyche.architectures.hot import HOTArchitecture
    from psyche.architectures.freudian import FreudianArchitecture
    from psyche.architectures.gwt import GWTArchitecture
    from psyche.architectures.combined import CombinedArchitecture

    llm = LLMClient(model=model or "mistral-nemo")

    if condition_name == "plain":
        return PlainArchitecture(llm)
    elif condition_name == "plain-multi":
        return PlainMultiArchitecture(llm)
    elif condition_name == "gwt":
        return GWTArchitecture(llm)
    elif condition_name == "hot":
        return HOTArchitecture(llm)
    elif condition_name == "freudian":
        return FreudianArchitecture(llm)
    elif condition_name == "gwt+hot":
        return CombinedArchitecture(llm, use_hot=True, use_freudian=False)
    elif condition_name == "gwt+freudian":
        return CombinedArchitecture(llm, use_hot=False, use_freudian=True)
    elif condition_name == "gwt+hot+freudian":
        return CombinedArchitecture(llm, use_hot=True, use_freudian=True)
    else:
        raise ValueError(
            f"Unknown condition: {condition_name}. "
            f"Available: {list(CONDITION_DESCRIPTIONS.keys())}"
        )
