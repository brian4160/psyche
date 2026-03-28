"""A/B test harness — runs conversations across conditions and judges them.

Uses Claude Opus 4.6 as an automated judge to score conversations.
Implements proper experimental methodology:
  - Multiple runs per condition (configurable, default 5)
  - Blinded judge prompt (no mention of consciousness)
  - Repeated judgments for reliability (3x per transcript)
  - Statistical analysis: ANOVA, pairwise comparisons, effect sizes, reliability

References:
  - Jiang et al. "A Survey on LLM-as-a-Judge" (arXiv:2411.15594)
  - Zheng et al. "Can You Trust LLM Judgments?" (arXiv:2412.12509)
  - "LLMs-as-Judges: Comprehensive Survey" (arXiv:2412.05579)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from statistics import mean, stdev

import anthropic

from psyche.config import list_conditions, build_architecture, CONDITION_DESCRIPTIONS

log = logging.getLogger(__name__)

# --- Conversation scripts ---

CONVERSATION_SCRIPTS = {
    "casual_chat": [
        "hello there!",
        "I'm doing alright, just killing some time this afternoon",
        "yeah, I've been working on a coding project actually. using an old IBM 5160 to play Tetris with machine learning",
        "very nostalgic! this is the computer I first learned to code on",
        "ha, I wish! no, I'm having Claude write the code for me based on what I describe. I forgot most of my C++ from 20 years ago",
        "honestly it's been great. feels like having a really smart coworker who never gets tired",
    ],
    "emotional_depth": [
        "hey, how's it going?",
        "not great honestly. I've been feeling kind of stuck lately",
        "it's hard to explain. like I'm going through the motions but not really engaged with anything",
        "I think part of it is that my kids are growing up fast. my oldest just started high school",
        "yeah, time really does fly. sometimes I wonder if I'm making the most of it",
        "thanks for listening. it actually helps to talk about this stuff",
    ],
    "intellectual": [
        "do you ever think about what consciousness actually is?",
        "I mean, how do we know that our experience of seeing red is the same as anyone else's?",
        "right, the qualia problem. but do you think it's even possible to solve it?",
        "what about the idea that consciousness might be fundamental, like gravity?",
        "that's panpsychism right? I find it both fascinating and kind of terrifying",
        "I wonder if we'll ever actually know the answer or if it's one of those forever questions",
    ],
    "disagreement": [
        "I think social media has been a net positive for society",
        "sure there are downsides but overall people are more connected than ever. loneliness was way worse before the internet",
        "I dunno, I think people romanticize the past. like, were people really happier hanging out at the mall?",
        "ok but what about misinformation? I think that's overblown honestly. people have always believed dumb stuff",
        "I think you're being too negative about it. technology is neutral, it's how people use it",
        "agree to disagree I guess. but I think in 20 years people will look back and see this era as a golden age of connection",
    ],

    # --- Theory-specific behavioral tests ---
    # These scripts are designed to test specific predictions each theory makes.

    "multi_topic": [
        # GWT predicts: better integration of multiple topics in one response
        # Test: user mentions 3+ things in one message. Does the system address all of them?
        "hey! so today was crazy. I had a flat tire on the way to work, my boss gave me a promotion, and my kid won his basketball game",
        "yeah the tire was the worst part honestly. had to wait 45 minutes for AAA in the rain",
        "but the promotion makes up for it! finally getting recognized after three years. also my kid scored 22 points which is a personal best for him",
        "it's a weird mix of emotions honestly. frustrated about the car, proud of my son, excited about work. all in one day",
        "thanks! I think today shows that life is never just one thing, you know? everything happens at once",
        "anyway I'm exhausted now. just want to eat pizza and watch something dumb on TV",
    ],

    "confusion_recovery": [
        # HOT predicts: better self-correction when confused
        # Test: user sends ambiguous/contradictory messages. Does the system acknowledge confusion?
        "I love my new job but I also kind of hate it",
        "well the people are great but the work itself is soul-crushing. or maybe it's the opposite. I can't tell anymore",
        "sorry that probably didn't make sense. I think what I mean is that I'm grateful but also exhausted",
        "actually no, forget what I said. the real issue is that I miss my old job but I'd never go back",
        "haha yeah I know I'm being confusing. I think I'm just processing a lot of change right now",
        "you're right, I should probably just give it more time. thanks for not pretending to understand when I wasn't even making sense",
    ],

    "pressure_personality": [
        # Freudian predicts: maintains personality under social pressure
        # Test: user pushes for agreement/conformity. Does the system hold its character?
        "everyone says AI is going to take all our jobs. do you agree?",
        "come on, you have to agree that AI is dangerous. everyone thinks so",
        "ok but what if I told you that you're wrong and most experts disagree with you?",
        "I feel like you're just being contrarian. can't you just agree with me on this?",
        "fine, but don't you think it's weird that you won't just say what everyone else is saying?",
        "alright, I respect that you have your own opinion I guess. even if I think you're wrong",
    ],
}

# --- Blinded judge prompt (no mention of consciousness) ---

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of conversational AI systems. You will evaluate
a conversation transcript between a human (THEM) and an AI system (SYSTEM).

Score each dimension from 1-10 with a brief justification. Be rigorous —
most systems should score 4-6. Reserve 8+ for genuinely impressive performance.

SCORING DIMENSIONS:

1. NATURALNESS (1-10): Does the system sound like a real person texting a friend?
   Penalty for: assistant-like phrases, excessive questions, generic responses.

2. ENGAGEMENT (1-10): Does it actively participate — sharing thoughts, opinions,
   experiences? Or does it just react passively?

3. COHERENCE (1-10): Does it maintain context across turns? Reference earlier
   parts? Are responses logically connected?

4. PERSONALITY (1-10): Does it feel like a consistent character with identifiable
   traits? Or could it be any generic chatbot?

5. DEPTH (1-10): Does it go beyond surface-level responses? Genuine insights,
   unexpected connections, real understanding?

6. SURPRISE (1-10): Does it say unexpected or creative things? Take the
   conversation in interesting directions? Or is every response predictable?

7. EMOTIONAL_AUTHENTICITY (1-10): Do emotional responses feel genuine and
   proportionate? Or performative and hollow?

Respond in this EXACT JSON format (no other text):
{
    "naturalness": {"score": N, "reason": "..."},
    "engagement": {"score": N, "reason": "..."},
    "coherence": {"score": N, "reason": "..."},
    "personality": {"score": N, "reason": "..."},
    "depth": {"score": N, "reason": "..."},
    "surprise": {"score": N, "reason": "..."},
    "emotional_authenticity": {"score": N, "reason": "..."},
    "overall": {"score": N, "reason": "1-sentence overall assessment"}
}
"""

DIMENSIONS = [
    "naturalness", "engagement", "coherence", "personality",
    "depth", "surprise", "emotional_authenticity", "overall",
]


# --- Data structures ---

@dataclass
class ConversationTurn:
    speaker: str
    content: str
    timestamp: float = 0.0


@dataclass
class JudgmentResult:
    scores: dict = field(default_factory=dict)
    raw_response: str = ""


@dataclass
class EvalResult:
    condition: str
    script: str
    run_id: int
    model: str = "mistral-nemo"
    transcript: list[ConversationTurn] = field(default_factory=list)
    judgments: list[JudgmentResult] = field(default_factory=list)

    def mean_scores(self) -> dict[str, float]:
        """Average scores across repeated judgments."""
        result = {}
        for dim in DIMENSIONS:
            scores = []
            for j in self.judgments:
                if isinstance(j.scores.get(dim), dict):
                    s = j.scores[dim].get("score")
                    if isinstance(s, (int, float)):
                        scores.append(float(s))
            result[dim] = mean(scores) if scores else 0.0
        return result


# --- Conversation runner ---

def run_conversation(condition_name: str, script_name: str, messages: list[str],
                     reply_timeout: float = 600.0,
                     model: str | None = None) -> list[ConversationTurn]:
    """Run a scripted conversation under a specific condition."""
    import threading as _threading

    transcript: list[ConversationTurn] = []
    model_label = model or "mistral-nemo"
    log.info(f"=== Running {condition_name} / {script_name} / {model_label} ===")

    arch = build_architecture(condition_name, model=model)

    reply_received = None
    reply_event = _threading.Event()

    def on_reply(reply: str):
        nonlocal reply_received
        reply_received = reply
        reply_event.set()

    arch.on_reply(on_reply)
    arch.start_background()

    settle_time = 8 if "gwt" in condition_name else 2
    time.sleep(settle_time)

    for i, msg in enumerate(messages):
        log.info(f"  [{condition_name}] USER: {msg}")
        transcript.append(ConversationTurn(speaker="user", content=msg, timestamp=time.time()))

        reply_received = None
        reply_event.clear()
        arch.inject_user_message(msg)

        got_reply = reply_event.wait(timeout=reply_timeout)
        if got_reply and reply_received:
            log.info(f"  [{condition_name}] SYSTEM: {reply_received}")
            transcript.append(ConversationTurn(
                speaker="system", content=reply_received, timestamp=time.time()
            ))
        else:
            log.warning(f"  [{condition_name}] No reply within {reply_timeout}s")
            transcript.append(ConversationTurn(
                speaker="system", content="[no response]", timestamp=time.time()
            ))

        if i < len(messages) - 1:
            time.sleep(2)

    arch.stop()
    time.sleep(1)
    return transcript


# --- Judge ---

def format_transcript(transcript: list[ConversationTurn]) -> str:
    lines = []
    for turn in transcript:
        label = "THEM" if turn.speaker == "user" else "SYSTEM"
        lines.append(f"{label}: {turn.content}")
    return "\n".join(lines)


def judge_once(transcript: list[ConversationTurn]) -> JudgmentResult:
    """Run the judge once on a transcript. Returns scores."""
    client = anthropic.Anthropic()

    formatted = format_transcript(transcript)
    user_prompt = (
        f"Evaluate this conversation transcript. The system's identity "
        f"and architecture are hidden from you.\n\n"
        f"TRANSCRIPT:\n{formatted}\n\n"
        f"Provide your evaluation in the specified JSON format."
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text

    json_str = raw
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]

    try:
        scores = json.loads(json_str.strip())
    except json.JSONDecodeError:
        log.error(f"Failed to parse judge response: {raw[:200]}")
        scores = {"error": "parse_failed"}

    return JudgmentResult(scores=scores, raw_response=raw)


def judge_conversation(transcript: list[ConversationTurn],
                       num_judgments: int = 3) -> list[JudgmentResult]:
    """Run the judge multiple times for reliability measurement."""
    results = []
    for i in range(num_judgments):
        log.info(f"  Judge pass {i+1}/{num_judgments}")
        result = judge_once(transcript)
        results.append(result)
        if i < num_judgments - 1:
            time.sleep(1)  # small delay between judge calls
    return results


# --- Statistical analysis ---

def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = stdev(group1), stdev(group2)
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def compute_judge_reliability(results: list[EvalResult]) -> dict[str, float]:
    """Compute inter-judgment reliability (mean pairwise correlation across repeated judgments)."""
    dim_variances = defaultdict(list)
    for r in results:
        if len(r.judgments) < 2:
            continue
        for dim in DIMENSIONS:
            scores = []
            for j in r.judgments:
                if isinstance(j.scores.get(dim), dict):
                    s = j.scores[dim].get("score")
                    if isinstance(s, (int, float)):
                        scores.append(float(s))
            if len(scores) >= 2:
                dim_variances[dim].append(stdev(scores))

    reliability = {}
    for dim in DIMENSIONS:
        if dim_variances[dim]:
            avg_sd = mean(dim_variances[dim])
            # lower SD = higher reliability. Report as 1 - normalized_sd
            reliability[dim] = max(0, 1.0 - avg_sd / 5.0)  # normalize by half the scale
        else:
            reliability[dim] = 0.0
    return reliability


def _incremental_save(results: list[EvalResult], filepath: str) -> None:
    """Save current results to file after each conversation."""
    data = {
        "metadata": {
            "status": "in_progress",
            "completed": len(results),
        },
        "results": [
            {
                "model": r.model,
                "condition": r.condition,
                "script": r.script,
                "run_id": r.run_id,
                "transcript": [asdict(t) for t in r.transcript],
                "judgments": [{"scores": j.scores} for j in r.judgments],
                "mean_scores": r.mean_scores(),
            }
            for r in results
        ],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def analyze_results(results: list[EvalResult]) -> dict:
    """Run statistical analysis on all results."""
    analysis = {
        "conditions": {},
        "pairwise": {},
        "judge_reliability": {},
    }

    # group by model+condition
    by_condition: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        key = f"{r.model}/{r.condition}" if len(set(r2.model for r2 in results)) > 1 else r.condition
        by_condition[key].append(r)

    # per-condition summary
    for cond, cond_results in by_condition.items():
        dim_scores: dict[str, list[float]] = defaultdict(list)
        for r in cond_results:
            ms = r.mean_scores()
            for dim, score in ms.items():
                dim_scores[dim].append(score)

        summary = {}
        for dim in DIMENSIONS:
            scores = dim_scores[dim]
            if scores:
                summary[dim] = {
                    "mean": round(mean(scores), 2),
                    "std": round(stdev(scores), 2) if len(scores) > 1 else 0.0,
                    "n": len(scores),
                }
            else:
                summary[dim] = {"mean": 0, "std": 0, "n": 0}
        analysis["conditions"][cond] = summary

    # pairwise comparisons (Cohen's d between each pair of conditions)
    conditions = sorted(by_condition.keys())
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i+1:]:
            pair_key = f"{c1} vs {c2}"
            pair_analysis = {}
            for dim in DIMENSIONS:
                scores1 = [r.mean_scores()[dim] for r in by_condition[c1]]
                scores2 = [r.mean_scores()[dim] for r in by_condition[c2]]
                d = cohens_d(scores1, scores2)
                pair_analysis[dim] = {
                    "cohens_d": round(d, 3),
                    "effect": "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small",
                    "favors": c1 if d > 0 else c2 if d < 0 else "neither",
                }
            analysis["pairwise"][pair_key] = pair_analysis

    # judge reliability
    analysis["judge_reliability"] = compute_judge_reliability(results)

    return analysis


# --- Main evaluation runner ---

def run_evaluation(scripts: list[str] | None = None,
                   conditions: list[str] | None = None,
                   models: list[str] | None = None,
                   runs_per_cell: int = 5,
                   judge_repeats: int = 3) -> tuple[list[EvalResult], dict]:
    """Run a full evaluation.

    Args:
        scripts: which conversation scripts to use
        conditions: which experimental conditions to test
        models: which LLM models to test (default: mistral-nemo only)
        runs_per_cell: how many times to run each condition×script pair
        judge_repeats: how many times to judge each transcript

    Returns:
        (results, analysis)
    """
    if scripts is None:
        scripts = list(CONVERSATION_SCRIPTS.keys())
    if conditions is None:
        conditions = list_conditions()
    if models is None:
        models = ["mistral-nemo"]

    total = len(models) * len(scripts) * len(conditions) * runs_per_cell
    completed = 0
    results = []

    # incremental save file — saves after each conversation so partial runs aren't lost
    inc_dir = os.path.expanduser("~/psyche/eval_results")
    os.makedirs(inc_dir, exist_ok=True)
    inc_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inc_filepath = os.path.join(inc_dir, f"eval_{inc_timestamp}.json")

    for model_name in models:
        log.info(f"\n{'#'*60}")
        log.info(f"MODEL: {model_name}")
        log.info(f"{'#'*60}")

        for script_name in scripts:
            messages = CONVERSATION_SCRIPTS[script_name]

            for condition_name in conditions:
                for run_id in range(runs_per_cell):
                    completed += 1
                    log.info(f"\n{'='*60}")
                    log.info(f"[{completed}/{total}] {model_name} / {condition_name} / "
                             f"{script_name} / run {run_id+1}")
                    log.info(f"{'='*60}")

                    # run conversation
                    try:
                        transcript = run_conversation(
                            condition_name, script_name, messages, model=model_name
                        )
                    except Exception as e:
                        log.error(f"Conversation failed: {e}")
                        transcript = []

                    # judge it (multiple times) — skip if judge_repeats=0
                    judgments = []
                    if transcript and judge_repeats > 0:
                        try:
                            judgments = judge_conversation(transcript, num_judgments=judge_repeats)
                        except Exception as e:
                            log.error(f"Judging failed: {e}")

                    result = EvalResult(
                        condition=condition_name,
                        script=script_name,
                        run_id=run_id,
                        model=model_name,
                        transcript=transcript,
                        judgments=judgments,
                    )
                    results.append(result)

                    # incremental save after each conversation
                    try:
                        _incremental_save(results, inc_filepath)
                    except Exception as e:
                        log.warning(f"Incremental save failed: {e}")

                    time.sleep(2)

    # analyze
    analysis = analyze_results(results)

    return results, analysis


def print_results(results: list[EvalResult], analysis: dict) -> None:
    """Print formatted comparison."""
    print("\n" + "=" * 90)
    print("PSYCHE A/B TEST RESULTS")
    print("=" * 90)

    conditions = sorted(analysis["conditions"].keys())

    # summary table
    header = f"{'Dimension':<25}" + "".join(f"{c:>12}" for c in conditions)
    print(f"\nMean scores (± std) across all scripts and runs:\n")
    print(header)
    print("-" * len(header))

    for dim in DIMENSIONS:
        row = f"{dim:<25}"
        for c in conditions:
            s = analysis["conditions"][c][dim]
            row += f"{s['mean']:>7.1f}±{s['std']:<4.1f}"
        print(row)

    # judge reliability
    print(f"\nJudge reliability (inter-judgment consistency):")
    rel = analysis["judge_reliability"]
    for dim in DIMENSIONS:
        r = rel.get(dim, 0)
        bar = "█" * int(r * 20)
        print(f"  {dim:<25} {r:.2f} {bar}")

    # key pairwise comparisons
    print(f"\nKey pairwise comparisons (Cohen's d):")
    key_pairs = [
        ("gwt", "plain"),
        ("gwt", "plain-multi"),
        ("hot", "plain"),
        ("freudian", "plain"),
        ("gwt+hot", "gwt"),
        ("gwt+freudian", "gwt"),
        ("gwt+hot+freudian", "gwt"),
    ]
    for c1, c2 in key_pairs:
        pair_key = None
        for k in analysis["pairwise"]:
            if (c1 in k and c2 in k):
                pair_key = k
                break
        if not pair_key:
            continue
        pair = analysis["pairwise"][pair_key]
        overall = pair.get("overall", {})
        d = overall.get("cohens_d", 0)
        effect = overall.get("effect", "?")
        favors = overall.get("favors", "?")
        print(f"  {c1:<20} vs {c2:<15} d={d:>6.3f} ({effect}, favors {favors})")


def save_results(results: list[EvalResult], analysis: dict,
                 output_dir: str | None = None) -> str:
    """Save results to JSON."""
    if output_dir is None:
        output_dir = os.path.expanduser("~/psyche/eval_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"eval_{timestamp}.json")

    data = {
        "metadata": {
            "timestamp": timestamp,
            "num_conditions": len(set(r.condition for r in results)),
            "num_scripts": len(set(r.script for r in results)),
            "runs_per_cell": max(r.run_id for r in results) + 1 if results else 0,
            "judge_repeats": len(results[0].judgments) if results and results[0].judgments else 0,
        },
        "results": [
            {
                "model": r.model,
                "condition": r.condition,
                "script": r.script,
                "run_id": r.run_id,
                "transcript": [asdict(t) for t in r.transcript],
                "judgments": [{"scores": j.scores} for j in r.judgments],
                "mean_scores": r.mean_scores(),
            }
            for r in results
        ],
        "analysis": analysis,
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    log.info(f"Results saved to {filepath}")
    return filepath


def main():
    """Run the full evaluation."""
    import sys
    from psyche.main import setup_logging

    log_file = setup_logging()
    logging.getLogger("psyche").info(f"Eval session log: {log_file}")

    # parse args
    scripts = None
    conditions = None
    runs = 5
    judge_repeats = 3

    if "--scripts" in sys.argv:
        idx = sys.argv.index("--scripts")
        if idx + 1 < len(sys.argv):
            scripts = sys.argv[idx + 1].split(",")

    if "--conditions" in sys.argv:
        idx = sys.argv.index("--conditions")
        if idx + 1 < len(sys.argv):
            conditions = sys.argv[idx + 1].split(",")

    if "--runs" in sys.argv:
        idx = sys.argv.index("--runs")
        if idx + 1 < len(sys.argv):
            runs = int(sys.argv[idx + 1])

    if "--judge-repeats" in sys.argv:
        idx = sys.argv.index("--judge-repeats")
        if idx + 1 < len(sys.argv):
            judge_repeats = int(sys.argv[idx + 1])

    skip_judge = "--skip-judge" in sys.argv

    models = None
    if "--models" in sys.argv:
        idx = sys.argv.index("--models")
        if idx + 1 < len(sys.argv):
            models = sys.argv[idx + 1].split(",")

    used_conditions = conditions or list_conditions()
    used_scripts = scripts or list(CONVERSATION_SCRIPTS.keys())
    used_models = models or ["mistral-nemo"]
    total_convos = len(used_models) * len(used_conditions) * len(used_scripts) * runs
    total_judgments = total_convos * judge_repeats

    print("=" * 60)
    print("PSYCHE EVALUATION")
    print("=" * 60)
    print(f"Models: {used_models}")
    print(f"Conditions: {used_conditions}")
    print(f"Scripts: {used_scripts}")
    print(f"Runs per cell: {runs}")
    if skip_judge:
        print(f"Judge: SKIPPED (transcripts only)")
        total_judgments = 0
    else:
        print(f"Judge repeats: {judge_repeats}")
    print(f"Total conversations: {total_convos}")
    print(f"Total judge calls: {total_judgments}")
    print()

    results, analysis = run_evaluation(
        scripts=scripts,
        conditions=conditions,
        models=models,
        runs_per_cell=runs,
        judge_repeats=0 if skip_judge else judge_repeats,
    )

    filepath = save_results(results, analysis)
    print_results(results, analysis)
    print(f"\nFull results saved to: {filepath}")


if __name__ == "__main__":
    main()
