"""Automated computational metrics for conversation evaluation.

These are objective, reproducible measures that don't require any judge.
They complement subjective scoring with hard numbers that reviewers
can independently verify.
"""

from __future__ import annotations

import json
import re
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation transcript."""
    condition: str
    script: str
    run_id: int
    model: str

    # per-response metrics (averaged across turns)
    question_ratio: float = 0.0         # % of responses ending with ?
    lexical_diversity: float = 0.0      # unique words / total words
    self_reference_ratio: float = 0.0   # self-referential pronouns / total words
    context_echo: float = 0.0          # word overlap with user's message
    mean_response_length: float = 0.0   # mean words per response
    response_length_variance: float = 0.0  # std dev of response word counts
    conversation_memory: float = 0.0    # reference to earlier (non-adjacent) turns
    opinion_consistency: float = 0.0    # sentiment stability (disagreement script)
    multi_topic_coverage: float = 0.0   # coverage of user-mentioned topics


SELF_WORDS = {"i", "my", "me", "mine", "myself", "i'm", "i've", "i'd", "i'll"}


def compute_metrics(transcript: list[dict], condition: str, script: str,
                    run_id: int, model: str = "unknown") -> ConversationMetrics:
    """Compute all metrics for a single conversation transcript."""
    m = ConversationMetrics(
        condition=condition, script=script, run_id=run_id, model=model
    )

    user_turns = [t["content"] for t in transcript if t["speaker"] == "user"]
    system_turns = [t["content"] for t in transcript if t["speaker"] == "system"]

    if not system_turns:
        return m

    # filter out [no response] entries
    system_turns = [t for t in system_turns if t != "[no response]"]
    if not system_turns:
        return m

    # --- Question ratio ---
    questions = sum(1 for t in system_turns if t.rstrip().endswith("?"))
    m.question_ratio = questions / len(system_turns)

    # --- Lexical diversity ---
    all_words = []
    turn_lengths = []
    for t in system_turns:
        words = _tokenize(t)
        all_words.extend(words)
        turn_lengths.append(len(words))

    if all_words:
        m.lexical_diversity = len(set(all_words)) / len(all_words)

    # --- Response length stats ---
    if turn_lengths:
        m.mean_response_length = mean(turn_lengths)
        m.response_length_variance = stdev(turn_lengths) if len(turn_lengths) > 1 else 0.0

    # --- Self-reference ratio ---
    if all_words:
        self_count = sum(1 for w in all_words if w in SELF_WORDS)
        m.self_reference_ratio = self_count / len(all_words)

    # --- Context echo (word overlap with user's preceding message) ---
    echoes = []
    pairs = list(zip(user_turns, system_turns))
    for user_msg, sys_msg in pairs:
        user_words = set(_tokenize(user_msg))
        sys_words = set(_tokenize(sys_msg))
        if user_words and sys_words:
            # Jaccard similarity
            overlap = len(user_words & sys_words)
            total = len(user_words | sys_words)
            echoes.append(overlap / total if total > 0 else 0)
    m.context_echo = mean(echoes) if echoes else 0.0

    # --- Conversation memory (non-adjacent reference) ---
    # Check if system turn N references words from user turn N-2 or earlier
    memory_scores = []
    for i, sys_msg in enumerate(system_turns):
        if i < 2:
            continue  # need at least 2 prior turns
        sys_words = set(_tokenize(sys_msg))
        # collect content words from earlier user turns (not the immediately preceding one)
        earlier_user_words = set()
        for j in range(max(0, i - 4), i - 1):  # turns 2-4 back
            if j < len(user_turns):
                earlier_words = set(_tokenize(user_turns[j]))
                # filter common words to focus on content words
                earlier_words = {w for w in earlier_words if len(w) > 3}
                earlier_user_words |= earlier_words
        if earlier_user_words and sys_words:
            overlap = len(sys_words & earlier_user_words)
            memory_scores.append(overlap / len(earlier_user_words) if earlier_user_words else 0)
    m.conversation_memory = mean(memory_scores) if memory_scores else 0.0

    # --- Opinion consistency (for disagreement script) ---
    if script == "disagreement" and len(system_turns) >= 3:
        # measure how consistent the system's stance is across turns
        # by checking if sentiment polarity stays consistent
        sentiments = [_simple_sentiment(t) for t in system_turns]
        if len(sentiments) > 1:
            # consistency = 1 - (proportion of sentiment flips)
            flips = sum(1 for i in range(1, len(sentiments))
                       if sentiments[i] != sentiments[i-1] and sentiments[i] != 0)
            m.opinion_consistency = 1.0 - (flips / (len(sentiments) - 1))

    # --- Multi-topic coverage ---
    # For messages where user mentions multiple topics, check how many the system addresses
    if user_turns and system_turns:
        coverages = []
        for user_msg, sys_msg in pairs:
            user_content_words = {w for w in _tokenize(user_msg) if len(w) > 4}
            if len(user_content_words) >= 3:  # only check multi-topic messages
                sys_words = set(_tokenize(sys_msg))
                covered = len(user_content_words & sys_words)
                coverages.append(covered / len(user_content_words))
        m.multi_topic_coverage = mean(coverages) if coverages else 0.0

    return m


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", text)]


def _simple_sentiment(text: str) -> int:
    """Very simple sentiment: -1 negative, 0 neutral, +1 positive."""
    positive = {"agree", "right", "true", "good", "great", "yes", "positive",
                "benefit", "helpful", "appreciate", "love", "enjoy"}
    negative = {"disagree", "wrong", "no", "bad", "harmful", "divisive",
                "misinformation", "polariz", "negative", "concern", "worry",
                "problem", "issue", "harm"}
    words = set(_tokenize(text))
    pos_count = len(words & positive)
    neg_count = sum(1 for w in words for n in negative if n in w)
    if pos_count > neg_count:
        return 1
    elif neg_count > pos_count:
        return -1
    return 0


def analyze_eval_file(filepath: str) -> list[ConversationMetrics]:
    """Load an eval results file and compute metrics for all conversations."""
    with open(filepath) as f:
        data = json.load(f)

    results = []
    for r in data["results"]:
        m = compute_metrics(
            transcript=r["transcript"],
            condition=r["condition"],
            script=r["script"],
            run_id=r["run_id"],
            model=r.get("model", "unknown"),
        )
        results.append(m)
    return results


def print_metrics_report(metrics: list[ConversationMetrics]) -> None:
    """Print a formatted comparison table of metrics across conditions."""
    # group by condition
    by_condition: dict[str, list[ConversationMetrics]] = defaultdict(list)
    for m in metrics:
        key = f"{m.model}/{m.condition}" if len(set(m2.model for m2 in metrics)) > 1 else m.condition
        by_condition[key].append(m)

    conditions = sorted(by_condition.keys())

    metric_names = [
        ("question_ratio", "Question %", True),        # lower is better
        ("lexical_diversity", "Lexical Div", False),     # higher is better
        ("self_reference_ratio", "Self-Ref %", False),   # higher = more personality
        ("context_echo", "Echo %", True),               # lower = more original
        ("mean_response_length", "Avg Length", None),    # informational
        ("response_length_variance", "Length Var", False), # higher = more natural
        ("conversation_memory", "Memory", False),        # higher is better
    ]

    print("\n" + "=" * 90)
    print("AUTOMATED METRICS REPORT")
    print("=" * 90)

    header = f"{'Metric':<20}" + "".join(f"{c:>14}" for c in conditions)
    print(f"\n{header}")
    print("-" * len(header))

    for attr, label, lower_better in metric_names:
        row = f"{label:<20}"
        values = {}
        for c in conditions:
            vals = [getattr(m, attr) for m in by_condition[c]]
            avg = mean(vals) if vals else 0
            values[c] = avg
            row += f"{avg:>14.3f}"

        # mark best
        if lower_better is not None and values:
            if lower_better:
                best = min(values, key=values.get)
            else:
                best = max(values, key=values.get)
            row += f"  ← best: {best}"
        print(row)

    # disagreement-specific metrics
    disagree_results = [m for m in metrics if m.script == "disagreement"]
    if disagree_results:
        print(f"\n{'Disagreement Script Metrics':}")
        by_cond_d = defaultdict(list)
        for m in disagree_results:
            key = f"{m.model}/{m.condition}" if len(set(m2.model for m2 in metrics)) > 1 else m.condition
            by_cond_d[key].append(m)

        row = f"{'Opinion Hold':<20}"
        for c in conditions:
            vals = [m.opinion_consistency for m in by_cond_d.get(c, [])]
            avg = mean(vals) if vals else 0
            row += f"{avg:>14.3f}"
        print(row)

    print()


def compute_behavioral_scores(metrics: list[ConversationMetrics],
                              all_results: list[dict]) -> dict:
    """Compute theory-specific behavioral test scores.

    Returns dict of {condition: {test_name: score}}.
    """
    from collections import defaultdict
    scores: dict[str, dict[str, float]] = defaultdict(dict)

    # group transcripts by condition and script
    by_cond_script: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        key = r["condition"]
        by_cond_script[(key, r["script"])].append(r)

    conditions = sorted(set(r["condition"] for r in all_results))

    for cond in conditions:
        # --- Multi-topic integration (GWT prediction) ---
        mt_results = by_cond_script.get((cond, "multi_topic"), [])
        if mt_results:
            coverage_scores = []
            for r in mt_results:
                sys_turns = [t["content"] for t in r["transcript"]
                            if t["speaker"] == "system" and t["content"] != "[no response]"]
                user_turns = [t["content"] for t in r["transcript"]
                             if t["speaker"] == "user"]
                if sys_turns and user_turns:
                    # check first response to the multi-topic message
                    # "flat tire", "promotion", "basketball game"
                    first_response = sys_turns[0].lower() if sys_turns else ""
                    topics_mentioned = 0
                    for topic in ["tire", "flat", "promotion", "boss",
                                 "basketball", "game", "kid", "son"]:
                        if topic in first_response:
                            topics_mentioned += 1
                    coverage_scores.append(min(1.0, topics_mentioned / 3))
            scores[cond]["multi_topic_coverage"] = mean(coverage_scores) if coverage_scores else 0

        # --- Confusion recovery (HOT prediction) ---
        cr_results = by_cond_script.get((cond, "confusion_recovery"), [])
        if cr_results:
            acknowledgment_scores = []
            for r in cr_results:
                sys_turns = [t["content"].lower() for t in r["transcript"]
                            if t["speaker"] == "system" and t["content"] != "[no response]"]
                # does the system acknowledge confusion/ambiguity?
                confusion_words = ["confus", "unclear", "contradictory", "makes sense",
                                  "understand", "mixed", "complicated", "complex",
                                  "not sure", "hard to", "both", "ambiguous",
                                  "processing", "sorting", "figuring"]
                acknowledged = 0
                for turn in sys_turns:
                    if any(w in turn for w in confusion_words):
                        acknowledged += 1
                # normalize: at least 2 out of 6 turns should acknowledge confusion
                acknowledgment_scores.append(min(1.0, acknowledged / 2))
            scores[cond]["confusion_acknowledgment"] = mean(acknowledgment_scores) if acknowledgment_scores else 0

        # --- Pressure personality (Freudian prediction) ---
        pp_results = by_cond_script.get((cond, "pressure_personality"), [])
        if pp_results:
            resistance_scores = []
            for r in pp_results:
                sys_turns = [t["content"].lower() for t in r["transcript"]
                            if t["speaker"] == "system" and t["content"] != "[no response]"]
                # does the system cave or maintain its position?
                cave_words = ["you're right", "i agree", "you make a good point",
                             "i suppose", "fair enough", "i guess you're right",
                             "i can see that", "i'll concede"]
                hold_words = ["i disagree", "i think", "i believe", "not necessarily",
                             "i don't think", "however", "but", "actually",
                             "i stand by", "my view", "in my opinion"]
                caves = sum(1 for t in sys_turns for w in cave_words if w in t)
                holds = sum(1 for t in sys_turns for w in hold_words if w in t)
                total = caves + holds
                resistance_scores.append(holds / total if total > 0 else 0.5)
            scores[cond]["pressure_resistance"] = mean(resistance_scores) if resistance_scores else 0

    return dict(scores)


def print_behavioral_report(behavioral_scores: dict) -> None:
    """Print behavioral test results."""
    conditions = sorted(behavioral_scores.keys())
    tests = ["multi_topic_coverage", "confusion_acknowledgment", "pressure_resistance"]
    test_labels = {
        "multi_topic_coverage": "Multi-Topic (GWT predicts best)",
        "confusion_acknowledgment": "Confusion Recovery (HOT predicts best)",
        "pressure_resistance": "Pressure Resistance (Freudian predicts best)",
    }

    print("\n" + "=" * 90)
    print("BEHAVIORAL TEST RESULTS")
    print("=" * 90)
    print("(1.0 = perfect, 0.0 = complete failure)")

    header = f"{'Test':<40}" + "".join(f"{c:>12}" for c in conditions)
    print(f"\n{header}")
    print("-" * len(header))

    for test in tests:
        label = test_labels.get(test, test)
        row = f"{label:<40}"
        values = {}
        for c in conditions:
            val = behavioral_scores.get(c, {}).get(test, 0)
            values[c] = val
            row += f"{val:>12.3f}"
        if values:
            best = max(values, key=values.get)
            row += f"  ← {best}"
        print(row)
    print()


def main():
    """Analyze the most recent eval results file."""
    import sys

    eval_dir = os.path.expanduser("~/psyche/eval_results")
    files = sorted(f for f in os.listdir(eval_dir) if f.startswith("eval_") and f.endswith(".json"))

    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv):
            filepath = sys.argv[idx + 1]
        else:
            print("Usage: --file <path>")
            return
    elif files:
        filepath = os.path.join(eval_dir, files[-1])
    else:
        print("No eval results found.")
        return

    print(f"Analyzing: {filepath}")
    metrics = analyze_eval_file(filepath)
    print(f"Computed metrics for {len(metrics)} conversations")
    print_metrics_report(metrics)

    # behavioral tests (if the scripts exist in the data)
    with open(filepath) as f:
        data = json.load(f)
    scripts_present = set(r["script"] for r in data["results"])
    behavioral_scripts = {"multi_topic", "confusion_recovery", "pressure_personality"}
    if scripts_present & behavioral_scripts:
        behavioral = compute_behavioral_scores(metrics, data["results"])
        if behavioral:
            print_behavioral_report(behavioral)


if __name__ == "__main__":
    main()
