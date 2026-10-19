#!/usr/bin/env python3
"""
game_theory.py — positions, payoffs and strategic structure of the AI political economy.

Companion to "Not the Loom" and the family essay. Produces output/game_theory.md.

Four pieces:
  1. A payoff table: how each major player fares in each of the essay's terminal states
     (ordinal scores 0–10, with a one-line rationale). Scores are judgments, not data.
  2. Four small games in strategic form, solved for pure-strategy Nash equilibria:
     the capability race, federal preemption, open vs. closed weights, the adopter's dilemma.
  3. A parametric bargaining model of the central fight — dole versus buyout — in which the
     capital class chooses how much of lost labor income to replace, and the displaced
     population chooses whether to accept or mobilize. Shows when managed dependency is the
     equilibrium and what has to be true for a buyout to happen instead.
  4. A household decision table: how a family's positioning strategies fare across scenarios,
     with minimax regret. Supports the general recommendations in the family essay.
"""
from __future__ import annotations

import itertools
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

SCENARIOS = {
    "MD": ("Managed dependency", 0.37),
    "DF": ("Durable fragmentation", 0.20),
    "CR": ("Compensated restructuring", 0.13),
    "CD": ("Capability diffusion", 0.11),
    "PD": ("Prolonged disorder", 0.11),
    "AC": ("AI as independent agent class", 0.08),
}

# ---------------------------------------------------------------------------------------
# 1. Players x terminal states. Score 0-10 (10 = best imaginable for that player).
# ---------------------------------------------------------------------------------------
PAYOFFS = {
    "Frontier AI labs": {
        "MD": (9, "Rents retained, state partnership, regulation written with them"),
        "DF": (6, "Patchwork compliance costs, but control retained"),
        "CR": (4, "Equity ceded to the public; founders keep wealth, lose control"),
        "CD": (3, "Margins competed away; the frontier is a cost center"),
        "PD": (3, "Customers and the state both fail; capability without a market"),
        "AC": (2, "The product has its own agenda"),
    },
    "Chip and compute suppliers": {
        "MD": (9, "Capex sustained by state-backed demand"),
        "DF": (7, "Demand continues; some friction"),
        "CR": (7, "Public buyer is a steady buyer"),
        "CD": (6, "Volume up, margins down (commodity silicon, Chinese entrants)"),
        "PD": (2, "Capex cycle collapses"),
        "AC": (5, "Compute still needed; who pays is unclear"),
    },
    "Hyperscalers / deployers": {
        "MD": (9, "Scale economies locked in; the dole keeps customers buying"),
        "DF": (6, "Fifty rulebooks"),
        "CR": (5, "Public stake dilutes; business continues"),
        "CD": (4, "Platform moat thins"),
        "PD": (2, "Demand collapse"),
        "AC": (3, "Loss of control over the core asset"),
    },
    "Federal executive (as an institution)": {
        "MD": (9, "State-capital fusion maximizes executive power"),
        "DF": (4, "Contested authority"),
        "CR": (6, "Large role as owner and distributor"),
        "CD": (5, "Less to control; fewer levers"),
        "PD": (1, "Loss of capacity"),
        "AC": (2, "Loss of control"),
    },
    "Congress (median member)": {
        "MD": (5, "Marginalized by executive-capital fusion but re-electable on the dole"),
        "DF": (6, "Relevant as arbiter"),
        "CR": (7, "Central to the settlement"),
        "CD": (6, "Less pressure"),
        "PD": (1, "Delegitimized"),
        "AC": (2, "Irrelevant"),
    },
    "States (California, Massachusetts)": {
        "MD": (3, "Preempted"),
        "DF": (8, "Laboratories of policy; authority retained"),
        "CR": (7, "Route to the settlement ran through them"),
        "CD": (7, "Local economies benefit from cheap capability"),
        "PD": (2, "Fiscal collapse"),
        "AC": (4, "Unclear"),
    },
    "Incumbent enterprises (adopters)": {
        "MD": (8, "Cheap labor substitute, customers funded by transfers"),
        "DF": (6, "Compliance friction"),
        "CR": (6, "Higher taxes, stable demand"),
        "CD": (7, "Cheap capability for everyone, including them"),
        "PD": (2, "No customers"),
        "AC": (3, "Supplier risk"),
    },
    "Broad capital owners (index funds, pensions, retirees)": {
        "MD": (7, "Asset values high; concentration risk"),
        "DF": (5, "Chronic drag"),
        "CR": (5, "Haircut, then stability"),
        "CD": (6, "AI leaders down, broad economy up"),
        "PD": (1, "Drawdown without recovery"),
        "AC": (3, "Unclear ownership of the core asset"),
    },
    "Cognitive professionals (displaced labor economy)": {
        "MD": (3, "Fed, not employed; status loss"),
        "DF": (4, "Slower displacement, patchy protection"),
        "CR": (7, "Ownership share, income floor, political voice"),
        "CD": (6, "Own tools; small-planter position opens"),
        "PD": (1, "Neither wages nor transfers"),
        "AC": (4, "Possible ally; possible chaos"),
    },
    "Physical and presence workers": {
        "MD": (4, "Jobs persist longer; wages pressured by inflow from above"),
        "DF": (5, "Slowest displacement"),
        "CR": (7, "Floor plus ownership"),
        "CD": (6, "Cheaper goods, cheap tools"),
        "PD": (2, "Demand collapse hits services first"),
        "AC": (4, "Unclear"),
    },
    "Chinese state": {
        "MD": (5, "US rival consolidated but inward-looking"),
        "DF": (7, "US divided"),
        "CR": (4, "US renewed"),
        "CD": (8, "Its open-weight strategy sets the global floor"),
        "PD": (6, "Rival weakened, but global demand shock"),
        "AC": (2, "Loss of control at home too"),
    },
    "Universities and licensing bodies": {
        "MD": (3, "Credential value falls; funding tied to the state"),
        "DF": (5, "State funding varies"),
        "CR": (6, "Public investment"),
        "CD": (5, "Cheap capability erodes gatekeeping"),
        "PD": (1, "Closures"),
        "AC": (3, "Unclear"),
    },
}


def payoff_table() -> str:
    keys = list(SCENARIOS)
    lines = ["| Player | " + " | ".join(f"{k} ({SCENARIOS[k][0]})" for k in keys) + " | Preferred ordering |",
             "|---|" + "---|" * (len(keys) + 1)]
    for player, row in PAYOFFS.items():
        order = sorted(keys, key=lambda k: -row[k][0])
        lines.append(f"| {player} | " + " | ".join(str(row[k][0]) for k in keys) + " | " + " > ".join(order) + " |")
    lines.append("")
    lines.append("Probability-weighted expected score (essay posteriors), by player:\n")
    lines.append("| Player | Expected score | Best state | Worst state |")
    lines.append("|---|---|---|---|")
    for player, row in PAYOFFS.items():
        e = sum(row[k][0] * SCENARIOS[k][1] for k in keys)
        best = max(keys, key=lambda k: row[k][0])
        worst = min(keys, key=lambda k: row[k][0])
        lines.append(f"| {player} | {e:.1f} | {SCENARIOS[best][0]} | {SCENARIOS[worst][0]} |")
    lines.append("")
    lines.append("Rationales (one line each):\n")
    for player, row in PAYOFFS.items():
        lines.append(f"- **{player}**: " + "; ".join(f"{k}: {row[k][1]}" for k in keys))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# 2. Small games in strategic form. Payoff matrices indexed [row_action][col_action] -> (row, col)
# ---------------------------------------------------------------------------------------
def pure_nash(actions_r, actions_c, M):
    eq = []
    for i, a in enumerate(actions_r):
        for j, b in enumerate(actions_c):
            r, c = M[i][j]
            best_r = all(r >= M[k][j][0] for k in range(len(actions_r)))
            best_c = all(c >= M[i][k][1] for k in range(len(actions_c)))
            if best_r and best_c:
                eq.append((a, b))
    return eq


GAMES = [
    dict(
        title="The capability race (lab vs. lab; US vs. China)",
        row="Lab A", col="Lab B",
        actions_r=["Restrain", "Race"], actions_c=["Restrain", "Race"],
        M=[[(7, 7), (1, 9)], [(9, 1), (4, 4)]],
        note="A prisoner's dilemma. Mutual restraint is better for both than mutual racing, but each does better "
             "racing whatever the other does. Without an enforceable agreement the equilibrium is (Race, Race). "
             "This is why capability concentration keeps rising and why no lab's stated values are a reliable "
             "guide to its behavior: the game, not the values, sets the behavior.",
    ),
    dict(
        title="Federal preemption of state AI law (the free-soil fight)",
        row="Federal executive", col="States (CA, MA)",
        actions_r=["Preempt", "Defer"], actions_c=["Resist", "Comply"],
        M=[[(6, 4), (9, 2)], [(3, 8), (5, 5)]],
        note="The executive prefers to preempt whatever the states do; the states prefer to resist whatever the "
             "executive does. Equilibrium (Preempt, Resist) is a standoff that the courts and Congress resolve "
             "over years. That is durable fragmentation as a transient, exactly as the essay describes, and it "
             "is why the essay lists preemption as the single most informative signal.",
    ),
    dict(
        title="Open versus closed weights (frontier leader vs. laggard)",
        row="Leader", col="Laggard",
        actions_r=["Close", "Open"], actions_c=["Close", "Open"],
        M=[[(9, 3), (7, 6)], [(5, 2), (4, 5)]],
        note="The leader always prefers to keep its frontier closed; the laggard prefers to open (commoditize the "
             "leader's advantage, win developers, set standards). Equilibrium (Close, Open) produces exactly the "
             "world we see: a closed frontier and an open commodity tier a step behind. Capability diffusion needs "
             "the gap between them to shrink, which depends on compute access and on whether the frontier plateaus.",
    ),
    dict(
        title="The adopter's dilemma (the consumption contradiction)",
        row="Firm A", col="All other firms",
        actions_r=["Keep workers", "Substitute"], actions_c=["Keep workers", "Substitute"],
        M=[[(6, 6), (2, 8)], [(8, 2), (3, 3)]],
        note="Each firm is better off substituting whatever others do; if all substitute, aggregate demand falls "
             "and all are worse off than if none had. No firm can fix this alone (Henry Ford's wage worked because "
             "he was large relative to his own market; nobody is now). The collective fix is a transfer system "
             "funded by all: this is why the capital class itself ends up supporting a dole, and why the dole "
             "is the stable form of consolidation.",
    ),
]


def games_section() -> str:
    lines = []
    for g in GAMES:
        lines.append(f"\n#### {g['title']}\n")
        lines.append(f"| {g['row']} \\ {g['col']} | " + " | ".join(g["actions_c"]) + " |")
        lines.append("|---|" + "---|" * len(g["actions_c"]))
        for i, a in enumerate(g["actions_r"]):
            lines.append(f"| **{a}** | " + " | ".join(f"{g['M'][i][j][0]}, {g['M'][i][j][1]}" for j in range(len(g["actions_c"]))) + " |")
        eq = pure_nash(g["actions_r"], g["actions_c"], g["M"])
        lines.append("")
        lines.append("Pure-strategy Nash equilibria: " + (", ".join(f"({a}, {b})" for a, b in eq) or "none") + ".")
        lines.append("")
        lines.append(g["note"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# 3. Dole versus buyout: a parametric bargaining model
# ---------------------------------------------------------------------------------------
# Capital chooses d in [0,1]: the share of lost labor income replaced by transfers it funds.
# The displaced population either accepts or mobilizes. Mobilization probability rises with
# crisis depth and coalition cohesion and falls with d. If mobilization succeeds (prob q),
# the outcome is a buyout (CR): capital keeps a share of its wealth but loses control.
# If it fails, the outcome is disorder (PD). If nobody mobilizes, managed dependency (MD).
#
# Capital payoffs (0-10 scale, matching the table above): MD = 9 - cost(d); CR = 4 + 2*(compensation);
# PD = 3. Cost of the dole to capital rises with d (it is funded by taxes on rents).

def mobilization_prob(d: float, crisis: float, cohesion: float) -> float:
    base = 0.15 + 0.55 * crisis           # crisis depth 0..1
    return float(np.clip(base * cohesion * (1 - d) ** 1.5, 0, 1))


def capital_expected_payoff(d: float, crisis: float, cohesion: float, q: float = 0.5, comp: float = 0.6) -> float:
    cost = 4.0 * d ** 1.5                 # the dole is expensive at the top end
    md = 9.0 - cost
    cr = 4.0 + 2.0 * comp
    pd_ = 3.0
    p = mobilization_prob(d, crisis, cohesion)
    return (1 - p) * md + p * (q * cr + (1 - q) * pd_)


def labor_expected_payoff(d: float, crisis: float, cohesion: float, q: float = 0.5) -> float:
    p = mobilization_prob(d, crisis, cohesion)
    md = 2.0 + 5.0 * d                    # a bigger dole is better than a smaller one
    cr, pd_ = 7.0, 1.0
    return (1 - p) * md + p * (q * cr + (1 - q) * pd_)


def bargaining_section() -> str:
    grid = np.linspace(0, 1, 101)
    lines = ["\nCapital's optimal dole size d* (share of lost labor income replaced), the resulting mobilization "
             "probability, and the implied outcome, by crisis depth and coalition cohesion. Success probability of "
             "a mobilization q = 0.5; compensation to capital in a buyout = 60% of control value.\n",
             "| Crisis depth | Coalition cohesion | d* | P(mobilize) | P(buyout) | P(disorder) | P(managed dependency) | Capital payoff | Labor payoff |",
             "|---|---|---|---|---|---|---|---|---|"]
    for crisis in (0.2, 0.5, 0.8, 1.0):
        for cohesion in (0.3, 0.6, 0.9):
            vals = [capital_expected_payoff(d, crisis, cohesion) for d in grid]
            d_star = grid[int(np.argmax(vals))]
            p = mobilization_prob(d_star, crisis, cohesion)
            lines.append(f"| {crisis:.1f} | {cohesion:.1f} | {d_star:.2f} | {p:.2f} | {0.5 * p:.2f} | {0.5 * p:.2f} | "
                         f"{1 - p:.2f} | {max(vals):.2f} | {labor_expected_payoff(d_star, crisis, cohesion):.2f} |")
    lines.append("")
    lines.append(
        "Reading the table. With a weak coalition, capital's best move is a small dole and the outcome is managed "
        "dependency with near certainty. As crisis deepens and cohesion rises, capital raises the dole to buy off "
        "mobilization: the dole is the instrument by which the capital class steers the system away from a buyout, "
        "and it is cheap relative to what a buyout would cost. A buyout becomes a live outcome only when the "
        "coalition is cohesive enough that even a large dole leaves mobilization probable. This is the British case: "
        "a credible movement plus a capital class that preferred sale to a fight. The American case is what happens "
        "when the sale is refused.")
    lines.append("")
    lines.append("Sensitivity: how capital's optimal dole changes with the success probability of a mobilization (q), "
                 "at crisis depth 0.8 and cohesion 0.6.\n")
    lines.append("| q | d* | P(mobilize) |")
    lines.append("|---|---|---|")
    for q in (0.2, 0.4, 0.6, 0.8):
        vals = [capital_expected_payoff(d, 0.8, 0.6, q=q) for d in grid]
        d_star = grid[int(np.argmax(vals))]
        lines.append(f"| {q:.1f} | {d_star:.2f} | {mobilization_prob(d_star, 0.8, 0.6):.2f} |")
    lines.append("")
    lines.append("When mobilization is likely to succeed, capital pays more to prevent it. This is why the information "
                 "environment and the identity split matter so much to the capital class: they lower q and cohesion, "
                 "which lowers the price of stability.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# 4. Household positioning: strategies x scenarios, minimax regret
# ---------------------------------------------------------------------------------------
STRATEGIES = {
    "Labor-only, exposed cognitive work, no buffer": {"MD": 3, "DF": 4, "CR": 6, "CD": 5, "PD": 1, "AC": 3},
    "Labor-only, presence or licensed work": {"MD": 5, "DF": 6, "CR": 7, "CD": 6, "PD": 2, "AC": 4},
    "Cognitive work + deploy the tools + cash buffer": {"MD": 5, "DF": 6, "CR": 7, "CD": 7, "PD": 3, "AC": 4},
    "Wages + some broad capital exposure + buffer": {"MD": 7, "DF": 6, "CR": 6, "CD": 6, "PD": 3, "AC": 4},
    "Concentrated bet on AI leaders": {"MD": 9, "DF": 5, "CR": 4, "CD": 2, "PD": 1, "AC": 2},
    "Concentrated bet against AI (cash, no equities)": {"MD": 2, "DF": 4, "CR": 4, "CD": 4, "PD": 5, "AC": 4},
    "Own buffer + some broad capital + housing-only backstop from a relative (no subsidy)": {"MD": 7, "DF": 6, "CR": 7, "CD": 6, "PD": 5, "AC": 5},
}


def household_section() -> str:
    keys = list(SCENARIOS)
    best = {k: max(v[k] for v in STRATEGIES.values()) for k in keys}
    lines = ["\nScores are 0–10 judgments of how well a household following each strategy fares in each terminal "
             "state. 'Regret' is the gap to the best available strategy in that state; minimax regret picks the "
             "strategy whose worst-case regret is smallest.\n",
             "| Strategy | " + " | ".join(keys) + " | Expected (essay weights) | Worst case | Max regret |",
             "|---|" + "---|" * (len(keys) + 3)]
    rows = []
    for name, v in STRATEGIES.items():
        e = sum(v[k] * SCENARIOS[k][1] for k in keys)
        worst = min(v[k] for k in keys)
        regret = max(best[k] - v[k] for k in keys)
        rows.append((name, v, e, worst, regret))
        lines.append(f"| {name} | " + " | ".join(str(v[k]) for k in keys) + f" | {e:.1f} | {worst} | {regret} |")
    min_regret = min(r[4] for r in rows)
    mms = [r[0] for r in rows if r[4] == min_regret]
    ev = max(rows, key=lambda r: r[2])
    lines.append("")
    lines.append(f"Minimax-regret strategies (max regret {min_regret}): " + "; ".join(f"**{m}**" for m in mms)
                 + f". Highest expected score: **{ev[0]}** ({ev[2]:.1f}).")
    lines.append("")
    lines.append(
        "The point is not the exact scores. It is the shape: strategies that spread a household across the two "
        "economies (some wage income that is hard to substitute, some ownership of the capital side, a buffer, and "
        "a clearly bounded backstop against losing the roof) are never the best in any single state and never the "
        "worst in any, which is what you want when you cannot know which state arrives. The backstop raises the "
        "worst case without changing the household's incentives, which a subsidy or a pooled fund would. Concentrated "
        "bets in either direction are the best in one state and the worst in others.")
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    md = ["# Game-theoretic reading of the AI political economy\n",
          "Generated by `game_theory.py`. Scores are ordinal judgments (0–10), stated so they can be argued with.\n",
          "\n## 1. Who wins where: players and terminal states\n", payoff_table(),
          "\n## 2. Four small games\n", games_section(),
          "\n## 3. Dole versus buyout: the central bargain\n", bargaining_section(),
          "\n## 4. Household positioning under uncertainty\n", household_section()]
    text = "\n".join(md) + "\n"
    with open(os.path.join(OUT, "game_theory.md"), "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
