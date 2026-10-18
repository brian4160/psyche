# Not the Loom: the family edition

Working folder for the family-facing companion to *Not the Loom* (October 2026): a plain-language letter about AI, work, and money for a family spanning retirees living on Social Security and equities, mid-career professionals with young children, and students starting out, plus the supporting analysis behind it.

## What is here

| File | What it is |
|---|---|
| `family-essay.md` | **The deliverable.** The letter to the family: mechanism, the two economies, six endings with odds, what each actor will say, the game among them, what it means by life stage, the family as a unit, signals to watch, caveats, sources. |
| `what-you-will-hear.md` | Appendix C. For each major actor (White House, Congress, frontier labs, chip makers, hyperscalers, Wall Street, employers, the Fed, China, states, universities, unions, pundits, AI assistants): the public line with sources, the incentives behind it, how to read it, and a translation table. |
| `model/two_economy.py` | Appendix A. The illustrative two-economy simulation, 2026–2040: capability, deployment, two labor markets, transfers, demand and crisis, valuation, household archetypes. Six parameter sets (five terminal states plus a "loom" comparator). |
| `model/MODEL.md` | Assumptions, calibration anchors, headline results, limits. |
| `model/charts.py` | Renders `model/output/macro.png`, `households.png`, `mechanism.png` from the model's CSV. |
| `model/game_theory.py` | Appendix B. Player payoffs by terminal state, four small games with Nash equilibria, the dole-versus-buyout bargaining model, household strategies with minimax regret. Writes `model/output/game_theory.md`. |
| `model/output/` | Generated results: `trajectories.csv`, `results.md`, `params.json`, `game_theory.md`, the three charts. |

## Regenerating

```
cd model
python3 two_economy.py
python3 charts.py
python3 game_theory.py
```

Python 3.12+, numpy, pandas, matplotlib.

## Editorial notes for the author

Things to look at before this goes to the family.

- **Voice and personal detail.** The letter is written in your voice and deliberately refers to life stages rather than people. Add names, warmth, and anything you know that changes a section. The "retired" section assumes a 55/45 split between Social Security and portfolio draw and a 60/40 portfolio; adjust if you know otherwise.
- **The analogy.** The slave-economy comparison is kept, with the same "structural, not moral" caveat as the long essay, in one paragraph plus the psychological-wage idea. Decide whether that is the right dose for this audience.
- **The model's harsher numbers.** The letter quotes the model's coin-flip chance of an involuntary job change by 2036 for mid-career professionals and the roughly 50 to 60 "new graduate opportunity" index in 2034. Both are flagged as crude mappings in the text and in `MODEL.md`; soften or cut if they read as alarmist for the people involved.
- **Facts that came from secondary coverage.** Several 2026 figures (the Stanford 19% gap, the Trustees Report dates, the preemption litigation, the California bills, physician AI adoption, the capital-spending totals) were verified against press and law-firm summaries, not primary documents. The sources list in the letter links what was used. Check anything that matters to a decision against the primary report.
- **The AI-assistance disclosure** is in "How this was written" in the letter and in `what-you-will-hear.md`. It is there on purpose and matches the "author position" caveat of the long essay.
- **Odds.** The scenario table uses the long essay's posteriors unchanged. If those move, update the table in the letter, `ESSAY_POSTERIOR` in `two_economy.py`, and `SCENARIOS` in `game_theory.py`.
