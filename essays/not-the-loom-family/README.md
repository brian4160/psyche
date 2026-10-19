# Not the Loom: the family edition

Working folder for the family-facing companion to *Not the Loom* (October 2026): a plain-language letter about AI, work, and money for a family spanning retirees living on Social Security and equities, mid-career professionals with young children, and students starting out, plus the supporting analysis behind it.

## What is here

| File | What it is |
|---|---|
| `family-essay.md` | **The deliverable.** The letter to the family: mechanism, the two economies, seven endings with all-in odds (including the chance the premise is wrong), how sure any of it is, what each actor will say, the game among them, what it means by life stage with each move written as insurance, an if-then playbook, the family backstop stated precisely (a roof, no subsidy, no pooling), signals to watch, caveats, sources. |
| `what-you-will-hear.md` | Appendix C. For each major actor: the public line with sources, the incentives behind it, how to read it, and a translation table. |
| `model/two_economy.py` | Appendix A. The illustrative two-economy simulation, 2026–2040, with a premise-wrong probability, a Monte Carlo over the shared parameters, and a retiree drawdown module. |
| `model/MODEL.md` | Assumptions, calibration anchors, the Monte Carlo and how its ranges relate to the essay's central case, headline results with ranges, limits. |
| `model/charts.py` | Renders the five charts in `model/output/`. |
| `model/game_theory.py` | Appendix B. Player payoffs by terminal state, four small games with Nash equilibria, the dole-versus-buyout bargaining model, household strategies with minimax regret. |
| `model/output/` | Generated results: `results.md` (central parameters), `uncertainty.md` and `uncertainty.csv` (ranges), `trajectories.csv`, `params.json`, `game_theory.md`, and `macro.png`, `households.png`, `mechanism.png`, `fan.png`, `drawdown.png`. |

## Regenerating

```
cd model
python3 two_economy.py      # ~10 s including the Monte Carlo
python3 charts.py
python3 game_theory.py
```

Python 3.12+, numpy, pandas, matplotlib.

## Before this goes to the family: things to verify

The environment the assistant worked in could reach web search results but not the primary sources themselves, so the figures below were checked against press and law-firm coverage only. Each is quoted in the letter; check it at the primary source before relying on it for a decision.

| Figure in the letter | Where to check |
|---|---|
| OASI trust fund depletion late 2032; ~78% of scheduled benefits payable (≈22% cut); CBO ≈28% | ssa.gov/oact/TRSUM (2026 Trustees Report summary); CBO's long-term Social Security projections |
| Entry-level employment in AI-exposed occupations 19% below peers, up from 13% | digitaleconomy.stanford.edu/news/canariesaug26 (and the working paper) |
| Recent-graduate unemployment ≈5.6–5.7%, underemployment ≈42% | newyorkfed.org/research/college-labor-market |
| Government social benefits ≈ a quarter the size of wages and salaries | BEA NIPA Table 2.1 (or FRED series W209RC1 and A576RC1) |
| Executive Order 14365 (December 11, 2025) and the DOJ AI Litigation Task Force; X.AI v. Weiser (Colorado), DOJ intervention April 2026 | federalregister.gov; the court docket |
| Social Security Fairness Act: signed January 5, 2025; repealed WEP and GPO | ssa.gov/benefits/retirement/social-security-fairness-act.html |
| Physicians using AI: 38% (2023) to 81% (2026); 17% for diagnosis | AMA augmented-intelligence survey |
| Hyperscaler AI capital spending $650–725B in 2026; top-10 S&P weight 36–40% | Company filings; S&P Dow Jones Indices |
| CS undergraduate enrollment −8.1% fall 2025; AI-exposed majors' starting pay −13%; up to 80 college closures by 2030 | National Student Clearinghouse; Census Bureau working paper; Philadelphia Fed |
| California SB 947 signed September 30, 2026 (automated decisions in discipline and termination) | leginfo.legislature.ca.gov; Governor's press release |

## Editorial notes for the author

- **Voice and personal detail.** The letter is in your voice and refers to life stages rather than people. Add names and anything you know that changes a section. The retiree numbers assume a 55/45 split between Social Security and portfolio draw; the drawdown table assumes a 4% real withdrawal rate and shows all-stock, 60/40, and bucket strategies. If the actual withdrawal rate or allocation differs much, rerun the model with `WITHDRAWAL` changed.
- **The premise-wrong probability (25%)** is the assistant's suggestion written in your voice. If you would put it elsewhere, change `P_PREMISE_WRONG` in `two_economy.py`, rerun, and update the odds table and the "short version" bullet in the letter.
- **The Monte Carlo ranges** are the assistant's judgments and are centered milder than your central case; the letter says so in "How sure am I?". If you want the bands centered on your central assumptions, narrow or shift `MC_RANGES`.
- **The backstop.** The letter states, in your voice and without figures, that the one commitment is a room if anyone loses their housing, and that there is no pooling and no subsidy. Adjust the wording if you want it softer or firmer; the household strategy table in `game_theory.py` is scored on that basis.
- **The analogy.** The slave-economy comparison is kept, with the "structural, not moral" caveat, in one paragraph plus the psychological-wage idea. Decide whether that is the right dose for this audience.
- **The harsher numbers** (roughly two in five chance of an involuntary job change by 2036 for a mid-career professional in exposed work if the premise holds; a new-graduate index around 67 in 2034) are now ranges with the old-pattern comparison beside them. Soften further if they read as alarmist for the people involved.
- **Odds.** The scenario table uses the long essay's posteriors scaled by 0.75. If those move, update `ESSAY_POSTERIOR` in `two_economy.py` and `SCENARIOS` in `game_theory.py` and the letter's table.

## Not done, and worth doing

- **One page per person.** The letter is long. A one-page version of each life-stage section, delivered in conversation, is what the family would actually use.
- **Occupation-specific exposure.** The model treats cognitive work as one pool. Splitting it (routine, expert-judgment, licensed, creative) and adding age cohorts would let each person find their own situation rather than an average.
- **Housing.** The largest household cost is absent from the model; it rises with asset values, which cuts against the working-age households and for the retired ones.
- **Primary-source verification** of the table above.
