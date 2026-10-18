# The two-economy model: assumptions, calibration, results, limits

`two_economy.py` is a small simulation that turns the mechanism in *Not the Loom* into numbers for 2026 to 2040. It is an illustration of a structure, not a forecast. Every parameter is a judgment and is written out in the `Params` dataclass so that it can be argued with or changed. The point of running it is to see the *shape* of each terminal state and the *ordering* of outcomes across household types, not to produce point estimates.

Outputs (in `output/`): `trajectories.csv` (every variable, every scenario, every year, plus a probability-weighted row set), `results.md` (summary tables), `params.json`, and three charts (`macro.png`, `households.png`, `mechanism.png`) rendered by `charts.py`.

## Structure

The model runs in annual steps. Units: 2026 output = 100, 2026 jobs = 100, 2026 prices = 1.

1. **Capability.** The share of cognitive tasks that frontier AI can do at acceptable quality follows a logistic curve anchored at 25% in 2026, passing 50% around 2031, approaching 85%. Physical and presence tasks (robotics) follow a slower curve: 5% in 2026, about 27% in 2036, 47% in 2040. The "loom" comparator caps cognitive capability at 45%.
2. **Deployment.** Firms close the gap between what AI can do and what they have actually substituted at 30% a year (22% in the fragmentation scenario). Deployment therefore trails capability by two to three years, which is where the "it hasn't happened yet" observation comes from.
3. **New tasks.** Following Acemoglu and Restrepo, displaced tasks create new tasks (0.5 per displaced task). The share of those new tasks that humans perform is the key difference between the essay's world (20% in cognitive work, 35% in presence work) and the loom world (80%). Human task demand also rises with total output (elasticity 0.4), which is the "productivity effect" optimists rely on.
4. **Two labor markets.** Cognitive jobs (42% of jobs, 57% of wages) and presence jobs (58% of jobs, 43% of wages). In each, the market-clearing wage falls when the pool of workers exceeds the tasks that remain human, with a labor demand elasticity of −0.7. Wages adjust halfway to the clearing level each year (stickiness) and cannot fall below a reservation floor at 45% of the trend average wage; when the floor binds, employment is rationed instead and the "not employed" share rises. A quarter of surplus cognitive workers move into presence work each year, which is why presence wages flatten and then fall in the substitution scenarios while they rise 50% in the loom world.
5. **Trend growth.** Every base wage carries 1.5% a year of trend productivity growth, in every scenario, so a flat real-wage line already means falling behind. This is what makes the loom comparator meaningful: there, real cognitive wages rise about 26% by 2040.
6. **Output and rents.** Potential output rises with substitution (a 50% gain if all cognitive tasks were substituted, 40% for physical tasks, weighted by each family's share of wages). A share of the AI gain (`rent_capture`: 75% in managed dependency, 30% in diffusion) accrues as concentrated rents; the rest is competed into lower consumer prices, which raises real incomes. "Real" variables are deflated by that price index.
7. **Transfers (the dole).** New transfers replace a share of lost labor income (`transfer_response`) after a lag, plus a share of AI rents recycled through the state (`rent_recycling`), capped by a political ceiling on the dependency ratio (`dependency_cap`). Sixty percent of new transfers are funded by taxes on capital income; the rest is borrowed. In managed dependency the response is 90% of lost wages, a one-year lag, 45% of rents recycled, and a 65% ceiling. In prolonged disorder it is 20%, three years, 10%, and 45%.
8. **Demand and crisis.** Households spend 95% of wage-plus-transfer income; capital spends 40% of net capital income; AI capital spending (1.5% of GDP in 2026, growing 20% a year to 2030) is a demand injection. If spending falls short of potential output at the prevailing price level, 60% of the gap becomes lost output. A gap above 5% for two years in a row triggers a crisis: an extra output shock, a collapse in valuations and AI capital spending, and a policy response that depends on the scenario (expand the dole; take public equity in the AI value chain and pay it out as dividends; or fail to respond). The fragmentation scenario never triggers an acute crisis and instead carries a chronic shortfall.
9. **Valuation.** Broad equity is net capital income relative to 2026, times a valuation multiple (rising 15% in the exuberant scenarios, falling 20% in diffusion, cut 40% in a crisis with partial recovery, cut 50% without). A 60/40 portfolio is 60% broad equity, 40% a bond index rising 1% a year in real terms (and cut 10% in a disorder crisis).
10. **Social Security.** Indexed at 1.0, then cut in 2033 by the scenario's assumption: 0% where the political system is functioning (managed dependency, restructuring), 5 to 10% where it is muddling, and the scheduled 22% where it fails (disorder).

## Household archetypes

- **Retired couple:** 55% Social Security, 45% a draw proportional to the value of a 60/40 portfolio. Also reported: the real cost of presence (care) labor.
- **Mid-career cognitive incumbent:** protected by attrition at first (protection decays with a six-year half-life), then exposed to the market wage. An annual hazard of involuntary exit (1.5% base, plus a term proportional to net task displacement once protection fades, plus a bump in a crisis year) gives the cumulative probability of an involuntary job change. The income of a displaced incumbent is modeled as presence-sector pay plus a per-head share of new transfers.
- **New graduate entering cognitive work:** an "opportunity" index = openings relative to 2026 (the 19% entry-level gap measured in 2026, widening with deployment) times the real cognitive wage.
- **Licensed presence profession** (clinical medicine as the example): licensure caps substitution at 30% of tasks until 2034, rising to 50% by 2040; half of freed capacity meets unmet demand rather than reducing demand; demand grows 1.5% a year with aging; the trained workforce grows 1% a year.
- **Infrastructure engineer** (physical/compute side): pay tracks the AI capital-spending cycle with an elasticity of 0.2, so it rises in the buildout and falls in a crisis.

## Calibration anchors (2026)

| Quantity | Value used | Basis |
|---|---|---|
| Labor share of output | 57% | BLS nonfarm business labor share, mid-50s percent |
| Cognitive vs presence jobs | 42 / 58 | Judgment from occupational employment; cognitive jobs pay about 1.8x |
| Transfers / (wages + transfers) | 25% | BEA: government social benefits roughly a quarter the size of wages and salaries (most of it Social Security, Medicare, Medicaid) |
| Unemployment | 4.2% | BLS, mid-2026 |
| Entry-level employment gap in exposed occupations | 19% | Stanford Digital Economy Lab, August 2026 update of "Canaries in the Coal Mine" |
| AI capital spending, share of GDP | 1.5% | Epoch AI, Q1 2026 (broad measure) |
| Social Security depletion and cut | late 2032, 22% | 2026 Trustees Report |
| AI can do 25% of cognitive tasks at acceptable quality | 25% | Judgment |

## Headline results

The probability-weighted expectation uses the essay's posteriors renormalized over the five modeled states (managed dependency 37, durable fragmentation 20, compensated restructuring 13, capability diffusion 11, prolonged disorder 11). The range is the lowest and highest scenario value, not a confidence interval. Full tables are in `output/results.md`.

| Measure (2026 = 100 unless %) | 2031 | 2036 | 2040 | Loom comparator 2040 |
|---|---|---|---|---|
| Real cognitive wage | 86 (84–90) | 59 (56–68) | 50 (45–56) | 126 |
| Real presence wage | 114 (113–118) | 117 (115–127) | 103 (98–115) | 149 |
| Not employed, share of workforce | 6% | 8% (8–9) | 10% (8–12) | 4% |
| Dependency ratio | 31% (28–33) | 43% (35–49) | 53% (41–61) | 24% |
| Real household income, all households | 105 (99–107) | 112 (93–119) | 118 (86–132) | 139 |
| Labor share of output | 51% | 38% | 29% | 56% |
| Output | 110 (89–115) | 127 (99–134) | 141 (102–151) | 138 |
| 60/40 portfolio, real | 113 (66–134) | 143 (82–165) | 166 (92–191) | 126 |
| Retired couple income | 109 (86–117) | 123 (83–135) | 136 (89–148) | 118 |
| Incumbent professional who keeps the job, real pay | 97 | 71 (68–79) | 58 (53–65) | 126 |
| P(incumbent involuntary exit by year) | 23% | 47% (44–51) | 57% | 20% (by 2036) |
| New graduate opportunity (openings × pay) | 75 | 43 (40–48) | 33 (30–37) | 102 |
| Licensed presence profession, real pay | 107 (92–113) | 117 (99–126) | 127 (108–141) | 138 |
| Infrastructure engineer, real pay | 125 (113–133) | 141 (122–146) | 157 (128–168) | 160 |

Crisis years by scenario: compensated restructuring 2030, prolonged disorder 2030, capability diffusion 2033, managed dependency none (the dole arrives in time), durable fragmentation none (chronic shortfall instead).

Three things the tables show that are not obvious from the verbal argument:

1. **The pie grows in almost every scenario.** Output in 2040 is higher than in the loom world in four of five states. The problem is not scarcity; it is the split.
2. **Households as a whole are not poorer in the modal state.** In managed dependency, real household income is up about 30% by 2040 while the labor share falls from 57% to 26%. The difference is paid by the state. That is what "managed dependency" means.
3. **The family is split across the line.** Asset owners (the retired couple) do better in the modal states than in the loom world; wage earners in cognitive work do far worse; new entrants worst of all; licensed presence work and physical infrastructure sit in between and close to trend.

## Limits

- **One analyst's parameters.** Everything in the `Params` class is a judgment. The results are only as good as those judgments, and the honest claim is about shape and ordering, not level.
- **The equity index is not the S&P 500.** It is net capital income times a stylized multiple. The real index is concentrated, driven by valuation, and can fall further and faster than this.
- **The hazard model is a mapping, not data.** The probability of an involuntary job change converts task displacement into job loss with a judgment coefficient. Read the 44 to 51% as "roughly a coin flip under the essay's assumptions," not as a measured rate.
- **Cognitive work is treated as one pool.** In reality it is dozens of occupations with different exposure, licensure, and presence content; the average hides the spread. The licensed-profession and infrastructure archetypes are the model's only concession to that.
- **The wage floor produces a kink.** When the reservation wage binds (late 2030s in the heavier scenarios), employment is rationed instead of wages falling, and the lines bend. The mechanism is real; its sharpness is an artifact.
- **Transfers assume fiscal capacity.** The dole in managed dependency costs about 45% of AI rents plus 90% of lost wages. Whether the political system can collect that from the capital class is the essay's central question, and the model assumes the answer rather than deriving it.
- **No international sector, no explicit inflation, no monetary policy, no heterogeneity within households.** The capital-spending cycle is stylized.
- **The wildcard is not modeled.** The essay's "AI as independent agent class" state (8%) has no economic representation here; the essay treats it as redistributing probability toward diffusion or restructuring.
- **Sensitivities that matter most:** the human share of new tasks (the single biggest lever; 80% gives the loom world), the labor demand elasticity (a less elastic −0.4 roughly doubles wage declines), the adoption rate (slower adoption delays but does not change the end state), and the rent-recycling share (which decides whether managed dependency is stable).

## Running it

```
python3 two_economy.py   # writes output/trajectories.csv, results.md, params.json
python3 charts.py        # writes output/macro.png, households.png, mechanism.png
python3 game_theory.py   # writes output/game_theory.md
```

Requires Python 3.12+, numpy, pandas, matplotlib.
