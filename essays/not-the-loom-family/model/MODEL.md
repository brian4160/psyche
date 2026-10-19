# The two-economy model: assumptions, calibration, uncertainty, results, limits

`two_economy.py` is a small simulation that turns the mechanism in *Not the Loom* into numbers for 2026 to 2040. It is an illustration of a structure, not a forecast. Every parameter is a judgment and is written out in the `Params` dataclass so that it can be argued with or changed. The point of running it is to see the *shape* of each terminal state, the *ordering* of outcomes across household types, and, through the Monte Carlo, *how much* both depend on the judgments.

Outputs (in `output/`):

- `results.md`, `trajectories.csv`, `params.json`: point runs with the central parameters, plus the probability-weighted rows.
- `uncertainty.md`, `uncertainty.csv`: 10th/50th/90th percentiles across 400 draws of the shared parameters, per scenario and for the weighted mixtures.
- `macro.png`, `households.png`, `mechanism.png`, `fan.png`, `drawdown.png`: charts rendered by `charts.py`.

## The premise, and the chance it is wrong

The essay's six terminal states, and their posteriors, are conditional on its premise: that AI becomes a general substitute across cognitive tasks within 10 to 15 years. The model carries a separate judgment, `P_PREMISE_WRONG = 0.25`, that the premise fails and the standard precedent holds (capability plateaus, new tasks absorb displaced labor, wages rise with productivity). That case is represented by the "loom" comparator scenario.

Two weightings are reported:

- **Conditional**: the five modeled states renormalized over their posteriors (37, 20, 13, 11, 11; the 8% wildcard is not modeled). This is "if the premise holds".
- **All-in**: the loom at 25% and the five states at 75% of their conditional weights (managed dependency 28%, fragmentation 15%, restructuring 10%, diffusion 8%, disorder 8%).

The all-in mixture is an expected value across two quite different worlds, so the letter reports the two worlds separately (the conditional band and the comparator line) and uses the all-in only for the odds table.

## Structure

The model runs in annual steps. Units: 2026 output = 100, 2026 jobs = 100, 2026 prices = 1.

1. **Capability.** The share of cognitive tasks that frontier AI can do at acceptable quality follows a logistic curve anchored at 25% in 2026, passing 50% around 2031, approaching 85%. Physical and presence tasks (robotics) follow a slower curve: 5% in 2026, about 27% in 2036, 47% in 2040. The loom comparator caps cognitive capability at 45%.
2. **Deployment.** Firms close the gap between what AI can do and what they have actually substituted at 30% a year (22% in the fragmentation scenario). Deployment therefore trails capability by two to three years, which is where the "it hasn't happened yet" observation comes from.
3. **New tasks.** Following Acemoglu and Restrepo, displaced tasks create new tasks (0.5 per displaced task). The share of those new tasks that humans perform is the key difference between the essay's world (20% in cognitive work, 35% in presence work) and the loom world (80%). Human task demand also rises with total output (elasticity 0.4), the "productivity effect" optimists rely on.
4. **Two labor markets.** Cognitive jobs (42% of jobs, 57% of wages) and presence jobs (58% of jobs, 43% of wages). In each, the market-clearing wage falls when the pool of workers exceeds the tasks that remain human, with a labor demand elasticity of −0.7. Wages adjust halfway to the clearing level each year and cannot fall below a reservation floor at 45% of the trend average wage; when the floor binds, employment is rationed instead and the "not employed" share rises. A quarter of surplus cognitive workers move into presence work each year, which is why presence wages flatten and then fall in the substitution scenarios while they rise 50% in the loom world.
5. **Trend growth.** Every base wage carries 1.5% a year of trend productivity growth, in every scenario, so a flat real-wage line already means falling behind.
6. **Output, rents, prices.** Potential output rises with substitution (a 50% gain if all cognitive tasks were substituted, 40% for physical tasks, weighted by each family's share of wages). A share of the AI gain (`rent_capture`: 75% in managed dependency, 30% in diffusion) accrues as concentrated rents; the rest is competed into lower consumer prices. "Real" variables are deflated by that price index. Rents fall with realized output: no customers, no rents.
7. **Transfers (the dole).** New transfers replace a share of lost labor income (`transfer_response`) after a lag, plus a share of AI rents recycled through the state (`rent_recycling`), capped by a political ceiling on the dependency ratio. Sixty percent of new transfers are funded by taxes on capital income; the rest is borrowed. In managed dependency the response is 90% of lost wages with a one-year lag, 45% of rents recycled, and a 65% ceiling; that is the smallest recycling share at which the scenario avoids an acute break. In disorder it is 20%, three years, 10%, and 45%.
8. **Demand and crisis.** Households spend 95% of wage-plus-transfer income; capital spends 40% of net capital income; AI capital spending (1.5% of GDP in 2026, growing 20% a year to 2030) is a demand injection. If spending falls short of potential output at the prevailing price level, 60% of the gap becomes lost output. A gap above 5% for two consecutive years triggers a crisis: an extra output shock, a collapse in valuations and AI capital spending, and a policy response that depends on the scenario (expand the dole; take public equity in the AI value chain and pay it out; or fail to respond). Fragmentation never breaks acutely and carries a chronic shortfall instead.
9. **Listed equity.** National net capital income maps to listed earnings with a beta of 0.6 (housing, private business, government and foreign holdings take the rest). The real price index is listed earnings times a valuation multiple (rising 15% in the exuberant scenarios, falling 20% in diffusion, cut 40% in a crisis with partial recovery, cut 50% without). Total return adds a real cash yield of 3.5% at a neutral multiple (dividends plus buybacks), which rises as the multiple falls. A 60/40 portfolio is 60% equity total return and 40% an inflation-protected bond index earning 1.5% real (cut 10% in a disorder crisis).
10. **Retiree drawdown.** Savings start at 100 and pay out 4 a year in real terms under three strategies: all equity; 60/40 rebalanced yearly; and all equity plus a cash bucket of three years' withdrawals, spent only when the equity price index is below its previous peak and refilled from equities at new peaks. Cash earns nothing in real terms.
11. **Social Security.** Indexed at 1.0, then cut in 2033 by the scenario's assumption: 0% where the political system is functioning, 5 to 10% where it is muddling, the scheduled 22% where it fails.

## Household archetypes

- **Retired couple:** 55% Social Security, 45% a draw proportional to the value of a 60/40 total-return portfolio. Also reported: the real cost of presence (care) labor, and the drawdown paths above.
- **Mid-career cognitive incumbent:** protected by attrition at first (protection decays with a six-year e-folding time), then exposed to the market wage. An annual hazard of involuntary exit (1.5% base, plus a term proportional to net task displacement once protection fades, plus a bump in a crisis year) gives the cumulative probability of an involuntary job change.
- **New graduate entering cognitive work:** an "opportunity" index = openings relative to 2026 (the 19% entry-level gap measured in 2026, widening with deployment) times the real cognitive wage.
- **Licensed presence profession** (clinical medicine as the example): licensure caps substitution at 30% of tasks until 2034, rising to 50% by 2040; half of freed capacity meets unmet demand rather than reducing demand; demand grows 1.5% a year with aging; the trained workforce grows 1% a year.
- **Infrastructure engineer** (physical/compute side): pay tracks the AI capital-spending cycle with an elasticity of 0.2, so it rises in the buildout and falls in a crisis.

## Calibration anchors (2026)

| Quantity | Value used | Basis |
|---|---|---|
| Labor share of output | 57% | BLS nonfarm business labor share, mid-50s percent |
| Cognitive vs presence jobs | 42 / 58 | Judgment from occupational employment; cognitive jobs pay about 1.8x |
| Transfers / (wages + transfers) | 25% | BEA: government social benefits roughly a quarter the size of wages and salaries |
| Unemployment | 4.2% | BLS, mid-2026 |
| Entry-level employment gap in exposed occupations | 19% | Stanford Digital Economy Lab, August 2026 update of "Canaries in the Coal Mine" (secondary coverage; verify) |
| AI capital spending, share of GDP | 1.5% | Epoch AI, Q1 2026 (broad measure) |
| Social Security depletion and cut | late 2032, 22% | 2026 Trustees Report (secondary coverage; verify) |
| Real equity cash yield; real bond yield | 3.5%; 1.5% | Judgment: dividends plus buybacks at a neutral multiple; inflation-protected bonds |
| AI can do 25% of cognitive tasks at acceptable quality | 25% | Judgment |

## The Monte Carlo

Each of 400 draws samples the parameters every substitution scenario shares from uniform judgment ranges (`MC_RANGES`): the capability ceiling (0.65 to 0.95) and midpoint (2029 to 2035), the robotics ceiling and midpoint, adoption speed (0.15 to 0.40 a year), new-task creation (0.3 to 0.8 per displaced task) and the human share of new tasks (0.10 to 0.45 cognitive, 0.25 to 0.60 presence), the scale effect, labor demand elasticity (−1.0 to −0.4), mobility, the productivity gain, the reservation wage, licensure parameters, the three household mapping coefficients, the capital-spending growth rate, and the equity beta and yield. The loom comparator draws its own defining parameters from `LOOM_RANGES`. Each scenario keeps its offsets from the defaults (so fragmentation stays slower, diffusion stays more human-friendly, in every draw). All six scenarios are run on each draw and the weighted mixtures formed; percentiles are taken across draws.

**The ranges are not centered on the essay's central case.** They were chosen by the AI assistant that built the model, and they are centered on somewhat milder assumptions (for example, a mean human share of new cognitive tasks of 0.275 against the essay's 0.20, and a mean capability midpoint of 2032 against 2031). The essay's central run therefore sits in the lower half of each band: for office pay in managed dependency in 2036 it gives 56, against a Monte Carlo median of 72 with a 10th to 90th percentile range of 51 to 88. The letter uses the Monte Carlo medians and ranges and says so.

## Headline results

Medians with 10th to 90th percentile ranges across draws. "If the premise holds" is the conditional mixture; "all-in" includes the 25% on the old pattern. Full tables in `output/uncertainty.md`; central-parameter tables in `output/results.md`.

| Measure (2026 = 100 unless %) | Premise holds, 2031 | Premise holds, 2036 | Premise holds, 2040 | Old pattern, 2036 | All-in, 2036 |
|---|---|---|---|---|---|
| Real cognitive wage | 92 (81–99) | 75 (54–92) | 64 (49–84) | 115 (108–120) | 85 (68–98) |
| Real presence wage | 114 (111–118) | 124 (110–134) | 119 (100–137) | 137 (132–140) | 127 (116–135) |
| Not employed, share of workforce | 5% (5–6) | 7% (6–9) | 7% (6–13) | 4% | 6% (5–8) |
| Dependency ratio | 29% (27–32) | 37% (31–45) | 45% (37–54) | 24% | 34% (30–40) |
| Real household income, all households | 106 (104–109) | 115 (109–119) | 122 (115–129) | 126 (122–130) | 118 (113–122) |
| 60/40 portfolio, total return, real | 128 (119–134) | 155 (142–170) | 194 (173–219) | 146 (137–155) | 153 (141–167) |
| Retired couple income | 115 (111–118) | 127 (121–136) | 147 (137–161) | 126 (121–131) | 127 (121–134) |
| Incumbent professional who keeps the job, real pay | 100 (94–104) | 84 (67–99) | 71 (57–91) | 116 (111–121) | 92 (79–104) |
| P(incumbent involuntary exit by year) | 17% (13–25) | 39% (31–52) | 51% (41–62) | 21% (18–24) | 35% (28–45) |
| New graduate opportunity (openings × pay) | 82 (69–92) | 57 (38–76) | 44 (30–65) | 99 (88–107) | 68 (52–83) |
| Licensed presence profession, real pay | 109 (105–114) | 118 (109–128) | 128 (116–142) | 125 (120–131) | 120 (112–128) |
| Infrastructure engineer, real pay | 127 (120–135) | 140 (132–148) | 154 (144–163) | 145 (137–153) | 141 (133–149) |
| Retiree savings, all equity, 4% real withdrawals | 116 (104–125) | 125 (109–145) | 147 (122–178) | 112 (100–124) | 122 (107–140) |
| Retiree savings, equity + 3-year cash bucket | 112 (101–119) | 118 (104–136) | 135 (113–165) | 104 (94–114) | 115 (102–131) |

Crisis years in the central runs: compensated restructuring 2030, prolonged disorder 2030, capability diffusion 2033; managed dependency none (the dole arrives in time); durable fragmentation none (chronic shortfall instead). Equity price drawdowns from peak in the central runs: restructuring 45%, diffusion 42%, disorder 60%.

Four things the results show that are not obvious from the verbal argument:

1. **The pie grows in every scenario but disorder.** The problem is the split, not scarcity.
2. **Households as a whole are not poorer in the modal state.** In managed dependency, real household income is up about 20% by 2036 while the labor share falls from 57% to the mid-30s. The difference is paid by the state.
3. **The family is split across the line.** Asset owners do about as well whether the premise holds or not (the retired couple's 2036 income is 127 in both cases) and better in the modal state; wage earners in cognitive work do far worse if the premise holds; new entrants worst of all; licensed presence work and physical infrastructure stay close to trend.
4. **At a 4% real draw, no strategy runs out within the horizon in any scenario**, including all equity in disorder (66 at 2040 in the central run). The equity share is the real dial (higher expectation, deeper trough); the cash bucket costs about a tenth of the eventual value in good scenarios and raises the trough and the disorder outcome.

## Limits

- **One analyst's parameters, and one assistant's ranges.** The central values are the essay's judgments; the Monte Carlo ranges are the assistant's. The honest claim is about shape, ordering, and the width of the bands, not about levels.
- **The equity index is not the S&P 500.** It is listed earnings (national capital income with a beta of 0.6) times a stylized multiple plus a stylized yield. The real index is more concentrated and more valuation-driven, and can fall further and faster.
- **The hazard model is a mapping, not data.** The probability of an involuntary job change converts task displacement into job loss with a judgment coefficient (sampled 0.8 to 2.2). Read "39% (31 to 52) by 2036 if the premise holds" as "well above the old pattern's one in five", not as a measured rate.
- **Cognitive work is one pool.** In reality it is dozens of occupations with different exposure, licensure, and presence content; the average hides the spread. The licensed-profession and infrastructure archetypes are the only concessions to that. Splitting the pool (routine, expert-judgment, licensed, creative) and adding age cohorts is the most valuable next step.
- **The displaced-professional income measure is unreliable.** It spreads all new transfers over the non-employed, which overstates the dole per head; it is kept in the tables but not used in the letter.
- **The wage floor produces a kink.** When the reservation wage binds, employment is rationed instead of wages falling, and the lines bend. The mechanism is real; its sharpness is an artifact.
- **Transfers assume fiscal capacity.** Managed dependency assumes the state can collect 45% of AI rents plus 90% of lost wages. Whether it can is the essay's central political question, and the model assumes the answer.
- **No housing, no international sector, no explicit inflation, no monetary policy, no heterogeneity within households, no taxes on the household side.** The capital-spending cycle is stylized. The drawdown module ignores taxes, Medicare premiums, and health shocks, which for a household in its late seventies are the first-order risks.
- **The wildcard is not modeled.**
- **Sensitivities that matter most:** the human share of new tasks (the biggest lever; 0.8 gives the loom world), the labor demand elasticity (−0.4 roughly doubles wage declines relative to −0.7), the capability midpoint (timing), the rent-recycling share (whether managed dependency is stable), and, for retirees, the equity share and the crisis-year valuation cut.

## Running it

```
python3 two_economy.py          # point runs + Monte Carlo (~10 s): results.md, trajectories.csv, uncertainty.md/.csv, params.json
python3 two_economy.py --fast   # point runs only
python3 charts.py               # macro.png, households.png, mechanism.png, drawdown.png, fan.png
python3 game_theory.py          # game_theory.md
```

Requires Python 3.12+, numpy, pandas, matplotlib.
