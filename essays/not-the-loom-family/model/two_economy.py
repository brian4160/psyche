#!/usr/bin/env python3
"""
two_economy.py — an illustrative two-economy model of AI labor substitution, 2026–2040.

Companion to "Not the Loom" (B. Anderson, Oct 2026) and to the family essay in the
parent folder.

WHAT THIS IS
  A small, transparent simulation that turns the essay's verbal mechanism into numbers.
  A general substitute for labor enters; the price of substitutable labor falls toward
  the substitute's cost; displaced workers crowd into the work that stays human; the
  capital economy gains; household consumption depends on how much of the lost wage
  income is replaced by transfers (the "dole"); and an unreplaced demand hole can tip
  the system into crisis. Each of the essay's modeled terminal states is a parameter
  set. A "loom" comparator shows what the standard reassurance assumes.

WHAT THIS IS NOT
  A forecast. Every parameter is a judgment, stated in the Params dataclass so it can be
  rejected or changed. Magnitudes are meant to show shape and relative ordering across
  scenarios, not point estimates. Units: 2026 output = 100; 2026 jobs = 100. "Real"
  means deflated by a consumer price index that falls when AI gains are competed into
  prices rather than captured as rents. Wages carry trend productivity growth of 1.5%
  a year in every scenario, so a flat real-wage line already means "falling behind".

Run:  python3 two_economy.py
      writes CSV, JSON and markdown tables to ./output  (charts: see charts.py)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
YEARS = list(range(2026, 2041))

# Posteriors from the essay's scenario table (terminal states at 2040). The wildcard
# (AI as an independent agent class, 8%) is not modeled numerically; the five modeled
# states are renormalized when computing probability-weighted outcomes.
ESSAY_POSTERIOR = {
    "managed_dependency": 0.37,
    "durable_fragmentation": 0.20,
    "compensated_restructuring": 0.13,
    "capability_diffusion": 0.11,
    "prolonged_disorder": 0.11,
}

# ----------------------------------------------------------------------------------
# 2026 calibration anchors (rounded; sources in MODEL.md)
# ----------------------------------------------------------------------------------
Y0 = 100.0            # output index
LABOR_SHARE0 = 0.57   # labor income / output
W0 = Y0 * LABOR_SHARE0
N_COG = 42.0          # cognitive / office / professional jobs (index, of 100 total)
N_PHYS = 58.0         # physical, presence, care, trades, in-person service jobs
S_COG = 0.567         # share of labor income earned in cognitive jobs
S_PHYS = 1 - S_COG
W_COG0 = W0 * S_COG / N_COG     # ~0.77 per job
W_PHYS0 = W0 * S_PHYS / N_PHYS  # ~0.43 per job (cognitive jobs pay ~1.8x)
W_AVG0 = W0 / (N_COG + N_PHYS)
DEP0 = 0.25           # transfers / (wages + transfers) for households, 2026
T0 = W0 * DEP0 / (1 - DEP0)     # ~19
U0 = 0.042            # unemployment rate, 2026
ENTRY_GAP0 = 0.19     # entry-level employment gap in AI-exposed occupations (Stanford, Aug 2026)
CAPEX_SHARE0 = 0.015  # AI-related investment, share of GDP, 2026 (Epoch AI, broad measure)
R_BASE0 = 5.0         # pre-existing tech rents (index) so an AI-rents index is defined


@dataclass
class Params:
    key: str
    name: str
    description: str

    # --- technology: capability frontier (logistic in time) ---------------------------
    # fraction of tasks in each family that frontier AI (or robotics) can perform at
    # acceptable quality. 2026 anchors and ceilings are judgments.
    cog_cap_2026: float = 0.25
    cog_cap_max: float = 0.85
    cog_cap_mid: float = 2031.0
    cog_cap_k: float = 0.45
    phys_cap_2026: float = 0.05
    phys_cap_max: float = 0.60
    phys_cap_mid: float = 2037.0
    phys_cap_k: float = 0.40

    # --- adoption: deployed substitution closes the gap to capability at this rate -----
    adoption_rate: float = 0.30
    deployed_2026_cog: float = 0.10
    deployed_2026_phys: float = 0.03

    # --- new tasks (Acemoglu–Restrepo reinstatement) -----------------------------------
    new_task_rate: float = 0.5        # new tasks created per unit of displaced tasks
    new_task_share_cog: float = 0.20  # share of those new tasks performed by humans
    new_task_share_phys: float = 0.35
    scale_effect: float = 0.4         # human task demand rises with output^(scale_effect)

    # --- labor market -------------------------------------------------------------------
    demand_elasticity: float = -0.7   # labor demand elasticity wrt the wage
    mobility: float = 0.25            # share of excess cognitive labor moving to presence work / yr
    wage_adjust: float = 0.5          # speed of wage adjustment toward market clearing
    reservation_wage: float = 0.45    # floor, relative to the (trend-growing) average wage
    reservation_from_transfers: float = 0.3  # new transfers raise the floor by this x (T_new/W0)
    labor_growth: float = 0.003
    presence_demand_growth: float = 0.012    # aging and income effects on presence work
    cog_demand_growth: float = 0.005
    wage_ceiling: float = 1.15        # wages can rise at most this much above trend in a tight market

    # --- productivity -------------------------------------------------------------------
    base_growth: float = 0.015
    phi_cog: float = 0.50   # potential-output gain if all cognitive tasks were substituted
    phi_phys: float = 0.40
    rent_capture: float = 0.70  # share of AI gains captured as concentrated rents (rest -> prices)

    # --- transfers (the dole) -----------------------------------------------------------
    transfer_response: float = 0.0  # share of lost labor income replaced by new transfers
    transfer_lag: int = 2
    dependency_cap: float = 0.60    # political ceiling on transfers / (wages + transfers)
    rent_recycling: float = 0.10    # share of AI rents the state recycles to households as transfers
    transfer_funding_from_capital: float = 0.6  # remainder is borrowed

    # --- demand feedback and crisis -----------------------------------------------------
    mpc_labor: float = 0.95
    mpc_capital: float = 0.40
    demand_feedback: float = 0.6    # share of a demand gap that shows up as lost output
    crisis_gap: float = 0.05        # demand gap (two consecutive years) that triggers a crisis
    crisis_year_forced: Optional[int] = None
    crisis_shock: float = 0.06      # extra output loss in the crisis year
    crisis_persistence: float = 0.0 # share of the shock that never recovers (used by "none")
    crisis_response: str = "none"   # none | dole | buyout

    # --- AI capital spending cycle (demand injection and the "picks and shovels") -------
    capex_growth: float = 0.20
    capex_plateau_year: int = 2030
    capex_after_plateau: float = 0.03

    # --- valuation ----------------------------------------------------------------------
    multiple_path: str = "neutral"  # neutral | exuberant | compressed

    # --- Social Security ----------------------------------------------------------------
    ss_cut_2033: float = 0.0        # 2026 Trustees Report: ~22% automatic cut at depletion, late 2032

    # --- licensed presence profession (e.g. clinical medicine) -------------------------
    license_cap_early: float = 0.30  # max substitution the license regime permits before 2034
    license_cap_late: float = 0.50   # ... by 2040
    license_absorption: float = 0.5  # share of freed capacity that reduces demand (rest meets unmet need)


def anchored_logistic(t: float, v2026: float, vmax: float, mid: float, k: float) -> float:
    """Logistic rising to vmax, shifted so that its 2026 value equals v2026."""
    g0 = 1.0 / (1.0 + math.exp(-k * (2026 - mid)))
    low = (v2026 - vmax * g0) / (1.0 - g0)
    g = 1.0 / (1.0 + math.exp(-k * (t - mid)))
    return low + (vmax - low) * g


def clear_market(S: float, H_pot: float, w0: float, w_prev: float, w_floor: float, p: Params):
    """One labor market. Returns (wage, employment, excess supply)."""
    eta = p.demand_elasticity
    ratio = S / max(H_pot, 1e-9)
    w_star = w0 * ratio ** (1.0 / eta)          # market-clearing wage
    w_star = min(w_star, w0 * p.wage_ceiling)   # bounded above
    w_star = max(w_star, w_floor)               # bounded below by the reservation wage
    w = w_prev + p.wage_adjust * (w_star - w_prev)
    w = max(w, w_floor)
    E = min(S, H_pot * (w / w0) ** eta)
    return w, E, S - E


def multiple_base(p: Params, i: int) -> float:
    if p.multiple_path == "exuberant":
        return 1.0 + 0.15 * min(1.0, i / 3.0)
    if p.multiple_path == "compressed":
        return 1.0 - 0.20 * min(1.0, i / 6.0)
    return 1.0


def run(p: Params) -> pd.DataFrame:
    rows = []
    g = p.base_growth
    r = p.new_task_rate
    D_c, D_p = p.deployed_2026_cog, p.deployed_2026_phys
    D_c0, D_p0 = D_c, D_p
    S_c, S_p = N_COG, N_PHYS
    w_c, w_p = W_COG0, W_PHYS0
    T = T0
    transfer_response = p.transfer_response
    dependency_cap = p.dependency_cap
    rent_recycling = p.rent_recycling
    demand_feedback = p.demand_feedback
    crisis_year: Optional[int] = None
    crisis_mult = 1.0
    gap_hist = [0.0]
    lost_hist = [0.0]
    public_equity_share = 0.0
    ai_haircut = 1.0
    bond = 1.0
    ss = 1.0
    capex = 1.0
    cum_survival = 1.0

    def task_factor(D: float, D0: float, nu: float) -> float:
        return (1 - D + nu * r * D) / (1 - D0 + nu * r * D0)

    for i, t in enumerate(YEARS):
        trend = (1 + g) ** i

        # --- technology ---------------------------------------------------------------
        cap_c = anchored_logistic(t, p.cog_cap_2026, p.cog_cap_max, p.cog_cap_mid, p.cog_cap_k)
        cap_p = anchored_logistic(t, p.phys_cap_2026, p.phys_cap_max, p.phys_cap_mid, p.phys_cap_k)
        D_c_prev = D_c
        if i > 0:
            D_c = D_c + p.adoption_rate * (cap_c - D_c)
            D_p = D_p + p.adoption_rate * (cap_p - D_p)
        dD_net = max(0.0, D_c - D_c_prev) * (1 - p.new_task_share_cog * r)

        # --- potential output and AI gains ---------------------------------------------
        gain = p.phi_cog * S_COG * (D_c - D_c0) + p.phi_phys * S_PHYS * (D_p - D_p0)
        Y_pot = Y0 * trend * (1 + gain)
        G_ai = Y_pot - Y0 * trend
        price = 1 - (1 - p.rent_capture) * G_ai / Y_pot   # competed gains lower consumer prices
        scale = (1 + gain) ** p.scale_effect               # more output -> more human tasks demanded

        # --- AI capital spending cycle -------------------------------------------------
        if i > 0:
            g_capex = p.capex_growth if t <= p.capex_plateau_year else p.capex_after_plateau
            if crisis_year is not None:
                yrs = t - crisis_year
                if yrs == 1:
                    g_capex = -0.45
                elif yrs > 1:
                    g_capex = 0.10 if p.crisis_response in ("dole", "buyout") else -0.02
            capex *= (1 + g_capex)
        capex_demand = CAPEX_SHARE0 * Y0 * trend * (capex - 1)   # investment above its 2026 trend share

        # --- human labor demand at trend wages (jobs) ------------------------------------
        H_c = N_COG * task_factor(D_c, D_c0, p.new_task_share_cog) * (1 + p.cog_demand_growth) ** i * scale
        H_p = N_PHYS * task_factor(D_p, D_p0, p.new_task_share_phys) * (1 + p.presence_demand_growth) ** i * scale

        # --- wages and employment -------------------------------------------------------
        w_c0_t, w_p0_t = W_COG0 * trend, W_PHYS0 * trend
        T_new = max(0.0, T - T0 * trend)
        w_floor = p.reservation_wage * W_AVG0 * trend * (1 + p.reservation_from_transfers * T_new / W0)
        w_c, E_c, X_c = clear_market(S_c, H_c, w_c0_t, w_c * (1 + g) if i > 0 else w_c, w_floor, p)
        w_p, E_p, X_p = clear_market(S_p, H_p, w_p0_t, w_p * (1 + g) if i > 0 else w_p, w_floor, p)
        L = S_c + S_p
        E = E_c + E_p
        not_employed = U0 + (L - E) / L * (1 - U0)   # frictional baseline plus structural excess
        W = w_c * E_c + w_p * E_p
        W_ref = W0 * trend
        lost = max(0.0, W_ref - W)
        lost_hist.append(lost)

        # --- transfers (policy rule with lag, ceiling, gradual adjustment) -------------
        lag_idx = max(0, len(lost_hist) - 1 - p.transfer_lag)
        R_pot = p.rent_capture * G_ai + R_BASE0 * trend
        T_target = T0 * trend + transfer_response * lost_hist[lag_idx] + rent_recycling * (R_pot - R_BASE0 * trend)
        T_target = min(T_target, dependency_cap / (1 - dependency_cap) * W)
        T = T + 0.5 * (T_target - T)
        T_new = max(0.0, T - T0 * trend)

        # --- demand feedback (fixed point) ---------------------------------------------
        A0 = Y0 - p.mpc_labor * (W0 + T0) - p.mpc_capital * (Y0 - W0)
        A = A0 * trend + capex_demand
        Y = Y_pot * crisis_mult
        for _ in range(10):
            K_gross = Y * price - W            # nominal: real output at the (lower) price level
            cap_tax = p.transfer_funding_from_capital * T_new
            dividend = public_equity_share * R_pot * (Y / Y_pot)
            K_net = K_gross - cap_tax - dividend
            AD = p.mpc_labor * (W + T + dividend) + p.mpc_capital * max(K_net, 0.0) + A
            gap = max(0.0, 1 - AD / (Y_pot * price))
            Y = Y_pot * (1 - demand_feedback * gap) * crisis_mult
        gap_hist.append(gap)

        # --- crisis trigger and response -----------------------------------------------
        triggered = False
        if crisis_year is None:
            if p.crisis_year_forced is not None and t == p.crisis_year_forced:
                triggered = True
            elif len(gap_hist) >= 3 and gap_hist[-1] > p.crisis_gap and gap_hist[-2] > 0.6 * p.crisis_gap:
                triggered = True
        if triggered:
            crisis_year = t
            crisis_mult = 1 - p.crisis_shock
            Y *= crisis_mult
            if p.crisis_response == "dole":
                transfer_response = max(transfer_response, 0.75)
                dependency_cap = max(dependency_cap, 0.65)
                rent_recycling = max(rent_recycling, 0.35)
            elif p.crisis_response == "buyout":
                transfer_response = max(transfer_response, 0.60)
                dependency_cap = max(dependency_cap, 0.60)
                rent_recycling = max(rent_recycling, 0.15)
                public_equity_share = 0.35
                ai_haircut = 0.80  # private holders compensated in part (the British model)
            else:  # none: fiscal stress, failure to respond
                transfer_response = min(transfer_response, 0.15)
                dependency_cap = min(dependency_cap, 0.40)
                rent_recycling = min(rent_recycling, 0.05)
                demand_feedback = 0.8
                bond *= 0.90
        elif crisis_year is not None:
            yrs = t - crisis_year
            if p.crisis_response in ("dole", "buyout"):
                crisis_mult = 1 - p.crisis_shock * max(0.0, 1 - yrs / 3.0)
            else:
                crisis_mult = 1 - p.crisis_shock * max(p.crisis_persistence, 1 - yrs / 8.0)

        R = R_pot * (Y / Y_pot)   # rents fall with realized output: no customers, no rents
        K_gross = Y * price - W
        cap_tax = p.transfer_funding_from_capital * T_new
        dividend = public_equity_share * R
        K_net = K_gross - cap_tax - dividend
        household_income = W + T + dividend
        dep_ratio = (T + dividend) / household_income

        # --- valuation ------------------------------------------------------------------
        M = multiple_base(p, i)
        if crisis_year is not None:
            yrs = t - crisis_year
            if p.crisis_response in ("dole", "buyout"):
                M *= 0.60 + 0.35 * min(1.0, yrs / 4.0)
            else:
                M *= 0.50
        broad = (K_net / (Y0 - W0)) * M / price
        ai_rents_index = (R / R_BASE0) * M * ai_haircut / price
        bond = bond * 1.01
        portfolio = 0.6 * broad + 0.4 * bond

        # --- Social Security ------------------------------------------------------------
        if t == 2033 and p.ss_cut_2033 > 0:
            ss = 1 - p.ss_cut_2033

        # --- household archetypes -------------------------------------------------------
        # Retired couple: ~55% Social Security, ~45% portfolio draw (as % of value)
        retiree_income = (0.55 * ss + 0.45 * portfolio) / price
        care_cost = (w_p / W_PHYS0) / price

        # Mid-career cognitive incumbent: protected by attrition early, exposed later
        prot = math.exp(-i / 6.0)
        rel = w_c / w_c0_t
        incumbent_wage = (1 - (1 - prot) * (1 - rel)) * trend / price
        hazard = 0.015 + (1 - prot) * dD_net * 1.5
        if crisis_year is not None and t - crisis_year in (0, 1):
            hazard += 0.06 if t == crisis_year else 0.03
        cum_survival *= (1 - hazard)
        dole_per_head = min(0.6 * w_c0_t, T_new / max(1.0, L - E)) if (L - E) > 0 else 0.0
        displaced_income = ((0.55 * (w_p / w_p0_t) * 0.9 * trend) + dole_per_head / W_COG0) / price

        # Entrant into cognitive work (new graduate): openings relative to 2026 x real wage
        entry_gap = min(0.60, ENTRY_GAP0 + 0.4 * max(0.0, D_c - D_c0))
        entrant = ((1 - entry_gap) / (1 - ENTRY_GAP0)) * rel * trend / price

        # Licensed presence profession (clinical medicine): licensure caps substitution,
        # unmet demand absorbs part of the freed capacity, aging raises demand
        cap_lic = p.license_cap_early if t < 2034 else p.license_cap_early + (p.license_cap_late - p.license_cap_early) * (t - 2034) / 6.0
        D_lic = min(0.5 * D_c, cap_lic)
        D_lic_net = D_lic * (1 - 0.2 * r)
        D_lic0_net = min(0.5 * D_c0, p.license_cap_early) * (1 - 0.2 * r)
        H_lic = (1 - p.license_absorption * D_lic_net) / (1 - p.license_absorption * D_lic0_net) * (1.015) ** i * scale
        S_lic = 1.01 ** i
        w_lic = min(p.wage_ceiling, (S_lic / H_lic) ** (1.0 / p.demand_elasticity))
        if crisis_year is not None and p.crisis_response == "none":
            w_lic *= 0.85
        licensed = w_lic * trend / price

        # Infrastructure engineer: wage tracks the AI capex cycle
        infra = min(1.4, capex ** 0.2) * trend / price

        rows.append(dict(
            scenario=p.key, scenario_name=p.name, year=t,
            cap_cog=cap_c, cap_phys=cap_p, deployed_cog=D_c, deployed_phys=D_p,
            jobs_cog_demand=H_c / N_COG, jobs_phys_demand=H_p / N_PHYS, supply_cog=S_c, supply_phys=S_p,
            emp_cog=E_c, emp_phys=E_p, not_employed=not_employed,
            wage_cog_rel_trend=rel, wage_phys_rel_trend=w_p / w_p0_t,
            real_wage_cog=w_c / W_COG0 / price, real_wage_phys=w_p / W_PHYS0 / price,
            labor_income=W, labor_income_ref=W_ref, transfers=T, dividend=dividend,
            household_income=household_income, real_household_income=household_income / price / (W0 + T0),
            dependency_ratio=dep_ratio, output=Y, potential_output=Y_pot, demand_gap=gap,
            labor_share=W / (Y * price), capital_income_net=K_net, ai_rents=R, ai_rents_share=R / (Y * price), price_level=price,
            capex_index=capex,
            crisis=int(crisis_year is not None and t >= crisis_year), crisis_year=crisis_year or 0,
            valuation_multiple=M, broad_equity=broad, ai_rents_index=ai_rents_index, bond=bond,
            portfolio_60_40=portfolio, social_security=ss,
            retiree_income=retiree_income, care_cost=care_cost,
            incumbent_wage=incumbent_wage, incumbent_hazard=hazard, incumbent_survival=cum_survival,
            displaced_income=displaced_income, entrant_index=entrant, entry_gap=entry_gap,
            licensed_index=licensed, infra_index=infra,
        ))

        # --- flows for next year --------------------------------------------------------
        movers = p.mobility * X_c
        S_c = (S_c - movers) * (1 + p.labor_growth)
        S_p = (S_p + movers) * (1 + p.labor_growth)

    return pd.DataFrame(rows)


SCENARIOS = [
    Params(
        key="loom", name="Comparator: the loom",
        description="Capability plateaus well short of a general substitute and most new tasks go to humans. "
                    "This is what the standard reassurance (steam, combustion, transistor) assumes.",
        cog_cap_max=0.45, cog_cap_mid=2030.0, phys_cap_max=0.25,
        new_task_rate=0.9, new_task_share_cog=0.80, new_task_share_phys=0.80,
        rent_capture=0.50, transfer_response=0.30, crisis_response="dole",
    ),
    Params(
        key="managed_dependency", name="Managed dependency",
        description="Nationalist-oligarchic consolidation with a dole. Transfers expand quickly enough to close most "
                    "of the consumption hole; rents stay concentrated; asset values rise.",
        transfer_response=0.90, transfer_lag=1, dependency_cap=0.65, rent_recycling=0.45,
        rent_capture=0.75, multiple_path="exuberant", crisis_response="dole",
    ),
    Params(
        key="durable_fragmentation", name="Durable fragmentation",
        description="Unresolved at 2040. Adoption is slower and patchier (state-level friction), transfers are "
                    "partial and capped by politics, a chronic demand shortfall persists without acute collapse.",
        adoption_rate=0.22, transfer_response=0.45, transfer_lag=2, dependency_cap=0.45, rent_recycling=0.15,
        rent_capture=0.65, multiple_path="neutral", crisis_response="dole", crisis_gap=1.0,
        ss_cut_2033=0.10,
    ),
    Params(
        key="compensated_restructuring", name="Compensated restructuring",
        description="The British buyout. Transfers lag, a crisis arrives, and the response is public equity in the "
                    "AI value chain paid out as dividends, with private holders partly compensated.",
        transfer_response=0.25, transfer_lag=3, dependency_cap=0.55,
        rent_capture=0.70, multiple_path="exuberant", crisis_response="buyout",
    ),
    Params(
        key="capability_diffusion", name="Capability diffusion",
        description="Open weights and cheap silicon flatten the gradient. Most AI gains are competed into lower "
                    "prices rather than captured as rents; more new tasks are run by humans with cheap AI.",
        rent_capture=0.30, new_task_share_cog=0.35, new_task_share_phys=0.45,
        transfer_response=0.40, transfer_lag=2, dependency_cap=0.50, rent_recycling=0.15,
        multiple_path="compressed", crisis_response="dole", ss_cut_2033=0.05,
    ),
    Params(
        key="prolonged_disorder", name="Prolonged disorder",
        description="Crisis without reconstruction. Transfers lag, the demand hole opens, the shock overwhelms "
                    "institutional capacity, and nothing coherent replaces it; Social Security takes its scheduled cut.",
        transfer_response=0.20, transfer_lag=3, dependency_cap=0.45,
        rent_capture=0.75, multiple_path="exuberant", crisis_response="none",
        crisis_shock=0.12, crisis_persistence=0.7, ss_cut_2033=0.22,
    ),
]


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def fmt_idx(x: float) -> str:
    return f"{100 * x:.0f}"


def summary_tables(df: pd.DataFrame) -> str:
    lines = []
    for yr in (2031, 2036, 2040):
        lines.append(f"\n### Macro picture, {yr} (2026 = 100 unless a percentage)\n")
        lines.append("| Scenario | AI can do (cognitive tasks) | Deployed | Human cognitive jobs demanded | Real cognitive wage | Real presence wage | Not employed | Labor share | AI rents, share of output | Dependency ratio | Real household income | Output | Broad equity (real) | Crisis year |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in SCENARIOS:
            d = df[(df.scenario == p.key) & (df.year == yr)].iloc[0]
            lines.append(
                f"| {p.name} | {fmt_pct(d.cap_cog)} | {fmt_pct(d.deployed_cog)} | {fmt_idx(d.jobs_cog_demand)} | "
                f"{fmt_idx(d.real_wage_cog)} | {fmt_idx(d.real_wage_phys)} | {fmt_pct(d.not_employed)} | "
                f"{fmt_pct(d.labor_share)} | {fmt_pct(d.ai_rents_share)} | {fmt_pct(d.dependency_ratio)} | "
                f"{fmt_idx(d.real_household_income)} | {d.output:.0f} | {fmt_idx(d.broad_equity)} | "
                f"{int(d.crisis_year) or '—'} |"
            )
    return "\n".join(lines)


def household_tables(df: pd.DataFrame) -> str:
    lines = ["\n### Household archetypes (real income or opportunity index, 2026 = 100)\n",
             "Trend productivity growth of 1.5% a year is included, so a flat line means falling behind the economy as a whole.\n"]
    lines.append("| Scenario | Retired couple 2033 | Retired couple 2040 | 60/40 portfolio 2033 | 60/40 portfolio 2040 | Social Security 2040 | Care-labor cost 2040 | Incumbent professional wage 2036 | P(involuntary exit by 2036) | Displaced professional income 2036 | New graduate (cognitive) 2030 | New graduate 2034 | New graduate 2038 | Licensed presence profession 2034 | Licensed 2040 | Infrastructure engineer 2030 | Infra 2036 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in SCENARIOS:
        d = df[df.scenario == p.key].set_index("year")
        lines.append(
            f"| {p.name} | {fmt_idx(d.loc[2033, 'retiree_income'])} | {fmt_idx(d.loc[2040, 'retiree_income'])} | "
            f"{fmt_idx(d.loc[2033, 'portfolio_60_40'])} | {fmt_idx(d.loc[2040, 'portfolio_60_40'])} | "
            f"{fmt_idx(d.loc[2040, 'social_security'])} | {fmt_idx(d.loc[2040, 'care_cost'])} | "
            f"{fmt_idx(d.loc[2036, 'incumbent_wage'])} | {fmt_pct(1 - d.loc[2036, 'incumbent_survival'])} | "
            f"{fmt_idx(d.loc[2036, 'displaced_income'])} | "
            f"{fmt_idx(d.loc[2030, 'entrant_index'])} | {fmt_idx(d.loc[2034, 'entrant_index'])} | {fmt_idx(d.loc[2038, 'entrant_index'])} | "
            f"{fmt_idx(d.loc[2034, 'licensed_index'])} | {fmt_idx(d.loc[2040, 'licensed_index'])} | "
            f"{fmt_idx(d.loc[2030, 'infra_index'])} | {fmt_idx(d.loc[2036, 'infra_index'])} |"
        )
    return "\n".join(lines)


def weighted(df: pd.DataFrame) -> pd.DataFrame:
    keys = list(ESSAY_POSTERIOR)
    w = np.array([ESSAY_POSTERIOR[k] for k in keys])
    w = w / w.sum()
    num = [c for c in df.select_dtypes("number").columns if c != "year"]
    parts = []
    for k, wk in zip(keys, w):
        d = df[df.scenario == k].set_index("year")[num] * wk
        parts.append(d)
    exp = sum(parts)
    exp["scenario"] = "probability_weighted"
    exp["scenario_name"] = "Probability-weighted (essay posteriors, five modeled states)"
    return exp.reset_index()


def expectation_table(df: pd.DataFrame, exp: pd.DataFrame) -> str:
    lines = ["\n### Probability-weighted expectation and range across the five modeled states\n"]
    lines.append("Weights: essay posteriors renormalized over the five modeled states (the 8% wildcard is excluded). "
                 "The range is the lowest and highest scenario value, not a confidence interval.\n")
    lines.append("| Measure | 2031 expected | 2031 range | 2036 expected | 2036 range | 2040 expected | 2040 range |")
    lines.append("|---|---|---|---|---|---|---|")
    measures = [
        ("Real cognitive wage (index)", "real_wage_cog", fmt_idx),
        ("Real presence wage (index)", "real_wage_phys", fmt_idx),
        ("Not employed, share of workforce", "not_employed", fmt_pct),
        ("Dependency ratio", "dependency_ratio", fmt_pct),
        ("Real household income, all households (index)", "real_household_income", fmt_idx),
        ("Labor share of output", "labor_share", fmt_pct),
        ("AI rents, share of output", "ai_rents_share", fmt_pct),
        ("Output (index)", "output", lambda x: f"{x:.0f}"),
        ("Broad equity, real (index)", "broad_equity", fmt_idx),
        ("60/40 portfolio, real (index)", "portfolio_60_40", fmt_idx),
        ("Retired couple income (index)", "retiree_income", fmt_idx),
        ("Incumbent professional wage (index)", "incumbent_wage", fmt_idx),
        ("P(incumbent involuntary exit by year)", None, fmt_pct),
        ("New graduate opportunity (index)", "entrant_index", fmt_idx),
        ("Licensed presence profession (index)", "licensed_index", fmt_idx),
        ("Infrastructure engineer (index)", "infra_index", fmt_idx),
    ]
    modeled = df[df.scenario.isin(ESSAY_POSTERIOR)]
    for label, col, f in measures:
        cells = [label]
        for yr in (2031, 2036, 2040):
            if col is None:
                e = 1 - exp[exp.year == yr]["incumbent_survival"].iloc[0]
                vals = 1 - modeled[modeled.year == yr]["incumbent_survival"]
            else:
                e = exp[exp.year == yr][col].iloc[0]
                vals = modeled[modeled.year == yr][col]
            cells += [f(e), f"{f(vals.min())} to {f(vals.max())}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    frames = [run(p) for p in SCENARIOS]
    df = pd.concat(frames, ignore_index=True)
    exp = weighted(df)
    df_all = pd.concat([df, exp], ignore_index=True)
    df_all.to_csv(os.path.join(OUT, "trajectories.csv"), index=False)
    with open(os.path.join(OUT, "params.json"), "w") as fh:
        json.dump([asdict(p) for p in SCENARIOS], fh, indent=2)

    md = ["# Two-economy model: results\n",
          "Generated by `two_economy.py`. Illustrative, not a forecast; see MODEL.md for assumptions.\n"]
    md.append("\n## Scenarios\n")
    for p in SCENARIOS:
        md.append(f"- **{p.name}** — {p.description}")
    md.append(summary_tables(df))
    md.append(household_tables(df))
    md.append(expectation_table(df, exp))
    with open(os.path.join(OUT, "results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
