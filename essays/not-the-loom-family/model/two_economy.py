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
  set. A "loom" comparator shows what the standard reassurance assumes, and carries the
  probability that the essay's premise is wrong (P_PREMISE_WRONG).

  Three layers of output:
    1. Point runs of each scenario with the central parameters (results.md, trajectories.csv).
    2. A Monte Carlo over the shared technology and labor-market parameters, reporting
       10th/50th/90th percentiles of each measure, both per scenario and for the
       probability-weighted "all-in" mixture (uncertainty.md, uncertainty.csv).
    3. A retiree drawdown module: what 4% real withdrawals do to savings under three
       strategies (all equity; 60/40; equity plus a three-year cash bucket).

WHAT THIS IS NOT
  A forecast. Every parameter is a judgment, stated in the Params dataclass so it can be
  rejected or changed. The Monte Carlo ranges are judgments too. Magnitudes show shape
  and ordering; the percentiles show how much the shape depends on the judgments.
  Units: 2026 output = 100; 2026 jobs = 100. "Real" means deflated by a consumer price
  index that falls when AI gains are competed into prices rather than captured as rents.
  Wages carry trend productivity growth of 1.5% a year in every scenario, so a flat
  real-wage line already means "falling behind".

Run:  python3 two_economy.py           (point runs + Monte Carlo; ~20 s)
      python3 two_economy.py --fast    (point runs only)
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict, replace
from typing import Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
YEARS = list(range(2026, 2041))

# Posteriors from the essay's scenario table (terminal states at 2040), CONDITIONAL on the
# essay's premise (a general substitute arrives within 10 to 15 years). The wildcard
# (AI as an independent agent class, 8%) is not modeled numerically; the five modeled
# states are renormalized.
ESSAY_POSTERIOR = {
    "managed_dependency": 0.37,
    "durable_fragmentation": 0.20,
    "compensated_restructuring": 0.13,
    "capability_diffusion": 0.11,
    "prolonged_disorder": 0.11,
}
# The letter author's judgment that the premise is wrong and the standard precedent holds
# (capability plateaus; new tasks absorb displaced labor). Represented by the loom comparator.
P_PREMISE_WRONG = 0.25


def all_in_weights() -> dict:
    s = sum(ESSAY_POSTERIOR.values())
    w = {k: (1 - P_PREMISE_WRONG) * v / s for k, v in ESSAY_POSTERIOR.items()}
    w["loom"] = P_PREMISE_WRONG
    return w


def conditional_weights() -> dict:
    s = sum(ESSAY_POSTERIOR.values())
    return {k: v / s for k, v in ESSAY_POSTERIOR.items()}


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
WITHDRAWAL = 4.0      # retiree drawdown: 4% of starting savings a year, real, constant
BUCKET_YEARS = 3      # cash bucket strategy: years of withdrawals held in cash
BOND_REAL_RETURN = 0.015   # inflation-protected bond, real
CASH_REAL_RETURN = 0.0     # cash, real


@dataclass
class Params:
    key: str
    name: str
    description: str

    # --- technology: capability frontier (logistic in time) ---------------------------
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

    # --- listed equity: how national capital income maps to the index -------------------
    equity_beta: float = 0.6         # share of a change in national capital income that reaches listed earnings
    equity_yield: float = 0.035      # real cash yield (dividends + buybacks) at a neutral multiple

    # --- household mappings (judgment coefficients, sampled in the Monte Carlo) --------
    entry_gap_slope: float = 0.4     # how fast the entry-level gap widens with deployment
    hazard_slope: float = 1.5        # involuntary-exit hazard per unit of net task displacement
    protection_years: float = 6.0    # e-folding time of incumbents' attrition protection


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


def add_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Retiree savings under constant real withdrawals (WITHDRAWAL per year from 100).

    all_equity: everything in the broad equity index.
    60/40:      rebalanced yearly between equity and the bond index.
    bucket:     equity plus BUCKET_YEARS of withdrawals in cash; withdrawals come from cash
                whenever equities are below their previous peak, and the bucket is refilled
                from equities when they make a new peak. Cash earns nothing in real terms.
    """
    broad = df.broad_equity.to_numpy()          # price index, used for the peak rule
    tr = df.equity_total_return.to_numpy()      # total return, used for growth
    bond = df.bond.to_numpy()
    eq_ret = np.diff(tr) / tr[:-1]
    bond_ret = np.diff(bond) / bond[:-1]

    v = 100.0
    path_a = [v]
    for r in eq_ret:
        v = max(0.0, (v - WITHDRAWAL) * (1 + r))
        path_a.append(v)

    v = 100.0
    path_b = [v]
    for re_, rb in zip(eq_ret, bond_ret):
        v = max(0.0, (v - WITHDRAWAL) * (1 + 0.6 * re_ + 0.4 * rb))
        path_b.append(v)

    bucket_size = WITHDRAWAL * BUCKET_YEARS
    eq, cash = 100.0 - bucket_size, bucket_size
    peak = broad[0]
    path_c = [eq + cash]
    for i, r in enumerate(eq_ret):
        if broad[i] < peak * 0.999 and cash >= WITHDRAWAL:
            cash -= WITHDRAWAL
        else:
            eq -= WITHDRAWAL
        eq = max(0.0, eq * (1 + r))
        cash *= (1 + CASH_REAL_RETURN)
        if broad[i + 1] >= peak:
            peak = broad[i + 1]
            refill = min(bucket_size - cash, eq)
            eq -= refill
            cash += refill
        path_c.append(eq + cash)

    df["draw_all_equity"] = path_a
    df["draw_60_40"] = path_b
    df["draw_bucket"] = path_c
    return df


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
    broad_prev = None
    tr = 1.0

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
        equity_earnings = 1 + p.equity_beta * (K_net / (Y0 - W0) - 1)   # listed earnings track capital income with beta < 1
        broad = equity_earnings * M / price                               # real price index
        if broad_prev is not None:
            tr = tr * (broad / broad_prev) * (1 + p.equity_yield / M)     # total return: price + cash yield (yield rises as the multiple falls)
        broad_prev = broad
        ai_rents_index = (R / R_BASE0) * M * ai_haircut / price
        bond = bond * (1 + BOND_REAL_RETURN)
        portfolio = 0.6 * tr + 0.4 * bond          # total-return 60/40, nothing withdrawn
        portfolio_80_20 = 0.8 * tr + 0.2 * bond

        # --- Social Security ------------------------------------------------------------
        if t == 2033 and p.ss_cut_2033 > 0:
            ss = 1 - p.ss_cut_2033

        # --- household archetypes -------------------------------------------------------
        retiree_income = (0.55 * ss + 0.45 * portfolio) / price
        care_cost = (w_p / W_PHYS0) / price

        prot = math.exp(-i / p.protection_years)
        rel = w_c / w_c0_t
        incumbent_wage = (1 - (1 - prot) * (1 - rel)) * trend / price
        hazard = 0.015 + (1 - prot) * dD_net * p.hazard_slope
        if crisis_year is not None and t - crisis_year in (0, 1):
            hazard += 0.06 if t == crisis_year else 0.03
        cum_survival *= (1 - hazard)
        dole_per_head = min(0.6 * w_c0_t, T_new / max(1.0, L - E)) if (L - E) > 0 else 0.0
        displaced_income = ((0.55 * (w_p / w_p0_t) * 0.9 * trend) + dole_per_head / W_COG0) / price

        entry_gap = min(0.60, ENTRY_GAP0 + p.entry_gap_slope * max(0.0, D_c - D_c0))
        entrant = ((1 - entry_gap) / (1 - ENTRY_GAP0)) * rel * trend / price

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
            labor_share=W / (Y * price), capital_income_net=K_net, ai_rents=R, ai_rents_share=R / (Y * price),
            price_level=price, capex_index=capex,
            crisis=int(crisis_year is not None and t >= crisis_year), crisis_year=crisis_year or 0,
            valuation_multiple=M, broad_equity=broad, equity_total_return=tr, ai_rents_index=ai_rents_index, bond=bond,
            portfolio_60_40=portfolio, portfolio_80_20=portfolio_80_20, social_security=ss,
            retiree_income=retiree_income, care_cost=care_cost,
            incumbent_wage=incumbent_wage, incumbent_hazard=hazard, incumbent_survival=cum_survival,
            incumbent_exit=1 - cum_survival,
            displaced_income=displaced_income, entrant_index=entrant, entry_gap=entry_gap,
            licensed_index=licensed, infra_index=infra,
        ))

        movers = p.mobility * X_c
        S_c = (S_c - movers) * (1 + p.labor_growth)
        S_p = (S_p + movers) * (1 + p.labor_growth)

    return add_drawdown(pd.DataFrame(rows))


SCENARIOS = [
    Params(
        key="loom", name="Comparator: the loom",
        description="Capability plateaus well short of a general substitute and most new tasks go to humans. "
                    "This is what the standard reassurance (steam, combustion, transistor) assumes, and it "
                    "carries the probability that the essay's premise is wrong.",
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
SCENARIO_BY_KEY = {p.key: p for p in SCENARIOS}
DEFAULTS = Params(key="_", name="_", description="_")

# ----------------------------------------------------------------------------------
# Monte Carlo: judgment ranges (uniform) for the parameters every substitution scenario
# shares. Scenario-specific values are kept as offsets from the defaults, so a scenario
# that is slower or more human-friendly than the default stays that way in every draw.
# ----------------------------------------------------------------------------------
MC_RANGES = {
    "cog_cap_max": (0.65, 0.95),
    "cog_cap_mid": (2029.0, 2035.0),
    "cog_cap_k": (0.30, 0.60),
    "phys_cap_max": (0.40, 0.75),
    "phys_cap_mid": (2034.0, 2042.0),
    "adoption_rate": (0.15, 0.40),
    "new_task_rate": (0.30, 0.80),
    "new_task_share_cog": (0.10, 0.45),
    "new_task_share_phys": (0.25, 0.60),
    "scale_effect": (0.20, 0.60),
    "demand_elasticity": (-1.0, -0.40),
    "mobility": (0.15, 0.40),
    "phi_cog": (0.30, 0.70),
    "reservation_wage": (0.35, 0.55),
    "license_cap_early": (0.20, 0.45),
    "license_absorption": (0.30, 0.70),
    "entry_gap_slope": (0.20, 0.60),
    "hazard_slope": (0.80, 2.20),
    "protection_years": (4.0, 9.0),
    "capex_growth": (0.10, 0.30),
    "equity_beta": (0.40, 0.80),
    "equity_yield": (0.025, 0.045),
}
# The comparator has its own ranges for the parameters that define it.
LOOM_RANGES = {
    "cog_cap_max": (0.35, 0.55),
    "cog_cap_mid": (2028.0, 2033.0),
    "new_task_share_cog": (0.65, 0.90),
    "new_task_share_phys": (0.65, 0.90),
    "new_task_rate": (0.70, 1.00),
}
BOUNDS = {
    "new_task_share_cog": (0.0, 0.95), "new_task_share_phys": (0.0, 0.95), "new_task_rate": (0.1, 1.2),
    "adoption_rate": (0.05, 0.6), "cog_cap_max": (0.3, 0.98), "phys_cap_max": (0.1, 0.95),
    "demand_elasticity": (-1.5, -0.2), "mobility": (0.05, 0.6), "scale_effect": (0.0, 1.0),
}
MC_MEASURES = [
    "real_wage_cog", "real_wage_phys", "not_employed", "dependency_ratio", "real_household_income",
    "labor_share", "output", "broad_equity", "portfolio_60_40", "retiree_income", "incumbent_wage",
    "incumbent_exit", "entrant_index", "licensed_index", "infra_index",
    "draw_all_equity", "draw_60_40", "draw_bucket",
]


def sample_params(rng: np.random.Generator) -> tuple[dict, dict]:
    shared = {k: float(rng.uniform(*v)) for k, v in MC_RANGES.items()}
    loom = dict(shared)
    for k, v in LOOM_RANGES.items():
        loom[k] = float(rng.uniform(*v))
    return shared, loom


def apply_sample(p: Params, sampled: dict, is_loom: bool) -> Params:
    changes = {}
    for k, v in sampled.items():
        if is_loom and k in LOOM_RANGES:
            val = v
        else:
            val = v + (getattr(p, k) - getattr(DEFAULTS, k))   # keep the scenario's offset
        lo, hi = BOUNDS.get(k, (-1e9, 1e9))
        changes[k] = float(min(max(val, lo), hi))
    return replace(p, **changes)


def weighted(df: pd.DataFrame, weights: dict, key: str, name: str) -> pd.DataFrame:
    num = [c for c in df.select_dtypes("number").columns if c != "year"]
    parts = [df[df.scenario == k].set_index("year")[num] * w for k, w in weights.items()]
    exp = sum(parts)
    exp["scenario"] = key
    exp["scenario_name"] = name
    return exp.reset_index()


def monte_carlo(n: int = 400, seed: int = 20261018) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w_all = all_in_weights()
    w_cond = conditional_weights()
    keys = [p.key for p in SCENARIOS]
    store = {k: {m: [] for m in MC_MEASURES} for k in keys + ["all_in", "conditional"]}
    for _ in range(n):
        shared, loom = sample_params(rng)
        frames = {}
        for p in SCENARIOS:
            q = apply_sample(p, loom if p.key == "loom" else shared, p.key == "loom")
            frames[p.key] = run(q).set_index("year")
        for k in keys:
            for m in MC_MEASURES:
                store[k][m].append(frames[k][m].to_numpy())
        for label, w in (("all_in", w_all), ("conditional", w_cond)):
            for m in MC_MEASURES:
                store[label][m].append(sum(frames[k][m].to_numpy() * wk for k, wk in w.items()))
    recs = []
    for k, measures in store.items():
        for m, arrs in measures.items():
            a = np.vstack(arrs)
            p10, p50, p90 = np.percentile(a, [10, 50, 90], axis=0)
            mean = a.mean(axis=0)
            for j, yr in enumerate(YEARS):
                recs.append(dict(kind=k, measure=m, year=yr, p10=p10[j], p50=p50[j], p90=p90[j], mean=mean[j]))
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------
PCT = {"not_employed", "dependency_ratio", "labor_share", "incumbent_exit", "ai_rents_share"}
LABELS = {
    "real_wage_cog": "Real cognitive wage (index)",
    "real_wage_phys": "Real presence wage (index)",
    "not_employed": "Not employed, share of workforce",
    "dependency_ratio": "Dependency ratio",
    "real_household_income": "Real household income, all households (index)",
    "labor_share": "Labor share of output",
    "output": "Output (index)",
    "broad_equity": "Broad equity, real (index)",
    "portfolio_60_40": "60/40 portfolio, total return, real (index)",
    "retiree_income": "Retired couple income (index)",
    "incumbent_wage": "Incumbent professional who keeps the job, real pay (index)",
    "incumbent_exit": "P(incumbent involuntary exit by year)",
    "entrant_index": "New graduate opportunity (openings x pay, index)",
    "licensed_index": "Licensed presence profession, real pay (index)",
    "infra_index": "Infrastructure engineer, real pay (index)",
    "draw_all_equity": "Retiree savings, all equity, 4% real withdrawals (start 100)",
    "draw_60_40": "Retiree savings, 60/40, 4% real withdrawals (start 100)",
    "draw_bucket": "Retiree savings, equity + 3-year cash bucket, 4% real withdrawals (start 100)",
}


def fmt(m: str, x: float) -> str:
    if m in PCT:
        return f"{100 * x:.0f}%"
    if m == "output":
        return f"{x:.0f}"
    return f"{100 * x:.0f}" if m.startswith(("real_", "broad", "portfolio", "retiree", "incumbent_wage", "entrant", "licensed", "infra")) else f"{x:.0f}"


def summary_tables(df: pd.DataFrame) -> str:
    lines = []
    for yr in (2031, 2036, 2040):
        lines.append(f"\n### Macro picture, {yr} (2026 = 100 unless a percentage; central parameters)\n")
        lines.append("| Scenario | AI can do (cognitive tasks) | Deployed | Human cognitive jobs demanded | Real cognitive wage | Real presence wage | Not employed | Labor share | AI rents, share of output | Dependency ratio | Real household income | Output | Broad equity (real) | Crisis year |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in SCENARIOS:
            d = df[(df.scenario == p.key) & (df.year == yr)].iloc[0]
            lines.append(
                f"| {p.name} | {100 * d.cap_cog:.0f}% | {100 * d.deployed_cog:.0f}% | {100 * d.jobs_cog_demand:.0f} | "
                f"{100 * d.real_wage_cog:.0f} | {100 * d.real_wage_phys:.0f} | {100 * d.not_employed:.0f}% | "
                f"{100 * d.labor_share:.0f}% | {100 * d.ai_rents_share:.0f}% | {100 * d.dependency_ratio:.0f}% | "
                f"{100 * d.real_household_income:.0f} | {d.output:.0f} | {100 * d.broad_equity:.0f} | "
                f"{int(d.crisis_year) or '—'} |"
            )
    return "\n".join(lines)


def household_tables(df: pd.DataFrame) -> str:
    lines = ["\n### Household archetypes (real income or opportunity index, 2026 = 100; central parameters)\n",
             "Trend productivity growth of 1.5% a year is included, so a flat line means falling behind the economy as a whole.\n"]
    lines.append("| Scenario | Retired couple 2033 | Retired couple 2040 | 60/40 portfolio 2033 | 60/40 portfolio 2040 | Social Security 2040 | Care-labor cost 2040 | Incumbent professional wage 2036 | P(involuntary exit by 2036) | Displaced professional income 2036 | New graduate (cognitive) 2030 | New graduate 2034 | New graduate 2038 | Licensed presence profession 2034 | Licensed 2040 | Infrastructure engineer 2030 | Infra 2036 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in SCENARIOS:
        d = df[df.scenario == p.key].set_index("year")
        lines.append(
            f"| {p.name} | {100 * d.loc[2033, 'retiree_income']:.0f} | {100 * d.loc[2040, 'retiree_income']:.0f} | "
            f"{100 * d.loc[2033, 'portfolio_60_40']:.0f} | {100 * d.loc[2040, 'portfolio_60_40']:.0f} | "
            f"{100 * d.loc[2040, 'social_security']:.0f} | {100 * d.loc[2040, 'care_cost']:.0f} | "
            f"{100 * d.loc[2036, 'incumbent_wage']:.0f} | {100 * d.loc[2036, 'incumbent_exit']:.0f}% | "
            f"{100 * d.loc[2036, 'displaced_income']:.0f} | "
            f"{100 * d.loc[2030, 'entrant_index']:.0f} | {100 * d.loc[2034, 'entrant_index']:.0f} | {100 * d.loc[2038, 'entrant_index']:.0f} | "
            f"{100 * d.loc[2034, 'licensed_index']:.0f} | {100 * d.loc[2040, 'licensed_index']:.0f} | "
            f"{100 * d.loc[2030, 'infra_index']:.0f} | {100 * d.loc[2036, 'infra_index']:.0f} |"
        )
    return "\n".join(lines)


def drawdown_tables(df: pd.DataFrame) -> str:
    lines = ["\n### Retiree savings under 4% real withdrawals (start = 100; central parameters)\n",
             "All equity: everything in the broad equity index. 60/40: rebalanced yearly. Bucket: equity plus three "
             "years of withdrawals in cash, spent down only when equities are below their previous peak and refilled "
             "at new peaks. 'Low point' is the lowest value reached in 2026 to 2040.\n",
             "| Scenario | All equity, low point | All equity, 2040 | 60/40, low point | 60/40, 2040 | Bucket, low point | Bucket, 2040 | Equity index peak-to-trough |",
             "|---|---|---|---|---|---|---|---|"]
    for p in SCENARIOS:
        d = df[df.scenario == p.key].set_index("year")
        peak = d.broad_equity.cummax()
        dd = float((d.broad_equity / peak - 1).min())
        lines.append(
            f"| {p.name} | {d.draw_all_equity.min():.0f} | {d.loc[2040, 'draw_all_equity']:.0f} | "
            f"{d.draw_60_40.min():.0f} | {d.loc[2040, 'draw_60_40']:.0f} | "
            f"{d.draw_bucket.min():.0f} | {d.loc[2040, 'draw_bucket']:.0f} | {100 * dd:.0f}% |"
        )
    return "\n".join(lines)


def uncertainty_tables(u: pd.DataFrame) -> str:
    w = all_in_weights()
    lines = ["# Uncertainty: Monte Carlo over the shared parameters\n",
             f"Each draw samples the technology and labor-market parameters from the judgment ranges in `MC_RANGES` "
             f"(the comparator from `LOOM_RANGES`), keeps each scenario's offsets, runs all six scenarios, and "
             f"forms the probability-weighted mixture. Weights (all-in): "
             + ", ".join(f"{k} {100 * v:.0f}%" for k, v in w.items())
             + f". P(premise wrong) = {100 * P_PREMISE_WRONG:.0f}%.\n",
             "\n## All-in expectation: 10th / 50th / 90th percentile across draws\n",
             "| Measure | 2031 | 2036 | 2040 |", "|---|---|---|---|"]
    a = u[u.kind == "all_in"].set_index(["measure", "year"])
    for m in MC_MEASURES:
        cells = [LABELS[m]]
        for yr in (2031, 2036, 2040):
            r = a.loc[(m, yr)]
            cells.append(f"{fmt(m, r.p50)} ({fmt(m, r.p10)} to {fmt(m, r.p90)})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n## Conditional on the premise (the essay's five states only): median and 10th to 90th percentile\n")
    lines.append("| Measure | 2031 | 2036 | 2040 |")
    lines.append("|---|---|---|---|")
    c = u[u.kind == "conditional"].set_index(["measure", "year"])
    for m in MC_MEASURES:
        cells = [LABELS[m]]
        for yr in (2031, 2036, 2040):
            r = c.loc[(m, yr)]
            cells.append(f"{fmt(m, r.p50)} ({fmt(m, r.p10)} to {fmt(m, r.p90)})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n## By scenario, 2036: median (10th to 90th percentile)\n")
    key_measures = ["real_wage_cog", "real_household_income", "dependency_ratio", "portfolio_60_40",
                    "incumbent_wage", "incumbent_exit", "entrant_index", "licensed_index", "draw_all_equity", "draw_bucket"]
    lines.append("| Scenario | " + " | ".join(LABELS[m] for m in key_measures) + " |")
    lines.append("|---|" + "---|" * len(key_measures))
    s = u.set_index(["kind", "measure", "year"])
    for p in SCENARIOS:
        cells = [p.name]
        for m in key_measures:
            r = s.loc[(p.key, m, 2036)]
            cells.append(f"{fmt(m, r.p50)} ({fmt(m, r.p10)}–{fmt(m, r.p90)})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    fast = "--fast" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    frames = [run(p) for p in SCENARIOS]
    df = pd.concat(frames, ignore_index=True)
    cond = weighted(df, conditional_weights(), "probability_weighted",
                    "Probability-weighted, conditional on the premise (five modeled states)")
    allin = weighted(df, all_in_weights(), "all_in_weighted",
                     f"All-in: premise wrong {100 * P_PREMISE_WRONG:.0f}% (loom) + essay posteriors")
    pd.concat([df, cond, allin], ignore_index=True).to_csv(os.path.join(OUT, "trajectories.csv"), index=False)
    with open(os.path.join(OUT, "params.json"), "w") as fh:
        json.dump({"P_PREMISE_WRONG": P_PREMISE_WRONG, "all_in_weights": all_in_weights(),
                   "mc_ranges": MC_RANGES, "loom_ranges": LOOM_RANGES,
                   "scenarios": [asdict(p) for p in SCENARIOS]}, fh, indent=2)

    md = ["# Two-economy model: results (central parameters)\n",
          "Generated by `two_economy.py`. Illustrative, not a forecast; see MODEL.md for assumptions and "
          "`uncertainty.md` for how much these numbers move when the shared parameters are varied.\n",
          "\n## Scenarios\n"]
    for p in SCENARIOS:
        md.append(f"- **{p.name}** — {p.description}")
    md.append(summary_tables(df))
    md.append(household_tables(df))
    md.append(drawdown_tables(df))
    md.append("\n### Point expectations (central parameters)\n")
    md.append("| Measure | All-in 2031 | All-in 2036 | All-in 2040 | Conditional 2036 | Loom 2036 |")
    md.append("|---|---|---|---|---|---|")
    loom = df[df.scenario == "loom"].set_index("year")
    for m in MC_MEASURES:
        md.append(f"| {LABELS[m]} | " + " | ".join(fmt(m, allin[allin.year == yr][m].iloc[0]) for yr in (2031, 2036, 2040))
                  + f" | {fmt(m, cond[cond.year == 2036][m].iloc[0])} | {fmt(m, loom.loc[2036, m])} |")
    with open(os.path.join(OUT, "results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))

    if not fast:
        u = monte_carlo()
        u.to_csv(os.path.join(OUT, "uncertainty.csv"), index=False)
        text = uncertainty_tables(u)
        with open(os.path.join(OUT, "uncertainty.md"), "w") as fh:
            fh.write(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
