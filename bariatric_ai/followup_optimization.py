from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import pulp


CONTACTS = ["telemed", "clinic", "dietitian", "lab"]


@dataclass
class FollowUpPlanItem:
    week: int
    contact: str


def _benefit_weights(risks: Dict[str, float], features: Dict[str, Any]) -> Dict[Tuple[int, str], float]:
    """
    Hand-crafted benefit weights (MVP).
    You can later learn these weights from data or expert survey.

    Intuition:
    - Early weeks: more important for readmission/complication
    - Later weeks: lab benefit grows if deficiency risk is high
    """
    p_readm = float(risks.get("p_readmission_30d", 0.0) or 0.0)
    p_comp = float(risks.get("p_complication_90d", 0.0) or 0.0)
    p_def = float(risks.get("p_any_def_12m", 0.0) or 0.0)

    vomiting = float(features.get("early_vomiting", 0.0) or 0.0)
    fluid = features.get("early_fluid_l_day", None)
    fluid_low = (fluid is not None and float(fluid) < 1.2)

    supp_ad = features.get("early_supplement_adherence", None)
    supp_low = (supp_ad is not None and float(supp_ad) < 0.6)

    benefit = {}

    for week in range(1, 13):  # 12 weeks
        # decay for early event risks
        early_factor = 1.0 if week <= 2 else (0.7 if week <= 4 else 0.4)
        mid_factor = 0.6 if week <= 6 else 0.4

        # contact-specific base benefit
        for c in CONTACTS:
            b = 0.0

            if c == "telemed":
                b += early_factor * (1.2 * p_readm + 0.8 * p_comp)
                if vomiting >= 1.0 or fluid_low:
                    b += 0.15  # triage advantage

            elif c == "clinic":
                b += early_factor * (1.6 * p_readm + 1.2 * p_comp)
                if vomiting >= 1.0:
                    b += 0.20  # can assess stenosis/dehydration

            elif c == "dietitian":
                b += mid_factor * 0.6  # general benefit
                if supp_low:
                    b += 0.10
                # protein issues can be added similarly

            elif c == "lab":
                # labs more valuable as deficiency risk grows; not very early
                lab_factor = 0.2 if week <= 2 else (0.6 if week <= 6 else 1.0)
                b += lab_factor * (1.4 * p_def)
                if supp_low:
                    b += 0.05

            benefit[(week, c)] = b

    return benefit


def optimize_followup_plan(
    *,
    risks: Dict[str, float],
    features: Dict[str, Any],
    weekly_capacity: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """
    Returns an optimized 12-week follow-up plan.

    weekly_capacity (optional):
      e.g. {"clinic": 1, "lab": 1, "dietitian": 1, "telemed": 2}
      Means max per week (for one patient plan you usually don't need it).
    """
    weekly_capacity = weekly_capacity or {}

    # Costs (relative)
    cost = {"telemed": 0.2, "clinic": 1.0, "dietitian": 0.6, "lab": 0.8}

    # Risk thresholds to enforce minimum early contact
    p_readm = float(risks.get("p_readmission_30d", 0.0) or 0.0)
    p_comp = float(risks.get("p_complication_90d", 0.0) or 0.0)
    p_def = float(risks.get("p_any_def_12m", 0.0) or 0.0)

    high_early_risk = (p_readm >= 0.08) or (p_comp >= 0.08)
    high_def_risk = (p_def >= 0.65)

    benefit = _benefit_weights(risks, features)

    # Optimization problem
    prob = pulp.LpProblem("followup_optimization", pulp.LpMaximize)

    x = pulp.LpVariable.dicts(
        "x",
        ((week, c) for week in range(1, 13) for c in CONTACTS),
        lowBound=0,
        upBound=1,
        cat="Binary"
    )

    # Objective: benefit - lambda*cost - burden penalty
    lambda_cost = 0.6
    burden_penalty = 0.15  # penalize having 2 contacts/week

    # additional variable: y_week_2 = 1 if week has 2 contacts
    y2 = pulp.LpVariable.dicts("y2", (week for week in range(1, 13)), 0, 1, cat="Binary")

    prob += pulp.lpSum(
        (benefit[(week, c)] - lambda_cost * cost[c]) * x[(week, c)]
        for week in range(1, 13)
        for c in CONTACTS
    ) - pulp.lpSum(burden_penalty * y2[week] for week in range(1, 13))

    # Constraints:
    # 1) max 2 contacts per week
    for week in range(1, 13):
        prob += pulp.lpSum(x[(week, c)] for c in CONTACTS) <= 2

    # Link y2: y2=1 if contacts==2 (relaxed linearization)
    for week in range(1, 13):
        prob += pulp.lpSum(x[(week, c)] for c in CONTACTS) >= 2 * y2[week]

    # 2) Clinic max 3 visits
    prob += pulp.lpSum(x[(week, "clinic")] for week in range(1, 13)) <= 3

    # 3) Labs spacing: no more than 1 lab in any 4-week window
    for week in range(1, 10):  # 1..9 start window
        prob += pulp.lpSum(x[(w, "lab")] for w in range(week, week + 4)) <= 1

    # 4) If high early risk, ensure at least one contact in week 1-2
    if high_early_risk:
        prob += pulp.lpSum(
            x[(week, "telemed")] + x[(week, "clinic")]
            for week in (1, 2)
        ) >= 1

    # 5) If high deficiency risk, ensure at least one lab by week 8-12 (or dietitian)
    if high_def_risk:
        prob += pulp.lpSum(x[(week, "lab")] for week in range(4, 13)) >= 1

    # Optional per-week capacity constraint (rarely needed for single patient plan)
    for c, cap in weekly_capacity.items():
        for week in range(1, 13):
            prob += x[(week, c)] <= cap  # cap 0/1 for per patient makes sense

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    plan: List[FollowUpPlanItem] = []
    for week in range(1, 13):
        for c in CONTACTS:
            if pulp.value(x[(week, c)]) >= 0.5:
                plan.append(FollowUpPlanItem(week=week, contact=c))

    plan.sort(key=lambda it: (it.week, it.contact))

    return {
        "status": pulp.LpStatus[prob.status],
        "high_early_risk": high_early_risk,
        "high_def_risk": high_def_risk,
        "objective_value": float(pulp.value(prob.objective)),
        "plan": [it.__dict__ for it in plan],
    }