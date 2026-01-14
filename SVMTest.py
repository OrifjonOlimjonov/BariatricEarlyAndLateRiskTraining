from bariatric_ai.followup_optimization import optimize_followup_plan

# Misol risklar (sizda modeldan chiqadi)
risks = {
    "p_readmission_30d": 0.054,
    "p_complication_90d": 0.043,
    "p_any_def_12m": 0.80,
}

# Misol feature triggerlar (sizda input features)
features = {
    "early_vomiting": 1,
    "early_fluid_l_day": 0.9,
    "early_supplement_adherence": 0.5,
}

plan = optimize_followup_plan(risks=risks, features=features)

print("Status:", plan["status"])
print("Objective:", plan["objective_value"])
print("High early risk:", plan["high_early_risk"])
print("High def risk:", plan["high_def_risk"])
print("\n=== Plan (12 hafta) ===")
for item in plan["plan"]:
    print(f"Week {item['week']:>2}: {item['contact']}")