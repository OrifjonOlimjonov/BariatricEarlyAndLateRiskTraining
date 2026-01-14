# data_generate.py
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def truncated_normal(rng, mean, sd, low, high, size):
    x = rng.normal(mean, sd, size)
    return np.clip(x, low, high)

TIMEPOINTS = ["discharge", "2w", "1m", "3m", "6m", "12m"]

def generate_synthetic_bariatric(n_patients=10_000, seed=42):
    rng = np.random.default_rng(seed)

    # --- Static patient profile ---
    age = truncated_normal(rng, mean=39, sd=10, low=18, high=65, size=n_patients).round().astype(int)
    female = rng.binomial(1, 0.65, n_patients)
    sex = np.where(female == 1, "F", "M")

    height_cm = truncated_normal(rng, mean=165, sd=9, low=145, high=195, size=n_patients)
    bmi = truncated_normal(rng, mean=43, sd=6, low=35, high=60, size=n_patients)
    height_m = height_cm / 100.0
    preop_weight = (bmi * (height_m ** 2))

    smoking = rng.binomial(1, 0.18, n_patients)

    # Comorbidities with BMI/age correlation
    p_dm2 = sigmoid(-6 + 0.12*bmi + 0.02*(age-40))
    dm2 = rng.binomial(1, p_dm2)

    p_htn = sigmoid(-5 + 0.10*bmi + 0.03*(age-45))
    htn = rng.binomial(1, p_htn)

    p_osa = sigmoid(-7 + 0.13*bmi + 0.25*(sex=="M"))
    osa = rng.binomial(1, p_osa)

    p_gerd = sigmoid(-1.2 + 0.35*smoking + 0.25*(bmi>45))
    gerd = rng.binomial(1, p_gerd)

    nafld = rng.binomial(1, sigmoid(-4 + 0.10*bmi))
    depression = rng.binomial(1, 0.20)

    # Labs pre-op
    hba1c = np.where(dm2==1, truncated_normal(rng, 7.5, 1.1, 5.8, 12.5, n_patients),
                    truncated_normal(rng, 5.5, 0.4, 4.5, 6.4, n_patients))
    vitd = truncated_normal(rng, 22, 8, 5, 60, n_patients)
    ferritin = truncated_normal(rng, 80 - 8*female, 40, 5, 300, n_patients)
    b12 = truncated_normal(rng, 420, 120, 120, 900, n_patients)
    albumin = truncated_normal(rng, 4.1, 0.25, 3.0, 5.0, n_patients)
    creatinine = truncated_normal(rng, 0.9 + 0.15*(sex=="M"), 0.2, 0.5, 1.8, n_patients)

    # Surgery-related
    asa = np.clip((2 + (age>50).astype(int) + (dm2==1).astype(int) + (osa==1).astype(int)), 2, 4)
    op_duration = truncated_normal(rng, 90 + 10*(bmi-40)/5 + 8*(asa-2), 20, 45, 180, n_patients).round()
    los_days = np.clip((2 + 0.03*(op_duration-90) + 0.6*(asa>=3) + rng.normal(0,0.7,n_patients)), 1, 10).round()

    # Adherence baseline tendencies
    base_diet = np.clip(rng.normal(0.70, 0.15, n_patients), 0.05, 1.0)
    base_supp = np.clip(rng.normal(0.65, 0.20, n_patients), 0.05, 1.0)

    # --- Generate longitudinal records (long format) ---
    rows = []
    for i, tp in enumerate(TIMEPOINTS):
        # Adherence drifts slightly
        diet_ad = np.clip(base_diet + rng.normal(0, 0.05, n_patients) - 0.02*(i>=3), 0.0, 1.0)
        supp_ad = np.clip(base_supp + rng.normal(0, 0.06, n_patients) - 0.03*(i>=4), 0.0, 1.0)

        # Symptoms: early more common if smoking/gerd/low adherence
        vomit = rng.binomial(1, sigmoid(-2.2 + 0.9*(i<=1) + 0.4*smoking + 0.6*(1-diet_ad)))
        reflux = rng.binomial(1, sigmoid(-2.0 + 0.8*gerd + 0.35*smoking + 0.25*(i<=2)))
        diarrhea = rng.binomial(1, sigmoid(-2.6 + 0.35*(i<=2) + 0.25*(1-diet_ad)))
        dumping = rng.binomial(1, sigmoid(-3.0 + 0.6*(i<=2) + 0.4*(1-diet_ad)))

        # Intake estimates
        protein = np.clip(rng.normal(70, 15, n_patients) + 15*(diet_ad-0.6) - 12*vomit, 20, 140)
        fluid = np.clip(rng.normal(1.8, 0.4, n_patients) + 0.6*(diet_ad-0.6) - 0.5*vomit, 0.4, 3.5)

        # Weight trajectory based on final TWL that depends on adherence/symptoms
        twl_12m_base = 0.22 + 0.08*sigmoid((bmi-40)/5)
        twl_12m = np.clip(twl_12m_base + 0.06*(base_diet-0.6) - 0.03*(vomit.mean()) + rng.normal(0,0.03,n_patients), 0.05, 0.45)

        # progress fractions for each timepoint
        frac = {"discharge":0.02, "2w":0.06, "1m":0.14, "3m":0.55, "6m":0.85, "12m":1.0}[tp]
        twl_t = twl_12m * frac
        weight_t = preop_weight * (1 - twl_t)
        bmi_t = weight_t / (height_m**2)

        # Labs evolve with supplementation/adherence
        # (Add missingness later in features step or here)
        ferritin_t = np.clip(ferritin - 20*frac*(1-supp_ad) - 5*vomit + rng.normal(0,10,n_patients), 3, 400)
        b12_t = np.clip(b12 - 80*frac*(1-supp_ad) - 25*vomit + rng.normal(0,40,n_patients), 80, 1200)
        vitd_t = np.clip(vitd - 6*frac*(1-supp_ad) + rng.normal(0,4,n_patients), 3, 80)
        albumin_t = np.clip(albumin - 0.25*frac*(1-diet_ad) - 0.15*vomit + rng.normal(0,0.08,n_patients), 2.5, 5.2)

        # HbA1c improves over time if DM2 and good TWL
        hba1c_t = hba1c - (1.0*frac*(dm2==1) * (0.6 + 0.8*twl_12m)) + rng.normal(0,0.2,n_patients)
        hba1c_t = np.clip(hba1c_t, 4.5, 12.5)

        df_tp = pd.DataFrame({
            "patient_id": np.arange(n_patients),
            "timepoint": tp,

            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "preop_bmi": bmi,
            "preop_weight_kg": preop_weight,

            "smoking": smoking,
            "dm2": dm2,
            "htn": htn,
            "osa": osa,
            "gerd": gerd,
            "nafld": nafld,
            "depression": depression,

            "asa": asa,
            "op_duration_min": op_duration,
            "los_days": los_days,

            "diet_adherence": diet_ad,
            "supplement_adherence": supp_ad,

            "vomiting": vomit,
            "reflux": reflux,
            "diarrhea": diarrhea,
            "dumping": dumping,

            "protein_g_day": protein,
            "fluid_l_day": fluid,

            "hba1c": hba1c_t,
            "ferritin": ferritin_t,
            "b12": b12_t,
            "vitd": vitd_t,
            "albumin": albumin_t,

            "weight_kg": weight_t,
            "bmi": bmi_t,
            # Keep final TWL for later patient-level label
            "twl_12m_true": twl_12m,
        })
        rows.append(df_tp)

    long_df = pd.concat(rows, ignore_index=True)

    # --- Patient-level outcomes (labels) ---
    # Use early follow-up signals (2w/1m) for generating outcomes.
    df_2w = long_df[long_df.timepoint=="2w"].set_index("patient_id")
    df_1m = long_df[long_df.timepoint=="1m"].set_index("patient_id")

    # Readmission 30d probability
    p_readm = sigmoid(
        -4.2
        + 0.6*(asa>=3)
        + 0.015*(op_duration-90)
        + 1.1*df_2w["vomiting"].values
        + 0.8*(df_2w["fluid_l_day"].values < 1.2)
        + 0.4*smoking
        + 0.3*(albumin < 3.6)
        + rng.normal(0, 0.3, n_patients)
    )
    readm_30d = rng.binomial(1, np.clip(p_readm, 0, 1))

    # Complication 90d probability
    p_comp = sigmoid(
        -4.5
        + 0.7*(asa>=3)
        + 0.02*(op_duration-90)
        + 0.9*df_2w["vomiting"].values
        + 0.6*df_1m["reflux"].values
        + 0.4*smoking
        + 0.3*(albumin < 3.6)
        + rng.normal(0, 0.35, n_patients)
    )
    comp_90d = rng.binomial(1, np.clip(p_comp, 0, 1))

    # Deficiencies at 12m (threshold-based labels from 12m labs)
    df_12m = long_df[long_df.timepoint=="12m"].set_index("patient_id")
    iron_def_12m = (df_12m["ferritin"].values < 20).astype(int)
    b12_def_12m = (df_12m["b12"].values < 200).astype(int)
    vitd_def_12m = (df_12m["vitd"].values < 20).astype(int)
    any_def_12m = ((iron_def_12m + b12_def_12m + vitd_def_12m) > 0).astype(int)

    outcomes = pd.DataFrame({
        "patient_id": np.arange(n_patients),
        "readmission_30d": readm_30d,
        "complication_90d": comp_90d,
        "iron_def_12m": iron_def_12m,
        "b12_def_12m": b12_def_12m,
        "vitd_def_12m": vitd_def_12m,
        "any_def_12m": any_def_12m,
        "twl_12m": df_12m["twl_12m_true"].values,  # true underlying
    })

    return long_df, outcomes