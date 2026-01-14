from __future__ import annotations

# Minimal schema. Siz keyin kengaytirasiz.
# Key not found bo'lsa, dastur default label = column name qiladi.

FIELD_SCHEMA = {
    # --- Static ---
    "age": {"label": "Yosh", "unit": "yil", "hint": "Masalan: 42", "type": "num"},
    "sex": {"label": "Jins", "unit": "", "hint": "F = ayol, M = erkak", "type": "choice", "choices": ["F", "M"]},
    "height_cm": {"label": "Bo‘y", "unit": "sm", "hint": "Masalan: 165", "type": "num"},
    "preop_bmi": {"label": "Operatsiyagacha BMI", "unit": "kg/m²", "hint": "Masalan: 46", "type": "num"},
    "preop_weight_kg": {"label": "Operatsiyagacha vazn", "unit": "kg", "hint": "Masalan: 125", "type": "num"},

    "smoking": {"label": "Chekish", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "dm2": {"label": "2-tur qandli diabet (DM2)", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "htn": {"label": "Arterial gipertenziya", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "osa": {"label": "Uyqu apnoesi (OSA)", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "gerd": {"label": "GERD/Refluks tarixi", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "nafld": {"label": "Jigar yog‘lanishi (NAFLD)", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "depression": {"label": "Depressiya", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},

    "asa": {"label": "ASA klass", "unit": "", "hint": "2–4", "type": "num"},
    "op_duration_min": {"label": "Operatsiya davomiyligi", "unit": "min", "hint": "Masalan: 105", "type": "num"},
    "los_days": {"label": "Stasionarda qolish", "unit": "kun", "hint": "Masalan: 3", "type": "num"},

    # --- Discharge dynamic ---
    "diet_adherence": {"label": "Discharge: Dietaga rioya", "unit": "0..1", "hint": "0.0–1.0", "type": "num"},
    "supplement_adherence": {"label": "Discharge: Qo‘shimchaga rioya", "unit": "0..1", "hint": "0.0–1.0", "type": "num"},
    "vomiting": {"label": "Discharge: Qusish", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "reflux": {"label": "Discharge: Refluks", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "diarrhea": {"label": "Discharge: Ich ketishi", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "dumping": {"label": "Discharge: Dumping belgilari", "unit": "", "hint": "Ha/Yo‘q", "type": "bool"},
    "protein_g_day": {"label": "Discharge: Protein qabul", "unit": "g/kun", "hint": "Masalan: 60", "type": "num"},
    "fluid_l_day": {"label": "Discharge: Suyuqlik", "unit": "L/kun", "hint": "Masalan: 1.6", "type": "num"},
    "weight_kg": {"label": "Discharge: Vazn", "unit": "kg", "hint": "Masalan: 123", "type": "num"},
    "bmi": {"label": "Discharge: BMI", "unit": "kg/m²", "hint": "Masalan: 45.2", "type": "num"},

    "hba1c": {"label": "Discharge: HbA1c", "unit": "%", "hint": "Masalan: 7.2", "type": "num"},
    "ferritin": {"label": "Discharge: Ferritin", "unit": "ng/mL", "hint": "Masalan: 70", "type": "num"},
    "b12": {"label": "Discharge: B12", "unit": "pg/mL", "hint": "Masalan: 420", "type": "num"},
    "vitd": {"label": "Discharge: Vitamin D", "unit": "ng/mL", "hint": "Masalan: 22", "type": "num"},
    "albumin": {"label": "Discharge: Albumin", "unit": "g/dL", "hint": "Masalan: 4.0", "type": "num"},

    # --- Early aggregated (2w + 1m) ---
    "early_diet_adherence": {"label": "(2 hafta–1 oy) Dietaga rioya (o‘rtacha)", "unit": "0..1", "hint": "0.0–1.0", "type": "num"},
    "early_supplement_adherence": {"label": "(2 hafta–1 oy) Qo‘shimchaga rioya (o‘rtacha)", "unit": "0..1", "hint": "0.0–1.0", "type": "num"},
    "early_vomiting": {"label": "(2 hafta–1 oy) Qusish (o‘rtacha)", "unit": "0..1", "hint": "0 yoki 1 (yoki 0..1)", "type": "num"},
    "early_reflux": {"label": "(2 hafta–1 oy) Refluks (o‘rtacha)", "unit": "0..1", "hint": "0 yoki 1 (yoki 0..1)", "type": "num"},
    "early_diarrhea": {"label": "(2 hafta–1 oy) Ich ketishi (o‘rtacha)", "unit": "0..1", "hint": "0 yoki 1 (yoki 0..1)", "type": "num"},
    "early_dumping": {"label": "(2 hafta–1 oy) Dumping (o‘rtacha)", "unit": "0..1", "hint": "0 yoki 1 (yoki 0..1)", "type": "num"},
    "early_protein_g_day": {"label": "(2 hafta–1 oy) Protein (o‘rtacha)", "unit": "g/kun", "hint": "Masalan: 45", "type": "num"},
    "early_fluid_l_day": {"label": "(2 hafta–1 oy) Suyuqlik (o‘rtacha)", "unit": "L/kun", "hint": "Masalan: 0.9", "type": "num"},
    "early_weight_kg": {"label": "1 oy: Vazn (last)", "unit": "kg", "hint": "Masalan: 118", "type": "num"},
    "early_bmi": {"label": "1 oy: BMI (last)", "unit": "kg/m²", "hint": "Masalan: 43.3", "type": "num"},
    "early_hba1c": {"label": "(2 hafta–1 oy) HbA1c (o‘rtacha)", "unit": "%", "hint": "Bo‘lsa kiriting", "type": "num"},
    "early_ferritin": {"label": "(2 hafta–1 oy) Ferritin (o‘rtacha)", "unit": "ng/mL", "hint": "Bo‘lsa kiriting", "type": "num"},
    "early_b12": {"label": "(2 hafta–1 oy) B12 (o‘rtacha)", "unit": "pg/mL", "hint": "Bo‘lsa kiriting", "type": "num"},
    "early_vitd": {"label": "(2 hafta–1 oy) Vitamin D (o‘rtacha)", "unit": "ng/mL", "hint": "Bo‘lsa kiriting", "type": "num"},
    "early_albumin": {"label": "(2 hafta–1 oy) Albumin (o‘rtacha)", "unit": "g/dL", "hint": "Bo‘lsa kiriting", "type": "num"},
    "early_weight_change_kg": {"label": "Discharge→1 oy vazn o‘zgarishi", "unit": "kg", "hint": "Masalan: -5", "type": "num"},
}