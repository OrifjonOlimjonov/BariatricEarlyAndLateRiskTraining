from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

# ---- Helpers ----
def _fmt_pct(p: float) -> str:
    return f"{p*100:.1f}%"

def _is_high(p: float, thr: float) -> bool:
    return p is not None and p >= thr

def _safe_get(d: Dict[str, Any], key: str, default=None):
    v = d.get(key, default)
    return default if v is None else v

# ---- Output schema ----
@dataclass
class RecommendationItem:
    title: str
    patient_text: str
    clinician_text: str
    triggers: List[str]
    priority: int  # 1=high, 2=medium, 3=low

def generate_recommendations(
    *,
    risks: Dict[str, float],
    features: Dict[str, Any],
    mode: str = "both",  # "patient" | "clinician" | "both"
) -> Dict[str, Any]:
    """
    risks keys expected (if available):
      - p_readmission_30d
      - p_complication_90d
      - p_any_def_12m
      - pred_twl_12m  (regression, 0..1)

    features: discharge + early_* features (whatever you have).
    Missing keys are handled gracefully.
    """

    # ---- thresholds (you can tune later) ----
    thr_readm_high = 0.08
    thr_comp_high  = 0.08
    thr_def_high   = 0.65

    # Labs (typical cutoffs; adapt to your local guideline)
    ferritin_low = 20.0
    b12_low = 200.0
    vitd_low = 20.0
    albumin_low = 3.5

    # Early features
    early_vomiting = float(_safe_get(features, "early_vomiting", _safe_get(features, "vomiting", 0)) or 0)
    early_reflux = float(_safe_get(features, "early_reflux", _safe_get(features, "reflux", 0)) or 0)
    early_diarrhea = float(_safe_get(features, "early_diarrhea", _safe_get(features, "diarrhea", 0)) or 0)

    early_fluid = _safe_get(features, "early_fluid_l_day", _safe_get(features, "fluid_l_day", None))
    early_protein = _safe_get(features, "early_protein_g_day", _safe_get(features, "protein_g_day", None))

    supp_ad = _safe_get(features, "early_supplement_adherence", _safe_get(features, "supplement_adherence", None))
    diet_ad = _safe_get(features, "early_diet_adherence", _safe_get(features, "diet_adherence", None))

    ferritin = _safe_get(features, "early_ferritin", _safe_get(features, "ferritin", None))
    b12 = _safe_get(features, "early_b12", _safe_get(features, "b12", None))
    vitd = _safe_get(features, "early_vitd", _safe_get(features, "vitd", None))
    albumin = _safe_get(features, "early_albumin", _safe_get(features, "albumin", None))

    dm2 = int(_safe_get(features, "dm2", 0) or 0)

    p_readm = risks.get("p_readmission_30d", None)
    p_comp = risks.get("p_complication_90d", None)
    p_def = risks.get("p_any_def_12m", None)
    pred_twl = risks.get("pred_twl_12m", None)

    items: List[RecommendationItem] = []

    # ---- 1) High-risk follow-up intensity ----
    if _is_high(p_readm, thr_readm_high) or _is_high(p_comp, thr_comp_high):
        triggers = []
        if _is_high(p_readm, thr_readm_high): triggers.append(f"Readmission riski { _fmt_pct(p_readm) }")
        if _is_high(p_comp, thr_comp_high): triggers.append(f"Asorat riski { _fmt_pct(p_comp) }")

        items.append(RecommendationItem(
            title="Kuzatuv (follow-up)ni kuchaytirish",
            patient_text=(
                "Sizda erta davrda (1–3 oy) muammo chiqish ehtimoli biroz yuqoriroq. "
                "Shuning uchun nazorat ko‘rigini kechiktirmang, suyuqlik va ovqatlanish rejimiga qat’iy amal qiling. "
                "Agar ahvol yomonlashsa (tinmay qusish, kuchli og‘riq, isitma, hushsizlik) zudlik bilan shifokorga murojaat qiling."
            ),
            clinician_text=(
                f"Risk: readm_30d={_fmt_pct(p_readm) if p_readm is not None else 'NA'}, "
                f"comp_90d={_fmt_pct(p_comp) if p_comp is not None else 'NA'}. "
                "Qisqa muddatli follow-up intervalini qisqartirish, rehidrasiya/antiemetic ehtiyojini baholash."
            ),
            triggers=triggers,
            priority=1
        ))

    # ---- 2) Hydration / vomiting ----
    dehydration_trigger = (early_fluid is not None and float(early_fluid) < 1.2) or (early_vomiting >= 1.0)
    if dehydration_trigger:
        trig = []
        if early_fluid is not None and float(early_fluid) < 1.2:
            trig.append(f"Suyuqlik kam: {early_fluid} L/kun")
        if early_vomiting >= 1.0:
            trig.append("Qusish bor")

        items.append(RecommendationItem(
            title="Suyuqlik va qusish nazorati",
            patient_text=(
                "Kun davomida oz-ozdan, tez-tez suyuqlik iching (maqsad: odatda 1.5–2.0 L/kun, "
                "agar shifokor boshqa cheklov qo‘ymagan bo‘lsa). "
                "Qusish bo‘lsa, suyuqlikni bir yo‘la ko‘p ichmang; mayda qultum qiling. "
                "Agar 6–8 soat siydik kamayib ketsa, bosh aylanishi bo‘lsa yoki qusish to‘xtamasa — shifokorga murojaat qiling."
            ),
            clinician_text=(
                "Dehidratatsiya/qusish triggerlandi. "
                "Early rehydration, antiemetic, stenoz/obstruktsiya istisno qilish, elektrolitlar nazorati ko‘rib chiqilsin."
            ),
            triggers=trig,
            priority=1
        ))

    # ---- 3) Protein / nutrition ----
    protein_low = (early_protein is not None and float(early_protein) < 60) or (albumin is not None and float(albumin) < albumin_low)
    if protein_low:
        trig = []
        if early_protein is not None and float(early_protein) < 60:
            trig.append(f"Protein past: {early_protein} g/kun")
        if albumin is not None and float(albumin) < albumin_low:
            trig.append(f"Albumin past: {albumin}")

        items.append(RecommendationItem(
            title="Protein va ovqatlanish rejimi",
            patient_text=(
                "Protein yetarli bo‘lishi tiklanish va mushaklarni saqlash uchun muhim. "
                "Imkon qadar protein manbalarini (shifokor/dietolog tavsiya qilgan) ko‘paytiring, "
                "ovqatni kichik porsiyada, tez-tez iste’mol qiling. "
                "Agar ovqat ko‘tara olmasangiz yoki tez-tez qusish bo‘lsa — ko‘rik zarur."
            ),
            clinician_text=(
                "Protein/albumin pastligi ehtimoli. Dietologik konsul'tatsiya, intolerans sababi (qusish, stenoz) tekshirilsin."
            ),
            triggers=trig,
            priority=2
        ))

    # ---- 4) Supplement adherence ----
    if supp_ad is not None and float(supp_ad) < 0.6:
        items.append(RecommendationItem(
            title="Vitamin va mineral qo‘shimchalariga rioya qilish",
            patient_text=(
                "Bariatrik jarrohlikdan keyin vitamin/mineral yetishmovchiligi tez-tez uchraydi. "
                "Shifokor belgilagan qo‘shimchalarni (multivitamin, B12, temir, D vitamini va boshqalar) muntazam qabul qilish juda muhim."
            ),
            clinician_text=(
                f"Supplement adherence past: {float(supp_ad):.2f}. "
                "Deficiency riskini kamaytirish uchun qo‘shimcha rejim va nazorat tahlillarini kuchaytirish."
            ),
            triggers=[f"supp_adherence={float(supp_ad):.2f}"],
            priority=2
        ))

    # ---- 5) Deficiency risk high (model-driven) ----
    if _is_high(p_def, thr_def_high):
        items.append(RecommendationItem(
            title="Mikronutrient yetishmovchiligi bo‘yicha kuchaytirilgan monitoring",
            patient_text=(
                "Sizda 12 oy ichida vitamin/mineral yetishmovchiligi ehtimoli yuqoriroq ko‘rinmoqda. "
                "Shuning uchun tahlillarni (temir zaxirasi, B12, D vitamini va boshqalar) vaqtida topshirish va "
                "qo‘shimchalarni muntazam ichish tavsiya etiladi."
            ),
            clinician_text=(
                f"Model: p_any_def_12m={_fmt_pct(p_def)} (thr={_fmt_pct(thr_def_high)}). "
                "Lab monitoring intervalini qisqartirish (masalan 3–6 oy), adherence va simptomlar bilan birga baholash."
            ),
            triggers=[f"p_any_def_12m={_fmt_pct(p_def)}"],
            priority=1
        ))

    # ---- 6) Lab-based specific recommendations ----
    if ferritin is not None and float(ferritin) < ferritin_low:
        items.append(RecommendationItem(
            title="Temir (ferritin) past",
            patient_text=(
                "Temir zaxirasi past ko‘rinadi. Shifokor bilan maslahatlashib, temir qo‘shimchasi va qayta tahlil rejasini belgilang. "
                "Holat: holsizlik, tez charchash, bosh aylanishi bo‘lsa albatta ayting."
            ),
            clinician_text=(
                f"Ferritin={float(ferritin):.1f} (<{ferritin_low}). Iron deficiency bo‘yicha protokolga muvofiq baholash/supplement."
            ),
            triggers=[f"ferritin={float(ferritin):.1f}"],
            priority=1
        ))

    if b12 is not None and float(b12) < b12_low:
        items.append(RecommendationItem(
            title="B12 vitamini past",
            patient_text=(
                "B12 vitamini past ko‘rinadi. Bu asab tizimi va qon ishlab chiqarilishi uchun muhim. "
                "Shifokor bilan maslahatlashib B12 qo‘shimchasini (tabletka yoki inʼeksiya) rejalashtiring."
            ),
            clinician_text=(
                f"B12={float(b12):.1f} (<{b12_low}). Bariatrikdan keyingi B12 protokoli bo‘yicha qo‘shimcha/monitoring."
            ),
            triggers=[f"b12={float(b12):.1f}"],
            priority=1
        ))

    if vitd is not None and float(vitd) < vitd_low:
        items.append(RecommendationItem(
            title="D vitamini past",
            patient_text=(
                "D vitamini past bo‘lishi suyaklar va umumiy holatga ta’sir qilishi mumkin. "
                "Shifokor tavsiyasiga ko‘ra D vitamini (va ko‘pincha kalsiy) qabul qilishni muhokama qiling."
            ),
            clinician_text=(
                f"VitD={float(vitd):.1f} (<{vitd_low}). D vitamini/kalsiy bo‘yicha protokolni ko‘rib chiqing."
            ),
            triggers=[f"vitd={float(vitd):.1f}"],
            priority=2
        ))

    # ---- 7) Reflux ----
    if early_reflux >= 1.0:
        items.append(RecommendationItem(
            title="Refluks (qayt qilish) simptomlari",
            patient_text=(
                "Refluks bezovta qilsa: ovqatni kichik porsiyada, sekin yeng; yotishdan oldin ovqat yemang; "
                "achchiq/yog‘li ovqatlarni cheklang. Belgilar kuchaysa shifokorga murojaat qiling."
            ),
            clinician_text=(
                "Refluks simptomi bor. Sleeve’dan keyin GERD kuchayishi mumkin; PPI/diagnostika ehtiyojini baholang."
            ),
            triggers=["early_reflux=1"],
            priority=2
        ))

    # ---- 8) Diarrhea ----
    if early_diarrhea >= 1.0:
        items.append(RecommendationItem(
            title="Ich ketishi (diareya)",
            patient_text=(
                "Ich ketishi bo‘lsa suyuqlikni ko‘paytiring, juda shirin/yog‘li ovqatlarni cheklang. "
                "Agar qon aralash ich ketish, isitma yoki suvsizlanish belgilari bo‘lsa shifokorga murojaat qiling."
            ),
            clinician_text=(
                "Diareya triggerlandi. Diet intoleransi, infektsiya yoki malabsorbsiya ehtimolini baholang."
            ),
            triggers=["early_diarrhea=1"],
            priority=3
        ))

    # ---- 9) Diabetes follow-up (if DM2) ----
    if dm2 == 1:
        items.append(RecommendationItem(
            title="Qandli diabet nazorati (DM2)",
            patient_text=(
                "Agar sizda qandli diabet bo‘lsa, vazn kamayishi bilan dori ehtiyoji o‘zgarishi mumkin. "
                "Qon shakarini nazorat qiling va dori dozasini faqat shifokor bilan kelishib o‘zgartiring."
            ),
            clinician_text=(
                "DM2 mavjud. Vazn yo‘qotish fonida glyukozani kuzatish, hipoglikemiya riski va dori titratsiyasi ko‘rib chiqilsin."
            ),
            triggers=["dm2=1"],
            priority=3
        ))

    # ---- 10) Weight loss expectation (regression-driven info) ----
    if pred_twl is not None:
        items.append(RecommendationItem(
            title="12 oyda vazn yo‘qotish prognozi",
            patient_text=(
                f"Model bo‘yicha 12 oyda taxminiy vazn yo‘qotish: {pred_twl*100:.1f}%. "
                "Bu prognoz bo‘lib, ovqatlanish va jismoniy faollikga rioya qilish natijani sezilarli o‘zgartiradi."
            ),
            clinician_text=(
                f"pred_twl_12m={pred_twl:.3f}. Past prognoz bo‘lsa adherence, simptomlar, psixologik/dietologik omillar ko‘rib chiqilsin."
            ),
            triggers=[f"pred_twl_12m={pred_twl:.3f}"],
            priority=3
        ))

    # Sort by priority then title
    items.sort(key=lambda it: (it.priority, it.title))

    # Build final response
    resp: Dict[str, Any] = {
        "risks": risks,
        "count": len(items),
        "items": [
            {
                "title": it.title,
                "priority": it.priority,
                "patient_text": it.patient_text,
                "clinician_text": it.clinician_text,
                "triggers": it.triggers,
            }
            for it in items
        ],
        "disclaimer_uz": (
            "Diqqat: Bu tavsiyalar sun’iy intellekt yordamida hisoblangan prognozlarga asoslangan bo‘lib, "
            "shifokor ko‘rigini va klinik qarorni almashtirmaydi."
        )
    }

    if mode == "patient":
        for it in resp["items"]:
            it.pop("clinician_text", None)
    elif mode == "clinician":
        for it in resp["items"]:
            it.pop("patient_text", None)

    return resp