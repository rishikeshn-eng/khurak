"""Safety checks over the medicines found in a piece of text.

Not medical advice: khurak flags things a pharmacist should look at. Thresholds are
approximate adult maxima from standard references (US FDA labels / OTC monographs,
BNF) and are deliberately conservative.
"""
import re

from .find import find

# ingredient -> (approx. adult max mg/day, note)
MAX_DAILY_MG = {
    "paracetamol": (4000, "liver damage; lower (2-3 g) with alcohol use, liver disease or low body weight"),
    "ibuprofen": (1200, "OTC limit; up to 3200 mg only under medical supervision"),
    "diclofenac": (150, ""), "aceclofenac": (200, ""), "nimesulide": (200, ""),
    "mefenamic acid": (1500, ""), "naproxen": (1250, ""), "etoricoxib": (120, ""),
    "cetirizine": (10, ""), "levocetirizine": (5, ""), "fexofenadine": (180, ""),
    "chlorphenamine": (24, ""), "diphenhydramine": (300, ""), "phenylephrine": (60, ""),
    "caffeine": (400, ""), "dextromethorphan": (120, ""), "tramadol": (400, ""), "montelukast": (10, ""),
}
CLASSES = {
    "NSAID": {"ibuprofen", "diclofenac", "aceclofenac", "nimesulide", "naproxen", "mefenamic acid", "etoricoxib",
              "piroxicam", "ketorolac", "indomethacin", "indometacin", "celecoxib", "flurbiprofen", "ketoprofen",
              "lornoxicam", "meloxicam", "dexketoprofen", "etodolac", "aspirin", "acetylsalicylic acid",
              "zaltoprofen", "tenoxicam", "dexibuprofen", "nabumetone"},
    "proton-pump inhibitor": {"omeprazole", "pantoprazole", "rabeprazole", "esomeprazole", "lansoprazole",
                              "dexlansoprazole", "dexrabeprazole"},
    "benzodiazepine-type sedative": {"alprazolam", "clonazepam", "diazepam", "lorazepam", "etizolam", "nitrazepam",
                                     "chlordiazepoxide", "clobazam", "zolpidem"},
    "opioid": {"tramadol", "tapentadol", "codeine", "morphine", "oxycodone", "fentanyl", "buprenorphine"},
    "antihistamine (cetirizine family)": {"cetirizine", "levocetirizine"},
    "SSRI/SNRI": {"sertraline", "fluoxetine", "escitalopram", "citalopram", "paroxetine", "fluvoxamine",
                  "duloxetine", "venlafaxine", "desvenlafaxine"},
    "anticoagulant/antiplatelet": {"warfarin", "acenocoumarol", "apixaban", "rivaroxaban", "dabigatran",
                                   "clopidogrel", "prasugrel", "ticagrelor"},
    "nitrate": {"isosorbide mononitrate", "isosorbide dinitrate", "nitroglycerin", "glyceryl trinitrate",
                "nicorandil"},
    "PDE5 inhibitor": {"sildenafil", "tadalafil", "vardenafil", "avanafil"},
}
# Well-established dangerous combinations (class or ingredient level).
INTERACTIONS = [
    ("NSAID", "anticoagulant/antiplatelet", "high", "serious bleeding risk"),
    ("opioid", "benzodiazepine-type sedative", "high", "dangerous sedation and slowed breathing"),
    ("opioid", "SSRI/SNRI", "high", "serotonin syndrome risk (especially tramadol/tapentadol)"),
    ("PDE5 inhibitor", "nitrate", "high", "severe, possibly fatal drop in blood pressure"),
]
SEVERE_DUP = {"paracetamol"} | CLASSES["NSAID"] | CLASSES["opioid"] | CLASSES["benzodiazepine-type sedative"]

_FREQ = [
    (re.compile(r"\b(\d)\s*-\s*(\d)\s*-\s*(\d)(?:\s*-\s*(\d))?\b"), None),
    (re.compile(r"\b(qid|four times|4 times)\b", re.I), 4), (re.compile(r"\b(tds|tid|thrice|three times|3 times)\b", re.I), 3),
    (re.compile(r"\b(bd|bid|twice|two times|2 times|subah[- ]shaam)\b", re.I), 2),
    (re.compile(r"\b(od|once|1 time|daily once|hs)\b", re.I), 1),
    (re.compile(r"\bevery\s+(4|6|8|12)\s*(?:hours|hrs|h)\b", re.I), "q"),
]


def doses_per_day(snippet):
    for rx, n in _FREQ:
        m = rx.search(snippet)
        if not m:
            continue
        if n is None:
            return sum(int(g) for g in m.groups() if g)
        if n == "q":
            return 24 // int(m.group(1))
        return n
    return None


def class_of(ing):
    return [c for c, members in CLASSES.items() if ing in members]


def check(text):
    found = find(text)
    items = []  # one per mention: {"label", "ingredients": {ing: mg}, "product"?}
    for b in found["brands"]:
        p = b["products"][0]
        fam_common = set.intersection(*[set(x.ingredients) for x in b["products"]])
        tail = text[b["end"]: b["end"] + 60]
        per_day = doses_per_day(tail)
        ings = {}
        for ing, mg in zip(p.ingredients, p.mg):
            if ing in fam_common:
                ings[ing] = mg
        items.append({"label": b["text"], "span": [b["start"], b["end"]], "product": p, "variants": len(b["products"]),
                      "ingredients": ings, "per_day": per_day, "match": b["match"], "confidence": b["confidence"]})
    for g in found["generics"]:
        items.append({"label": g["text"], "span": [g["start"], g["end"]], "product": None, "variants": 0,
                      "ingredients": {g["ingredient"]: None}, "per_day": None, "match": "generic", "confidence": 1.0})

    findings = []
    # 1. same active ingredient in two or more mentioned items
    by_ing = {}
    for k, it in enumerate(items):
        for ing in it["ingredients"]:
            by_ing.setdefault(ing, []).append(k)
    for ing, ks in by_ing.items():
        labels = sorted({items[k]["label"] for k in ks}, key=str.lower)
        if len(ks) >= 2 and len(labels) >= 2:
            findings.append({"severity": "high" if ing in SEVERE_DUP else "medium", "kind": "duplicate_ingredient",
                             "ingredient": ing, "items": labels,
                             "message": f"{', '.join(labels)} all contain {ing}. Taking them together doubles the dose."})
    # 2. two different drugs of the same class
    for cls, members in CLASSES.items():
        present = {}
        for k, it in enumerate(items):
            for ing in it["ingredients"]:
                if ing in members:
                    present.setdefault(ing, set()).add(it["label"])
        if len(present) >= 2 and not cls.startswith(("anticoagulant", "nitrate", "PDE5")):
            ings = sorted(present)
            findings.append({"severity": "high" if cls in ("NSAID", "opioid", "benzodiazepine-type sedative") else "medium",
                             "kind": "same_class", "class": cls, "ingredients": ings,
                             "message": f"Two different {cls}s together ({', '.join(ings)}): same effect and side effects, stacked."})
    # 3. dangerous combinations
    classes_present = {}
    for it in items:
        for ing in it["ingredients"]:
            for c in class_of(ing):
                classes_present.setdefault(c, set()).add(f"{ing} ({it['label']})")
    for a, b, sev, why in INTERACTIONS:
        if a in classes_present and b in classes_present:
            findings.append({"severity": sev, "kind": "interaction", "classes": [a, b],
                             "message": f"{'; '.join(sorted(classes_present[a]))} with {'; '.join(sorted(classes_present[b]))}: {why}."})
    # 4. daily total above an adult maximum (only when strength AND frequency are stated)
    totals, parts = {}, {}
    for it in items:
        if not it["per_day"]:
            continue
        for ing, mg in it["ingredients"].items():
            if mg and ing in MAX_DAILY_MG:
                totals[ing] = totals.get(ing, 0) + mg * it["per_day"]
                parts.setdefault(ing, []).append(f"{it['label']} {mg:g} mg x{it['per_day']}")
    for ing, tot in totals.items():
        lim, note = MAX_DAILY_MG[ing]
        if tot > lim:
            findings.append({"severity": "high", "kind": "daily_dose", "ingredient": ing, "total_mg": tot, "limit_mg": lim,
                             "message": f"About {tot:g} mg of {ing} per day ({' + '.join(parts[ing])}) exceeds the usual adult maximum of {lim} mg{('; ' + note) if note else ''}."})
    # 5. product-level warnings
    for it in items:
        p = it["product"]
        if not p:
            continue
        if "nimesulide" in p.ingredients and re.search(r"suspension|syrup|drops|paediatric|pediatric|kid", p.name, re.I):
            findings.append({"severity": "high", "kind": "banned_use", "items": [it["label"]],
                             "message": f"{p.name} is a liquid nimesulide product; India banned nimesulide for children under 12 in 2011."})
        if p.discontinued:
            findings.append({"severity": "low", "kind": "discontinued", "items": [it["label"]],
                             "message": f"{p.name} is listed as discontinued."})
        if not p.complete:
            msg = f"{p.name}: the open dataset lists only two ingredients for this product; a third may be missing."
            if p.missing_hint:
                msg += " Its name mentions a strength that matches no listed ingredient."
            findings.append({"severity": "info", "kind": "incomplete_data", "items": [it["label"]], "message": msg})
        if it["variants"] > 1:
            msg = f"'{it['label']}' matches {it['variants']} products; only ingredients common to all of them were used."
            sev = "info"
            if not it["ingredients"]:
                msg = (f"'{it['label']}' matches {it['variants']} products with no ingredient in common: this brand name is "
                       f"used for different drugs. Check the exact product and strength on the strip.")
                sev = "medium"
            findings.append({"severity": sev, "kind": "ambiguous", "items": [it["label"]], "message": msg})
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    findings.sort(key=lambda f: order[f["severity"]])
    return {"items": [{**{k: v for k, v in it.items() if k != "product"},
                       "product": it["product"].as_dict() if it["product"] else None} for it in items],
            "findings": findings,
            "verdict": "danger" if any(f["severity"] == "high" for f in findings) else
                       "caution" if any(f["severity"] == "medium" for f in findings) else "ok"}
