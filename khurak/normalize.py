"""Ingredient-name normalisation.

Indian labels use British/older names (Amoxycillin, Frusemide, Lignocaine), salt
forms (Diclofenac Sodium vs Diclofenac Potassium) and odd spellings. Two products
share an active ingredient if their *active moieties* match, so we strip salt
words and map synonyms to one canonical name.
"""
import re

SYNONYMS = {
    "acetaminophen": "paracetamol", "amoxycillin": "amoxicillin", "frusemide": "furosemide",
    "lignocaine": "lidocaine", "albuterol": "salbutamol", "levalbuterol": "levosalbutamol",
    "epinephrine": "adrenaline", "norepinephrine": "noradrenaline", "glyburide": "glibenclamide",
    "cephalexin": "cefalexin", "cefalexine": "cefalexin", "ciprofloxacine": "ciprofloxacin",
    "chlorpheniramine": "chlorphenamine", "chlorpheniramine maleate": "chlorphenamine",
    "chlorphenamine maleate": "chlorphenamine", "beclomethasone": "beclometasone",
    "dicyclomine": "dicycloverine", "thyroxine": "levothyroxine", "l-thyroxine": "levothyroxine",
    "vitamin c": "ascorbic acid", "vitamin b12": "cyanocobalamin", "mecobalamin": "methylcobalamin",
    "cholecalciferol": "vitamin d3", "colecalciferol": "vitamin d3", "aspirin": "acetylsalicylic acid",
    "acetyl salicylic acid": "acetylsalicylic acid", "ascorbic acid": "ascorbic acid",
    "guaiphenesin": "guaifenesin", "sulphamethoxazole": "sulfamethoxazole", "rifampin": "rifampicin",
    "isoniazide": "isoniazid", "metronidazol": "metronidazole", "tinidazol": "tinidazole",
    "ambroxol hydrochloride": "ambroxol", "levocetrizine": "levocetirizine", "cetrizine": "cetirizine",
    "montelukast sodium": "montelukast", "pantoprazole sodium": "pantoprazole",
    "dextromethorphan hydrobromide": "dextromethorphan", "phenylephrine hydrochloride": "phenylephrine",
}

# Salt / hydrate / ester words that don't change the active moiety.
SALT_WORDS = {
    "sodium", "potassium", "calcium", "magnesium", "hydrochloride", "hcl", "hydrobromide", "maleate",
    "sulphate", "sulfate", "besylate", "besilate", "succinate", "tartrate", "mesylate", "mesilate",
    "citrate", "fumarate", "phosphate", "acetate", "dihydrate", "trihydrate", "monohydrate",
    "hemihydrate", "anhydrous", "bitartrate", "dipropionate", "propionate", "furoate", "valerate",
    "proxetil", "axetil", "pivoxil", "trihydrochloride", "dihydrochloride", "malate", "lactate",
    "gluconate", "nitrate", "bromide", "chloride", "hyclate", "monosodium", "disodium", "cilexetil",
    "medoxomil", "olamine", "pamoate", "embonate", "stearate", "estolate", "ethylsuccinate", "base",
}
# Salt words that ARE the active ingredient on their own (don't strip to nothing).
_KEEP_WHOLE = {"potassium chloride", "sodium chloride", "calcium carbonate", "magnesium hydroxide",
               "sodium bicarbonate", "calcium citrate", "zinc sulphate", "ferrous sulphate",
               "magnesium sulphate", "potassium citrate", "sodium citrate", "calcium gluconate"}

_PAREN = re.compile(r"\(([^)]*)\)")
_STRENGTH = re.compile(r"([\d.]+)\s*(mg|mcg|µg|g|iu|%|ml|units?)\b", re.I)


def canonical(name: str) -> str:
    """'Diclofenac Sodium' -> 'diclofenac'; 'Amoxycillin' -> 'amoxicillin'."""
    n = re.sub(r"[^a-z0-9 \-]", " ", name.lower())
    n = re.sub(r"\s+", " ", n).strip()
    if n in SYNONYMS:
        return SYNONYMS[n]
    if n in _KEEP_WHOLE:
        return n
    words = [w for w in n.split() if w not in SALT_WORDS]
    n2 = " ".join(words) if words else n
    return SYNONYMS.get(n2, n2)


def parse_component(text: str):
    """'Paracetamol (650mg)' -> ('paracetamol', 650.0, 'mg', 'Paracetamol', '650mg')."""
    text = (text or "").strip().strip("+").strip()
    if not text:
        return None
    m = _PAREN.search(text)
    raw_name = _PAREN.sub("", text).strip()
    strength_txt = m.group(1).strip() if m else ""
    val, unit = None, None
    sm = _STRENGTH.search(strength_txt)
    if sm and "/" not in strength_txt.split(sm.group(0))[0]:
        val, unit = float(sm.group(1)), sm.group(2).lower()
        if unit == "g":
            val, unit = val * 1000, "mg"
        if unit in ("mcg", "µg"):
            val, unit = val / 1000, "mg"
    per_volume = "/" in strength_txt  # e.g. 250mg/5ml — a concentration, not a unit dose
    return {"name": canonical(raw_name), "label": raw_name, "strength": strength_txt,
            "mg": val if (unit == "mg" and not per_volume) else None}


def split_composition(comp: str):
    return [p for p in (parse_component(x) for x in re.split(r"\s\+\s|\+", comp or "")) if p]
