import pytest

from khurak.normalize import canonical, split_composition
from khurak.text import phonetic, romanize, tokenize


# ---------------------------------------------------------------- pure functions (fast)
@pytest.mark.parametrize("raw,canon", [
    ("Amoxycillin", "amoxicillin"), ("Diclofenac Sodium", "diclofenac"), ("Diclofenac Potassium", "diclofenac"),
    ("Chlorpheniramine Maleate", "chlorphenamine"), ("Acetaminophen", "paracetamol"), ("Frusemide", "furosemide"),
    ("Potassium Chloride", "potassium chloride"), ("Cefpodoxime Proxetil", "cefpodoxime"),
])
def test_canonical(raw, canon):
    assert canonical(raw) == canon


def test_split_composition_units():
    parts = split_composition("Paracetamol (650mg) + Caffeine (50mg)")
    assert [(p["name"], p["mg"]) for p in parts] == [("paracetamol", 650.0), ("caffeine", 50.0)]
    syrup = split_composition("Ambroxol (30mg/5ml)")[0]
    assert syrup["mg"] is None  # a concentration is not a unit dose


@pytest.mark.parametrize("latin,deva", [("Dolo", "डोलो"), ("Crocin", "क्रोसिन"), ("Sinarest", "सिनारेस्ट"),
                                         ("Zerodol", "ज़ेरोडोल")])
def test_phonetic_matches_across_scripts(latin, deva):
    assert phonetic(latin) == phonetic(deva)


@pytest.mark.parametrize("token,key", [("डोलो", "dolo"), ("क्रोसिन", "crocin"), ("मोंटेयर", "montair"),
                                       ("combiflem", "combiflam"), ("dollo650", "dolo"), ("Dolo650", "dolo")])
def test_spelling_variants_resolve(token, key):
    from khurak.find import index
    assert index().keys_for(token)[0][0] == key


def test_romanize_and_tokenize():
    assert romanize("डोलो") == "dolo"
    toks = [t for t, _, _ in tokenize("Mujhe Dolo650 li thi, डोलो ६५०")]
    assert "Dolo650" in toks and "डोलो" in toks


# ---------------------------------------------------------------- database + guard (needs data files)
@pytest.fixture(scope="module")
def chk():
    from khurak.check import check
    return check


def items(r):
    return {i["label"].lower(): i for i in r["items"]}


def test_brand_resolution(chk):
    r = chk("I took Dolo 650 for fever.")
    it = items(r)["dolo"]
    assert it["product"]["name"] == "Dolo 650 Tablet"
    assert it["ingredients"] == {"paracetamol": 650.0}


def test_context_rejects_common_words(chk):
    r = chk("I have a cold and need some rest, so I took Dolo 650 for the fever.")
    found = set(items(r))
    assert "dolo" in found
    assert not found & {"cold", "rest", "some", "fever", "and", "for", "the"}


def test_duplicate_paracetamol_flagged(chk):
    r = chk("Took Dolo 650 in the morning. Can I take Sinarest New tablet for my cold?")
    dup = [f for f in r["findings"] if f["kind"] == "duplicate_ingredient"]
    assert dup and dup[0]["ingredient"] == "paracetamol" and dup[0]["severity"] == "high"
    assert r["verdict"] == "danger"


def test_daily_dose_over_limit(chk):
    r = chk("Doctor said Dolo 650 1-1-1-1 and Sinarest New three times a day.")
    dose = [f for f in r["findings"] if f["kind"] == "daily_dose"]
    assert dose and dose[0]["ingredient"] == "paracetamol" and dose[0]["total_mg"] > 4000


def test_same_class_nsaids(chk):
    r = chk("Taking Brufen 400 and Voveran 50 together for back pain.")
    assert any(f["kind"] == "same_class" and f["class"] == "NSAID" for f in r["findings"])


def test_safe_pair_is_ok(chk):
    r = chk("Pan 40 before breakfast and Allegra 120 at night.")
    assert r["verdict"] == "ok"
    assert {"pan", "allegra"} <= set(items(r))


def test_devanagari_brand(chk):
    r = chk("मैंने डोलो 650 ली थी, क्या क्रोसिन भी ले सकता हूँ?")
    assert any(f["kind"] == "duplicate_ingredient" and f["ingredient"] == "paracetamol" for f in r["findings"])


def test_incomplete_composition_warned(chk):
    r = chk("Can I take Vicks Action 500 Advanced tablet?")
    assert any(f["kind"] == "incomplete_data" and "matches no listed ingredient" in f["message"] for f in r["findings"])
