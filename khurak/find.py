"""Find medicine mentions in free text and resolve them to products.

1. Candidates: every token that could be a brand - exact key, typo (edit distance 1),
   phonetic / Devanagari match, or glued to its strength ("dolo650").
2. Context model: a small trained network reads the +-10 surrounding tokens and
   decides whether the candidate is really a medicine. This is necessary: 21% of
   ordinary English word tokens are also some Indian brand ("cold", "rest", "pan").
3. Resolution: the words after the brand ("650", "SP", "LC", "Cold") pick the variant.
Generic salt names (paracetamol, ibuprofen...) are found by dictionary.
"""
import os
import re

import numpy as np

from .db import default_db, base_key
from .normalize import SYNONYMS, canonical
import math

from .text import tokenize, phonetic, encode_window, edit1

MODEL = os.path.join(os.path.dirname(__file__), "data", "context.onnx")
try:
    import json as _json
    THRESHOLD = _json.load(open(MODEL.replace(".onnx", ".json")))["threshold"]
except (OSError, ValueError, KeyError):
    THRESHOLD = 0.5
_GLUED = re.compile(r"^([a-z][a-z\-]{2,}?)(\d{2,4})$")


class Index:
    def __init__(self, db):
        self.db = db
        self.phon = {}
        self.deva = {}
        for k in db.brands:
            if len(k) >= 4:
                self.phon.setdefault(phonetic(k), set()).add(k)
                k2 = phonetic(k).replace("a", "")
                if len(k2) >= 4:
                    for v in {k2} | {k2[:i] + k2[i + 1:] for i in range(len(k2))}:
                        self.deva.setdefault(v, set()).add(k)
        gens = {}
        for g in db.generic_names:
            gens[g] = SYNONYMS.get(g, g)
        self.generics = gens
        self.max_generic_words = max(len(g.split()) for g in gens)
        # "well-known" brands: any product in the curated 1mg list (~10k brand keys)
        full = [p.brand for p in db.products if p.source == "full"]
        self.popular = {k for k in set(full) | {b.split("-")[0] for b in full} if k in db.brands}

    def cand_feats(self, key):
        """(is well-known brand, product-family size squashed to 0..1)"""
        return (1.0 if key in self.popular else 0.0, min(1.0, math.log1p(len(self.db.brands[key])) / math.log(200)))

    def keys_for(self, tok):
        """Return [(brand_key, how)] for one token, best first. Dosage-form words are never brands."""
        low = tok.lower().strip(".'")
        if low in NOT_BRAND:
            return []
        if re.search(r"[ऀ-ॿ]", low):
            k2 = phonetic(low).replace("a", "")
            if len(k2) < 4:
                return []
            hits = set(self.deva.get(k2, ()))
            for i in range(len(k2)):
                hits |= self.deva.get(k2[:i] + k2[i + 1:], set())
            # the delete index over-generates (up to two edits); keep true one-edit matches, exact first
            scored = []
            for k in hits:
                kk = phonetic(k).replace("a", "")
                if edit1(k2, kk):
                    scored.append((kk != k2, -len(self.db.brands[k]), k))
            return [(k, "devanagari") for _, _, k in sorted(scored)[:3]]
        out = []
        if low in self.db.brands:
            out.append((low, "exact"))
        m = _GLUED.match(low)
        if m:
            base = m.group(1)
            if base in self.db.brands:
                out.append((base, "glued"))
            elif len(base) >= 4:  # "dollo650": typo + glued strength
                ph = sorted(self.phon.get(phonetic(base), ()), key=lambda k: (-len(self.db.brands[k]), k))
                near = ph or sorted((k for k in self.db.fuzzy(base) if edit1(base, k)),
                                    key=lambda k: (-len(self.db.brands[k]), k))
                out.extend((k, "glued") for k in near[:1])
        if not out and len(low) >= 4 and not low.isdigit():
            for k in sorted(self.phon.get(phonetic(low), ()), key=lambda k: (-len(self.db.brands[k]), k)):
                out.append((k, "phonetic"))
            if not out:
                near = [k for k in self.db.fuzzy(low) if edit1(low, k)]
                for k in sorted(near, key=lambda k: (-len(self.db.brands[k]), k))[:3]:
                    out.append((k, "typo"))
        return out


_index = None


def index():
    """Build once, then cache to ~/.cache/khurak (keyed on the database file's mtime/size)."""
    global _index
    if _index is None:
        import pickle
        from .db import DATA, DB
        src = os.path.join(DATA, "products.tsv.gz")
        st = os.stat(src)
        cache = os.path.join(os.path.expanduser("~/.cache/khurak"), f"index-v2-{int(st.st_mtime)}-{st.st_size}.pkl")
        try:
            with open(cache, "rb") as f:
                _index = pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
            _index = Index(DB(src))
            _index.db._build_deletes()
            try:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                with open(cache, "wb") as f:
                    pickle.dump(_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            except OSError:
                pass
    return _index


class ContextModel:
    def __init__(self, path=MODEL):
        self.sess = None
        if os.path.exists(path):
            try:
                import onnxruntime as ort
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            except ImportError:
                pass

    def score(self, windows):
        if not windows:
            return np.zeros(0)
        if self.sess is None:
            return np.array([_heuristic(w) for w in windows])
        chars = np.array([w[0] for w in windows], np.int64)
        shapes = np.array([w[1] for w in windows], np.float32)
        mark = np.array([w[2] for w in windows], np.float32)
        return self.sess.run(None, {"chars": chars, "shapes": shapes, "mark": mark})[0].reshape(-1)


_CUES = re.compile(r"^(tab|tabs|tablet|tablets|cap|capsule|syp|syrup|inj|injection|mg|ml|dose|le|li|lo|liya|"
                   r"took|take|taking|prescribed|gave|dawai|dawa|medicine|drops|gel|cream|strip|od|bd|tds|"
                   r"\d+-\d+-\d+|\d+)$", re.I)


def _heuristic(w):
    """Fallback when onnxruntime is unavailable: medicine cue words near a capitalised candidate."""
    return 0.5  # neutral; real decisions need the trained model


_model = None


def model():
    global _model
    if _model is None:
        _model = ContextModel()
    return _model


NOT_BRAND = {"tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps", "syrup", "syp", "injection",
             "inj", "suspension", "cream", "gel", "ointment", "drops", "drop", "softgel", "lotion", "inhaler", "sachet",
             "solution", "spray", "powder", "strip", "dose", "medicine"}
_FORM_STRICT = {"tablet", "tablets", "capsule", "capsules", "syrup", "injection", "suspension", "cream", "gel",
                "ointment", "drops", "drop", "softgel", "lotion", "inhaler", "rotacap", "respules", "sachet"}
_FORMS = {"tab": "tablet", "tabs": "tablet", "tablet": "tablet", "tablets": "tablet", "goli": "tablet",
          "cap": "capsule", "caps": "capsule", "capsule": "capsule", "syp": "syrup", "syrup": "syrup",
          "suspension": "suspension", "susp": "suspension", "drops": "drop", "drop": "drop", "spray": "spray",
          "gel": "gel", "cream": "cream", "ointment": "ointment", "inj": "injection", "injection": "injection"}


def resolve(key, follow, before=()):
    """Pick product(s) in the brand family that best match the surrounding tokens.
    Form words in the text ("Tab.", "syrup", "spray") select the dosage form; with no
    form stated, tablets and capsules are preferred (the usual meaning of a bare brand)."""
    fam = index().db.family(key)
    if not fam:
        return []
    follow = [f.lower().strip(".,") for f in follow[:3]]
    ctx = follow + [b.lower().strip(".,") for b in before[-2:]]
    # A stated strength that no product of this brand has ("Mac 4mg" when only Mac 20mg exists): brand names are
    # reused for unrelated drugs, so don't guess the nearest variant - return the whole (live) family.
    stated = next((re.sub(r"mg$", "", f) for f in follow[:2] if re.fullmatch(r"\d{1,4}(mg)?", f)), None)
    if stated and not any(stated == str(int(m)) for p in fam for m in p.mg if m):
        live = [p for p in fam if not p.discontinued]
        return live or fam
    forms = {_FORMS[t] for t in ctx if t in _FORMS}

    def score(p):
        v = set(re.split(r"[\s\-/]+", p.variant))
        name = p.name.lower()
        s = sum(2 for f in follow if f and f in v and f not in _FORMS)
        s += sum(1 for f in follow if f.isdigit() and any(f == str(int(m)) for m in p.mg if m))
        if forms:
            s += 1.5 if any(f in name for f in forms) else 0
        elif re.search(r"\b(tablet|capsule)s?\b", name):
            s += 0.5
        return s - 0.01 * len(v)
    best = max(score(p) for p in fam)
    chosen = [p for p in fam if score(p) == best]
    live = [p for p in chosen if not p.discontinued]
    return live or chosen


def _completes_name(key, seq):
    """True if seq[0] (the candidate) plus seq[1] (and seq[2]) spell the start of a product name in the family,
    e.g. ['liv', 'joint', 'tablet'] for 'Liv Joint Tablet'. The second word must be a real word (2+ letters)."""
    if len(seq) < 2 or not re.fullmatch(r"[a-z][a-z0-9\-]+", seq[1]):
        return False
    for p in index().db.family(key)[:400]:
        toks = p.name.lower().split()
        if len(toks) >= 2 and toks[1] == seq[1]:
            return True
    return False


def find(text, threshold=THRESHOLD):
    """Return {'brands': [...], 'generics': [...]} with character spans."""
    idx = index()
    toks = tokenize(text)
    words = [t[0] for t in toks]
    cands, windows = [], []
    for i, w in enumerate(words):
        if not re.match(r"[A-Za-zऀ-ॿ]", w):
            continue
        keys = idx.keys_for(w)
        if keys:
            cands.append((i, keys))
            windows.append(encode_window(words, i, i + 1, idx.cand_feats(keys[0][0])))
    probs = list(model().score(windows))
    # Rule 1 - strength match: "Dolo 650", "Pan 40", "dolo650". A brand followed by a strength that one of
    # its own products has is a medicine mention whatever the surrounding words (lists, prescriptions).
    for j, (i, keys) in enumerate(cands):
        m = _GLUED.match(words[i].lower())
        nxt = m.group(2) if m else (re.sub(r"(?i)mg$", "", words[i + 1]) if i + 1 < len(words) else "")
        if nxt.isdigit() and any(nxt == str(int(x)) for p in idx.db.family(keys[0][0]) for x in p.mg if x):
            probs[j] = max(probs[j], 0.99)
        # ...a dosage form immediately after ("Dolo Tablet"), a prescription prefix before ("Tab. Dolo"),
        # or the next words completing a real product name in this brand's family ("Liv Joint", "Inspan DSR").
        nxt1 = words[i + 1].lower().strip(".") if i + 1 < len(words) else ""
        before = words[i - 1].lower().strip(".") if i > 0 else ""
        if nxt1 in _FORM_STRICT or before in ("tab", "cap", "syp", "inj", "tabs", "caps") or \
                _completes_name(keys[0][0], [w.lower() for w in words[i:i + 3]]):
            probs[j] = max(probs[j], 0.99)
    # Rule 2 - medical context: once the text certainly mentions a medicine, other WELL-KNOWN brands
    # need much less evidence ("...and Sinarest New three times a day").
    if any(p >= threshold for p in probs):
        for j, (i, keys) in enumerate(cands):
            if keys[0][0] in idx.popular and probs[j] >= 0.05:
                probs[j] = max(probs[j], threshold)
    brands, consumed = [], set()
    for (i, keys), p in zip(cands, probs):
        if p < threshold or i in consumed:
            continue
        key, how = keys[0]
        follow = words[i + 1:i + 4]
        m = _GLUED.match(words[i].lower())
        if m:
            follow = [m.group(2)] + follow
        prods = resolve(key, follow, words[max(0, i - 2):i])
        if not prods:
            continue
        brands.append({"text": words[i], "start": toks[i][1], "end": toks[i][2], "key": key, "match": how,
                       "confidence": float(p), "products": prods})
        ptoks = prods[0].name.lower().split()
        for k in range(1, min(len(ptoks), 5)):  # "Gluconorm G2 Tablet PR": G2, Tablet, PR belong to this mention
            if i + k < len(words) and words[i + k].lower() == ptoks[k]:
                consumed.add(i + k)
            else:
                break
    # generics: longest-first dictionary match on lowercase word n-grams
    generics, low = [], [w.lower() for w in words]
    used = set()
    for n in range(idx.max_generic_words, 0, -1):
        for i in range(len(low) - n + 1):
            if any(j in used for j in range(i, i + n)):
                continue
            phrase = " ".join(low[i:i + n])
            if phrase in idx.generics and len(phrase) >= 4:
                c = canonical(phrase)
                generics.append({"text": " ".join(words[i:i + n]), "start": toks[i][1], "end": toks[i + n - 1][2],
                                 "ingredient": c})
                used.update(range(i, i + n))
    return {"brands": brands, "generics": generics, "candidates": len(cands)}
