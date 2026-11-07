"""Tokenisation, Devanagari romanisation and phonetic keys.

Indian users write brand names in Latin letters, in Devanagari ("डोलो 650"), with
typos ("combiflem") or glued to the strength ("dolo650"). Matching works on a
phonetic key that is the same for all of these.
"""
import re

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*|[ऀ-ॣ०-ॿ]+|\S")
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
         "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
         "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
         "ल": "l", "ळ": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "क़": "q", "ख़": "kh",
         "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y"}
_VOW = {"अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
        "ऑ": "o", "ऋ": "ri", "ऍ": "e"}
_MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o",
          "ृ": "ri", "ॅ": "e"}
VIRAMA, NUKTA = "्", "़"


def romanize(word):
    """Rough Devanagari -> Latin (schwa deletion at word end)."""
    w = word.translate(_DEVA_DIGITS)
    out, i = [], 0
    while i < len(w):
        ch = w[i]
        if i + 1 < len(w) and w[i + 1] == NUKTA and ch + NUKTA in _CONS:
            ch, i = ch + NUKTA, i + 1
        if ch in _CONS:
            out.append(_CONS[ch])
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if nxt in _MATRA:
                out.append(_MATRA[nxt])
                i += 1
            elif nxt == VIRAMA:
                i += 1
            elif nxt and (nxt in _CONS or nxt in "ंँ"):
                out.append("a")
        elif ch in _VOW:
            out.append(_VOW[ch])
        elif ch in "ंँ":
            out.append("n")
        elif ch == "ः":
            out.append("h")
        elif ch.isascii():
            out.append(ch)
        i += 1
    return "".join(out)


def phonetic(word):
    """Spelling-insensitive key: 'Combiflam', 'kombiflam', 'कॉम्बिफ्लेम' -> similar keys."""
    w = word.lower()
    if re.search(r"[ऀ-ॿ]", w):
        w = romanize(w)
    w = re.sub(r"[^a-z0-9]", "", w)
    w = re.sub(r"c(?=[eiy])", "s", w)  # soft c: Crocin ~ krosin
    for a, b in (("chh", "ch"), ("ph", "f"), ("ck", "k"), ("q", "k"), ("c", "k"), ("z", "j"), ("w", "v"),
                 ("x", "ks"), ("y", "i"), ("th", "t"), ("dh", "d"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
                 ("sh", "s"), ("ai", "e"), ("ei", "e"), ("au", "o"), ("ou", "o"), ("oo", "u")):
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w)
    return w


# ---- Latin -> Devanagari (approximate), used only to synthesise training text
_L2D_CONS = [("chh", "छ"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"), ("dh", "ध"),
             ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("k", "क"), ("g", "ग"), ("j", "ज"), ("t", "ट"),
             ("d", "ड"), ("n", "न"), ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"),
             ("l", "ल"), ("v", "व"), ("w", "व"), ("s", "स"), ("h", "ह"), ("f", "फ़"), ("z", "ज़"),
             ("c", "क"), ("q", "क"), ("x", "क्स")]
_L2D_V = [("aa", "ा", "आ"), ("ai", "ै", "ऐ"), ("au", "ौ", "औ"), ("ee", "ी", "ई"), ("oo", "ू", "ऊ"),
          ("a", "", "अ"), ("i", "ि", "इ"), ("u", "ु", "उ"), ("e", "े", "ए"), ("o", "ो", "ओ")]


def to_devanagari(word):
    w, out, i, prev_cons = word.lower(), [], 0, False
    while i < len(w):
        if w[i].isdigit():
            out.append(w[i]); i += 1; prev_cons = False; continue
        for lat, matra, ind in _L2D_V:
            if w.startswith(lat, i):
                out.append(matra if prev_cons else ind)
                i += len(lat); prev_cons = False
                break
        else:
            for lat, dev in _L2D_CONS:
                if w.startswith(lat, i):
                    if prev_cons:
                        out.append("्")
                    out.append(dev)
                    i += len(lat); prev_cons = True
                    break
            else:
                i += 1; prev_cons = False
    return "".join(out)


def tokenize(text):
    return [(m.group(0), m.start(), m.end()) for m in TOKEN.finditer(text)]


# ---- character encoding for the context model
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789-'.,:;/()+%&!?") + [chr(c) for c in range(0x0900, 0x0980)]
CHAR_ID = {c: i + 2 for i, c in enumerate(CHARS)}  # 0 = pad, 1 = unknown
MAX_CHARS = 18
WINDOW = 10  # tokens either side of the candidate


def encode_token(tok):
    ids = [CHAR_ID.get(c, 1) for c in tok.lower()[:MAX_CHARS]]
    return ids + [0] * (MAX_CHARS - len(ids))


def shape(tok):
    """[is_capitalised, is_all_caps, has_digit, is_devanagari, is_number]"""
    return [float(tok[:1].isupper()), float(tok.isupper() and len(tok) > 1), float(any(c.isdigit() for c in tok)),
            float(bool(re.search(r"[ऀ-ॿ]", tok))), float(bool(re.fullmatch(r"[\d.]+", tok)))]


N_SHAPE = 7  # 5 token-shape features + 2 candidate features (well-known brand, family size)


def encode_window(tokens, ci, cj, cand=(0.0, 0.0)):
    """tokens: list of strings; candidate occupies [ci, cj). Returns (chars, shapes, mark) for 2W+1 slots.
    `cand` = (is a well-known brand, normalised product-family size), written into the candidate slot only."""
    L = WINDOW
    left = tokens[max(0, ci - L):ci]
    right = tokens[cj:cj + L]
    c = " ".join(tokens[ci:cj])
    slots = [""] * (L - len(left)) + left + [c] + right + [""] * (L - len(right))
    chars = [encode_token(t) if t else [0] * MAX_CHARS for t in slots]
    shapes = [(shape(t) if t else [0.0] * 5) + [0.0, 0.0] for t in slots]
    shapes[L][5], shapes[L][6] = float(cand[0]), float(cand[1])
    mark = [0.0] * len(slots)
    mark[L] = 1.0
    return chars, shapes, mark


def edit1(a, b):
    """True if a and b are within one edit (insert, delete, substitute or adjacent transposition)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]
