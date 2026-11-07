"""Build training / test data for the context model -> pipeline/ctx_{train,test}.npz

Positives: real brand surface forms placed into message templates (Gemini-written
           natural messages + hand-written prescription / Hindi / chatbot patterns).
Negatives: every OTHER candidate token in those messages ("cold", "rest", "pain"...)
           plus every candidate token in real English and Hindi Wikipedia sentences.
Splits: templates 80/20 and brand keys 85/15, so the test set uses unseen templates
        AND unseen brands. Wikipedia articles are split by article.
"""
import hashlib
import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from khurak.find import index  # noqa: E402
from khurak.text import tokenize, encode_window, to_devanagari  # noqa: E402

HERE = os.path.dirname(__file__)
RAW = os.path.expanduser("~/Downloads/claude/rawdata2/meds")
rng = random.Random(7)
idx = index()
db = idx.db

HAND = [
    "Tab. [MED1] 1-0-1 x 5 days", "Tab [MED1] OD after food", "Syp. [MED1] 5 ml TDS", "Cap. [MED1] 0-0-1 for 7 days",
    "Rx: [MED1] BD, [MED2] SOS for fever", "Inj. [MED1] stat", "[MED1] 1 tab twice daily after meals",
    "Mujhe bukhar hai, [MED1] le sakta hoon?", "kal raat [MED1] li thi par sar dard abhi bhi hai",
    "doctor ne [MED1] aur [MED2] diya hai, khana khane ke baad lena hai na?",
    "मैंने [MED1] ली थी, क्या अब [MED2] ले सकता हूँ?", "बच्चे को [MED1] सिरप कितना देना है?",
    "डॉक्टर ने [MED1] लिखी है, यह किस लिए है?", "[MED1] खाने के बाद लेनी है या पहले?",
    "Is [MED1] safe in pregnancy?", "What is [MED1] used for?", "Can I take [MED1] with alcohol?",
    "[MED1] contains which salt?", "Bhaiya ek strip [MED1] aur ek [MED2] bhej do",
    "price of [MED1] has gone up again", "[MED1] is out of stock at the chemist, any substitute?",
    "My mother takes [MED1] for sugar and [MED2] for BP. Is that okay?",
    "[MED1] se neend bahut aati hai kya?", "took [MED1] at 8am, fever still 101",
    "The doctor switched me from [MED1] to [MED2].", "Discharge meds: [MED1], [MED2], continue for 2 weeks.",
    "[MED1] lene ke baad pet mein jalan ho rahi hai", "can i give [MED1] to my 5 year old for cold?",
    "Sir [MED1] ka generic kya hai?", "I have a cold and need rest, should I take [MED1]?",
    "after the pain started I had [MED1] and some rest", "is [MED1] better than [MED2] for joint pain?",
]
# sentences where colliding ordinary words appear WITHOUT any brand (pure hard negatives)
NEG_HAND = [
    "I have a bad cold and fever since yesterday, need some rest", "Take plenty of rest and drink water",
    "my sugar levels are high in the morning", "the pain in my back is getting worse", "Life has been busy",
    "He is very active and plays every day", "my head is heavy and body is aching", "plus I have a sore throat",
    "Dawai khane ke baad leni hai", "thoda rest karo aur paani piyo", "sugar control mein nahi hai",
    "Total cost was 450 rupees", "she needs care and calm sleep", "Is it safe to travel with a cold?",
]


def bucket(s, mod=100):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def surface(p):
    """A realistic way a person writes this product."""
    toks = p.name.split()
    brand = toks[0]
    rest = [t for t in toks[1:] if t.lower() not in ("tablet", "capsule", "syrup", "injection", "suspension", "oral",
                                                     "drops", "drop", "cream", "gel", "ointment", "solution")]
    r = rng.random()
    if r < 0.30:
        s = brand
    elif r < 0.70:
        s = " ".join([brand] + rest[:2])
    else:
        s = p.name
    if rng.random() < 0.10:
        s = rng.choice(["Tab. ", "Tab ", "Syp. ", "Cap. "]) + s
    r = rng.random()
    if r < 0.25:
        s = s.lower()
    elif r < 0.30:
        s = s.upper()
    if rng.random() < 0.08 and len(brand) >= 5:  # typo in the brand word
        b = s.split()[0] if not s.startswith(("Tab", "Syp", "Cap", "TAB", "tab")) else None
        if b and len(b) >= 5:
            i = rng.randrange(1, len(b) - 1)
            op = rng.choice(["drop", "dup", "swap"])
            nb = b[:i] + b[i + 1:] if op == "drop" else (b[:i] + b[i] + b[i:] if op == "dup" else
                                                        b[:i - 1] + b[i] + b[i - 1] + b[i + 1:])
            s = s.replace(b, nb, 1)
    m = re.match(r"^([A-Za-z]+) (\d{2,4})\b", s)
    if m and rng.random() < 0.25:
        s = s.replace(m.group(0), m.group(1) + m.group(2), 1)  # glued: dolo650
    return s


def fill(tpl, brands_ok, deva, popular_ok):
    """Return (text, [(start,end) of brand words]) or None.
    Real messages mostly mention well-known brands, so 40% of fills come from that list."""
    spans, text = [], tpl
    for slot in ("[MED1]", "[MED2]"):
        if slot not in text:
            continue
        key = rng.choice(popular_ok) if rng.random() < 0.4 else rng.choice(brands_ok)
        p = rng.choice(db.family(key))
        s = surface(p)
        if deva and rng.random() < 0.35:
            w = re.sub(r"[^A-Za-z]", "", p.name.split()[0])
            s = to_devanagari(w) + (" " + " ".join(s.split()[1:2]) if len(s.split()) > 1 else "")
        pos = text.index(slot)
        text = text[:pos] + s + text[pos + len(slot):]
        # brand word = first token of s that isn't a prescription prefix
        off = 0
        for t, a, b in tokenize(s):
            if t.lower().strip(".") in ("tab", "syp", "cap", "inj"):
                continue
            if t == ".":
                continue
            spans.append((pos + a, pos + b, key))
            break
        for i, (a, b, k) in enumerate(spans[:-1]):
            if a > pos:
                spans[i] = (a + len(s) - len(slot), b + len(s) - len(slot), k)
    return text, spans


def examples_from(text, spans):
    toks = tokenize(text)
    words = [t[0] for t in toks]
    out = []
    for i, (w, a, b) in enumerate(toks):
        if not re.match(r"[A-Za-zऀ-ॿ]", w):
            continue
        keys = idx.keys_for(w)
        if not keys:
            continue
        label = int(any(a >= s and b <= e for s, e, _ in spans))
        out.append((encode_window(words, i, i + 1, idx.cand_feats(keys[0][0])), label))
    return out


def wiki_sentences(path, n_articles, lang):
    df = pd.read_parquet(path, columns=["id", "text"]).head(n_articles)
    for aid, txt in zip(df.id, df.text):
        for s in re.split(r"(?<=[.!?।])\s+", txt)[:40]:
            if 25 <= len(s) <= 300 and not re.search(r"\b(mg|tablet|drug|medicine|pharma|dose)\b", s, re.I):
                yield str(aid), s


def main():
    keys = [k for k in db.brands if re.match(r"^[a-z]", k)]
    train_keys = [k for k in keys if bucket(k) < 85]
    test_keys = [k for k in keys if bucket(k) >= 85]
    pop_train = [k for k in train_keys if k in idx.popular]
    pop_test = [k for k in test_keys if k in idx.popular]
    gem = [json.loads(l)["text"] for l in open(os.path.join(HERE, "templates_gemini.jsonl"), encoding="utf-8")]
    gem = list(dict.fromkeys(gem))
    tpls = [(t, "gemini") for t in gem] + [(t, "hand") for t in HAND]
    tr_t = [t for t in tpls if bucket(t[0]) < 80]
    te_t = [t for t in tpls if bucket(t[0]) >= 80]
    print("templates train/test", len(tr_t), len(te_t), "| brand keys train/test", len(train_keys), len(test_keys))

    data = {"train": [], "test": []}
    for split, T, K, PK, reps in (("train", tr_t, train_keys, pop_train, 70), ("test", te_t, test_keys, pop_test, 25)):
        for tpl, src in T:
            deva = bool(re.search(r"[ऀ-ॿ]", tpl))
            for _ in range(reps):
                r = fill(tpl, K, deva, PK)
                if r:
                    for ex in examples_from(*r):
                        data[split].append(ex + ("template",))
        for t in NEG_HAND:
            for ex in examples_from(t, []):
                data[split].append(ex + ("hardneg",))
    for path, lang, n in ((os.path.join(RAW, "wiki_en.parquet"), "en", 4000), (os.path.join(RAW, "wiki_hi.parquet"), "hi", 4000)):
        for aid, s in wiki_sentences(path, n, lang):
            split = "test" if bucket(aid) >= 85 else "train"
            for ex in examples_from(s, []):
                data[split].append(ex + ("wiki_" + lang,))
    for split, rows in data.items():
        rng.shuffle(rows)
        if split == "train":  # cap wiki negatives at 4x positives
            pos = sum(r[1] for r in rows)
            keep, wiki = [], 0
            for r in rows:
                if r[2].startswith("wiki"):
                    wiki += 1
                    if wiki > 4 * pos:
                        continue
                keep.append(r)
            rows = keep
        chars = np.array([r[0][0] for r in rows], np.uint8)
        shapes = np.array([r[0][1] for r in rows], np.uint8)
        y = np.array([r[1] for r in rows], np.uint8)
        src = np.array([r[2] for r in rows])
        np.savez_compressed(os.path.join(HERE, f"ctx_{split}.npz"), chars=chars, shapes=shapes, y=y, src=src)
        from collections import Counter
        print(split, len(rows), "pos", int(y.sum()), dict(Counter(src.tolist())))


if __name__ == "__main__":
    main()
