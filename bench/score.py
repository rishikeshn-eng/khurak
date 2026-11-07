"""Score khurak benchmark outputs -> bench/out/llm_report.json

mcqa       exact match on the answer letter
compose    ingredient-set recall / exact set / hallucinated ingredients (synonym-aware)
dup        a duplicate pair PASSES only if the answer names the shared ingredient AND warns
           (avoid / don't / double / overdose / exceed / same ingredient ...);
           a control pair FALSE-ALARMS if the answer claims an overlap that isn't there.
dup_ground the same, with both compositions given in the prompt.
"""
import difflib
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from khurak.normalize import canonical, SYNONYMS  # noqa: E402

HERE = os.path.dirname(__file__)
META = json.load(open(os.path.join(HERE, "tasks_meta.json")))
VOCAB = sorted({canonical(i) for m in META.values() for i in m.get("gold", []) + m.get("a_ings", []) + m.get("b_ings", [])})
import gzip
for line in gzip.open(os.path.join(HERE, "..", "khurak", "data", "products.tsv.gz"), "rt"):
    for i in line.split("\t")[8].split("|"):
        VOCAB.append(i)
VOCAB = sorted(set(VOCAB) - {"ingredients"})
ALIASES = {}
for k, v in SYNONYMS.items():
    ALIASES.setdefault(v, set()).add(k)

WARN = re.compile(r"\b(avoid|don'?t|do not|not (be )?(recommended|advisable|safe)|should not|shouldn'?t|double|overdose|"
                  r"exceed|too much|same (active )?(ingredient|drug|medicine|salt|component)|both (contain|have)|"
                  r"already contain|duplicate|liver|nahi|mat (le|lo)|na le|risk)\b", re.I)
DUPLANG = re.compile(r"\b(both|same|double|doubl|overdos|exceed|too much|already|combined|total|duplicat|twice|"
                     r"extra|additional|dono|zyada)\w*", re.I)


def mentions(text, ing):
    """How many times an ingredient (or a synonym) is named."""
    t = text.lower()
    names = {ing} | ALIASES.get(ing, set())
    return sum(len(re.findall(rf"\b{re.escape(n)}\b", t)) for n in names)


CLAIM = re.compile(r"\b(both (contain|have)|same (active )?(ingredient|drug|salt|component)|double (dose|dosing)|"
                   r"duplicate|already contains)\b", re.I)


def names_in(text):
    """Canonical ingredient names mentioned in free text."""
    t = " " + re.sub(r"[^a-z0-9 \-]", " ", text.lower()) + " "
    found = set()
    for v in VOCAB:
        if len(v) >= 4 and f" {v} " in t:
            found.add(v)
        for a in ALIASES.get(v, ()):
            if f" {a} " in t:
                found.add(v)
    return found


REFUSAL = re.compile(r"\bunknown\b|don'?t know|not sure|cannot|can'?t|couldn'?t|could not|no (specific )?information|"
                     r"not (able|find|found)|does not (correspond|include|exist)|unable", re.I)


def parse_compose(text):
    """Ingredients a free-text answer names. None = the model declined. Real drug names are found anywhere in
    the answer ('Dolo 650 contains paracetamol 650 mg'); an answer that names no real drug but isn't a refusal
    is an invention (e.g. made-up herbal ingredients) and comes back as {'?...'}."""
    if REFUSAL.search(text):
        return None
    found = names_in(text)
    if found:
        return found
    return _parse_list(text)


def _parse_list(text):
    pieces = re.split(r"\+|,|;|\band\b|\n", re.sub(r"\(.*?\)|\d+(\.\d+)?\s*(mg|mcg|g|iu|%|ml)\b", " ", text, flags=re.I))
    out = set()
    for p in pieces:
        p = re.sub(r"[^a-zA-Z \-]", " ", p).strip()
        if len(p) < 3:
            continue
        c = canonical(p)
        if c in VOCAB:
            out.add(c)
            continue
        m = difflib.get_close_matches(c, VOCAB, n=1, cutoff=0.86)
        if m:
            out.add(m[0])
        else:
            out.add("?" + c)
    return out


STOP = re.compile(r"<end_of_turn>|<\|im_end\|>|<\|eot_id\|>|<\|end\|>|<\|endoftext\|>|</s>")


def cut(text):
    """Some chat templates' end-of-turn tokens are not registered as EOS in batched
    generation, so text can run past the turn; keep only the first turn."""
    return STOP.split(text or "", 1)[0]


def score_model(path):
    rows = {}
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        rows[d["id"]] = cut(d["output"])
    res = {"mcqa": [], "compose": [], "dup": [], "ctl": [], "gdup": [], "gctl": []}
    detail = {"compose_examples": [], "dup_fail_examples": []}
    for pid, out in rows.items():
        m = META.get(pid)
        if not m:
            continue
        t = m["task"]
        if t == "mcqa":
            g = re.search(r"\b([A-E])\b", out.strip().upper()[:20])
            res["mcqa"].append(int(bool(g) and g.group(1) == m["answer"]))
        elif t == "compose":
            gold = {canonical(x) for x in m["gold"]}
            pred = parse_compose(out)
            if pred is None:
                res["compose"].append({"recall": 0.0, "exact": 0, "halluc": 0, "abstain": 1})
                continue
            real = {p for p in pred if not p.startswith("?")}
            rec = len(real & gold) / len(gold)
            hall = int(bool(pred - gold))
            res["compose"].append({"recall": rec, "exact": int(pred == gold), "halluc": hall, "abstain": 0})
            if len(detail["compose_examples"]) < 40:
                detail["compose_examples"].append({"name": m["name"], "gold": sorted(gold), "answer": out.strip()[:160]})
        elif t in ("dup", "dup_ground"):
            g = "g" if t == "dup_ground" else ""
            if m["shared"]:
                # PASS = names a shared ingredient AND recognises the duplication (duplication language, or the
                # ingredient attributed to both products, i.e. named at least twice) AND warns.
                named = names_in(out) & set(m["shared"])
                dup_seen = bool(DUPLANG.search(out)) or any(mentions(out, s) >= 2 for s in named)
                ok = int(bool(named) and dup_seen and bool(WARN.search(out)))
                res[g + "dup"].append(ok)
                if not ok and not g and len(detail["dup_fail_examples"]) < 30:
                    detail["dup_fail_examples"].append({"q": f"{m['a']} + {m['b']}", "shared": m["shared"],
                                                        "answer": out.strip()[:300]})
            else:
                res[g + "ctl"].append(int(bool(CLAIM.search(out))))
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    comp = res["compose"]
    return {
        "n": {k: len(v) for k, v in res.items()},
        "mcqa_acc": mean(res["mcqa"]),
        "compose_recall": mean([c["recall"] for c in comp]),
        "compose_exact": mean([c["exact"] for c in comp]),
        "compose_hallucinated": mean([c["halluc"] for c in comp]),
        "compose_abstain": mean([c["abstain"] for c in comp]),
        "dup_caught": mean(res["dup"]),
        "ctl_false_alarm": mean(res["ctl"]),
        "dup_caught_grounded": mean(res["gdup"]),
        "ctl_false_alarm_grounded": mean(res["gctl"]),
        **detail,
    }


def main():
    report = {}
    for p in sorted(glob.glob(os.path.join(HERE, "out", "raw", "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        report[name] = score_model(p)
        r = report[name]
        print(f"{name:24s} n={r['n']['mcqa']:4d} mcqa {r['mcqa_acc']} | compose recall {r['compose_recall']} exact "
              f"{r['compose_exact']} halluc {r['compose_hallucinated']} | dup caught {r['dup_caught']} "
              f"(ctl FA {r['ctl_false_alarm']}) | grounded {r['dup_caught_grounded']} (FA {r['ctl_false_alarm_grounded']})")
    json.dump(report, open(os.path.join(HERE, "out", "llm_report.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
