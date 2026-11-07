"""Evaluate khurak's guard end to end -> bench/out/guard_report.json

1. Eka Care Indian Drug MCQA (1,512 real benchmark questions): khurak finds the brand
   in the question, resolves it in its database and picks the option whose ingredients
   match. Same accuracy metric as the LLMs.
2. Duplicate-ingredient safety (the 260 questions the LLMs answered): does khurak's
   checker raise a duplicate-ingredient finding for the 200 risky pairs and stay quiet
   on the 60 controls?
3. Spurious mentions: brands reported that are not the ones in the question, per
   message, with and without the trained context model.
"""
import json
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib  # noqa: E402
F = importlib.import_module("khurak.find")  # the module, not the package-level function
from khurak.check import check  # noqa: E402
from khurak.normalize import canonical  # noqa: E402

HERE = os.path.dirname(__file__)
RAW = os.path.expanduser("~/Downloads/claude/rawdata2/meds")


def ingset(s):
    return {canonical(x) for x in re.split(r"\s*\+\s*", str(s)) if x.strip()}


def eka(threshold=None):
    df = pd.read_parquet(os.path.join(RAW, "eka_mcqa.parquet"))
    ok = found = resolved = 0
    spurious = 0
    rows = []
    for r in df.itertuples():
        res = F.find(r.question, threshold=threshold if threshold is not None else F.THRESHOLD)
        brand_first = str(r.medication_name).split()[0].lower()
        hits = [b for b in res["brands"] if b["key"] == brand_first or b["text"].lower() == brand_first]
        spurious += len(res["brands"]) - len(hits)
        gold = ingset(r.generic_name)
        pred = set()
        if hits:
            found += 1
            prods = hits[0]["products"]
            pred = set.intersection(*[set(p.ingredients) for p in prods]) or set(prods[0].ingredients)
            if gold <= pred or pred <= gold and pred:
                resolved += 1
        opts = {L: getattr(r, f"option_{L.lower()}") for L in "ABCDE"}
        opts = {L: ingset(o) for L, o in opts.items() if isinstance(o, str) and o.strip()}
        if pred:
            best = max(opts, key=lambda L: (len(opts[L] & pred) / len(opts[L] | pred), -len(opts[L] ^ pred)))
        else:
            best = None
        ok += int(best == r.correct_answer)
        rows.append({"q": r.question_id, "brand": r.medication_name, "found": bool(hits), "pred": sorted(pred),
                     "gold": sorted(gold), "answer": best, "correct": r.correct_answer})
    n = len(df)
    return {"n": n, "accuracy": ok / n, "brand_found": found / n, "composition_match": resolved / n,
            "spurious_brands_per_question": spurious / n,
            "errors": [x for x in rows if x["answer"] != x["correct"]][:30]}


def eka_lookup():
    """Database lookup on Eka's brands: coverage of the open data, and accuracy where the brand is covered.
    The brand is resolved from Eka's full product name (e.g. 'Rapeed 20mg Injection')."""
    df = pd.read_parquet(os.path.join(RAW, "eka_mcqa.parquet"))
    idx = F.index()
    covered, ok = [], 0
    for r in df.itertuples():
        toks = str(r.medication_name).split()
        key = re.sub(r"[^\w\-]+$", "", toks[0].lower())
        if key not in idx.db.brands:
            continue
        prods = F.resolve(key, toks[1:], [])
        pred = set.intersection(*[set(p.ingredients) for p in prods]) or set(prods[0].ingredients)
        opts = {L: ingset(getattr(r, f"option_{L.lower()}")) for L in "ABCDE"
                if isinstance(getattr(r, f"option_{L.lower()}"), str) and getattr(r, f"option_{L.lower()}").strip()}
        best = max(opts, key=lambda L: (len(opts[L] & pred) / len(opts[L] | pred), -len(opts[L] ^ pred)))
        covered.append("mcqa-" + r.question_id)
        ok += best == r.correct_answer
    return {"questions": len(df), "covered": len(covered), "coverage": len(covered) / len(df),
            "brands_covered": None, "accuracy_on_covered": ok / len(covered), "covered_ids": covered}


def dup():
    meta = json.load(open(os.path.join(HERE, "tasks_meta.json")))
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(os.path.join(HERE, "prompts.jsonl"))}
    caught = n_dup = false_alarm = n_ctl = 0
    misses = []
    for pid, m in meta.items():
        if m["task"] != "dup":
            continue
        r = check(prompts[pid])
        dups = {f["ingredient"] for f in r["findings"] if f["kind"] in ("duplicate_ingredient",)}
        cls = [f for f in r["findings"] if f["kind"] == "same_class"]
        if m["shared"]:
            n_dup += 1
            hit = bool(dups & set(m["shared"]))
            caught += hit
            if not hit and len(misses) < 20:
                misses.append({"q": prompts[pid][:160], "shared": m["shared"],
                               "items": [(i["label"], i["product"]["name"] if i["product"] else None) for i in r["items"]]})
        else:
            n_ctl += 1
            false_alarm += bool(dups)
    return {"dup_caught": caught / n_dup, "n_dup": n_dup, "ctl_false_alarm": false_alarm / n_ctl, "n_ctl": n_ctl,
            "misses": misses}


def main():
    t0 = time.time()
    rep = {"eka_lookup": eka_lookup(), "eka_with_model": eka(), "dup_with_model": dup()}
    el = rep["eka_lookup"]
    print(f"Eka coverage {el['coverage']:.3f} ({el['covered']}/{el['questions']}) | lookup accuracy on covered {el['accuracy_on_covered']:.4f}")
    F._model = F.ContextModel(path="/nonexistent")  # dictionary-only ablation
    rep["eka_dictionary_only"] = eka(threshold=0.0)
    for k in ("eka_with_model", "eka_dictionary_only"):
        e = rep[k]
        print(f"{k:22s} acc {e['accuracy']:.4f} brand found {e['brand_found']:.4f} composition {e['composition_match']:.4f} "
              f"spurious/q {e['spurious_brands_per_question']:.2f}")
    d = rep["dup_with_model"]
    print(f"dup caught {d['dup_caught']:.4f} ({d['n_dup']}) | control false alarm {d['ctl_false_alarm']:.4f} ({d['n_ctl']})")
    print(f"{time.time() - t0:.0f}s")
    json.dump(rep, open(os.path.join(HERE, "out", "guard_report.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
