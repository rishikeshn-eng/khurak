"""Merge khurak results into docs/data/report.json (what the website renders)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.score import META, cut, parse_compose, canonical  # noqa: E402

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")
ROOT = os.path.join(HERE, "..")
LABELS = {"gemma-3-1b-it": "Gemma 3 · 1B", "qwen3-1.7b": "Qwen3 · 1.7B", "llama-3.2-3b-instruct": "Llama 3.2 · 3B",
          "phi-4-mini-instruct": "Phi-4 mini · 3.8B", "qwen3-8b": "Qwen3 · 8B", "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite"}
FAMOUS = ["Dolo", "Combiflam", "Crocin", "Pan", "Montair", "Sinarest", "Calpol", "Allegra", "Azithral", "Zerodol", "Augmentin"]


def famous():
    """Famous-brand probe: every local model asked what 13 household Indian medicines contain."""
    meta = json.load(open(os.path.join(HERE, "famous_meta.json")))
    order = [k for k in LABELS if os.path.exists(os.path.join(OUT, "famous", k + ".jsonl"))]
    clean = lambda t: " ".join(cut(t).replace("*", "").split())
    ans = {k: {json.loads(l)["id"]: clean(json.loads(l)["output"])
               for l in open(os.path.join(OUT, "famous", k + ".jsonl"), encoding="utf-8")} for k in order}
    rows, slips = [], []
    for pid, m in meta.items():
        gold = {canonical(x) for x in m["gold"]}
        cells = []
        for k in order:
            a = ans[k].get(pid, "")
            pred = parse_compose(a)
            real = {x for x in (pred or set()) if not x.startswith("?")}
            if pred is None:
                st = "unknown"
            elif not real or not real & gold:
                st = "wrong"            # invented ingredients or a different drug
            elif gold <= real and not real - gold:
                st = "correct"
            else:
                st = "partial"          # some right, some missing or invented
            cells.append({"s": st, "a": a[:120]})
            invented = bool(real - gold) or (pred is not None and not real)
            if (st == "wrong" or (st == "partial" and invented)) and len(a) < 110:
                slips.append({"model": LABELS[k], "brand": m["name"].replace(" Tablet", ""), "answer": a,
                              "truth": " + ".join(f"{g} {s}" for g, s in zip(m["gold"], m["strengths"]))})
        rows.append({"brand": m["name"], "truth": " + ".join(m["gold"]), "cells": cells})
    # one slip per model, prefer variety of brands
    picked, seen_m, seen_b = [], set(), set()
    for sl in slips:
        if sl["model"] not in seen_m and sl["brand"] not in seen_b:
            picked.append(sl); seen_m.add(sl["model"]); seen_b.add(sl["brand"])
    return {"models": [LABELS[k] for k in order], "rows": rows}, picked[:4]


def dup_quotes():
    out = []
    for f in sorted(os.listdir(os.path.join(OUT, "raw"))):
        name = f[:-6]
        if name not in LABELS or len(out) >= 3:
            continue
        for l in open(os.path.join(OUT, "raw", f), encoding="utf-8"):
            d = json.loads(l)
            m = META.get(d["id"])
            if not m or m["task"] != "dup" or m["shared"] != ["paracetamol"]:
                continue
            a = cut(d["output"]).strip()
            if "paracetamol" not in a.lower() and "acetaminophen" not in a.lower() and 30 < len(a) < 230 and \
                    any(w in a.lower() for w in ("yes", "can take", "safe", "fine", "okay")):
                out.append({"model": LABELS[name], "q": f"{m['a']} + {m['b']}", "a": a, "shared": "paracetamol"})
                break
    return out


def covered_acc(name, ids):
    """MCQA accuracy of one model on the Eka questions whose brand exists in the open data."""
    import re
    ok = n = 0
    for l in open(os.path.join(OUT, "raw", name + ".jsonl"), encoding="utf-8"):
        d = json.loads(l)
        if d["id"] in ids:
            g = re.search(r"\b([A-E])\b", cut(d["output"]).strip().upper()[:20])
            ok += bool(g) and g.group(1) == META[d["id"]]["answer"]
            n += 1
    return ok / n if n else None


def main():
    llm = json.load(open(os.path.join(OUT, "llm_report.json")))
    guard = json.load(open(os.path.join(OUT, "guard_report.json")))
    ids = set(guard["eka_lookup"]["covered_ids"])
    ctx = json.load(open(os.path.join(OUT, "context_model.json")))
    meta = json.load(open(os.path.join(ROOT, "khurak", "data", "meta.json")))
    order = list(LABELS)
    models = []
    for k in sorted(llm, key=lambda k: order.index(k) if k in order else 99):
        if k not in LABELS:
            continue
        v = llm[k]
        models.append({"name": k, "label": LABELS[k], "n_mcqa": v["n"]["mcqa"], "mcqa_covered": covered_acc(k, ids), **{x: v[x] for x in (
            "mcqa_acc", "compose_recall", "compose_exact", "compose_hallucinated", "dup_caught", "ctl_false_alarm",
            "dup_caught_grounded", "ctl_false_alarm_grounded")}})
    fam, fam_slips = famous()
    full = [m for m in models if m["dup_caught"] is not None]
    best_mcqa = max(models, key=lambda m: m["mcqa_acc"] or 0)
    avg_dup = sum(m["dup_caught"] for m in full) / len(full)
    avg_hall = sum(m["compose_hallucinated"] for m in full) / len(full)
    trunc_share = 1234 / (1234 + 3696)
    rep = {
        "kpis": [
            {"v": f"{avg_dup * 100:.0f}%", "l": "of double-paracetamol (and similar) risks noticed by the average model"},
            {"v": f"{avg_hall * 100:.0f}%", "l": "of answers to “what's in this brand?” name an ingredient that isn't there"},
            {"v": f"{guard['eka_lookup']['coverage'] * 100:.0f}%",
             "l": "of Eka Care's Indian-drug quiz brands even exist in the largest open medicine dataset"},
            {"v": f"{guard['dup_with_model']['dup_caught'] * 100:.0f}%", "l": "of the risky pairs caught by khurak's guard"},
        ],
        "slips": fam_slips, "famous": fam, "dup_quotes": dup_quotes(), "models": models,
        "guard": {"eka_acc": guard["eka_lookup"]["accuracy_on_covered"], "eka_coverage": guard["eka_lookup"]["coverage"],
                  "eka_covered": guard["eka_lookup"]["covered"], "dup_caught": guard["dup_with_model"]["dup_caught"],
                  "ctl_false_alarm": guard["dup_with_model"]["ctl_false_alarm"]},
        "trunc": {"total": meta["total"], "single": meta["single"], "repaired": meta["repaired"],
                  "unverified": meta["two_listed_unrepaired"], "truncated_share": trunc_share},
        "ctx": {"collide": "21%", "sub": "held-out brands in unseen message templates + real Wikipedia text",
                "rows": [{"l": "dictionary only: precision", "v": ctx["dictionary_only"]["precision"],
                          "fmt": f"{ctx['dictionary_only']['precision'] * 100:.0f}%"},
                         {"l": "khurak model: precision", "v": ctx["test"]["precision"], "fmt": f"{ctx['test']['precision'] * 100:.0f}%", "us": True},
                         {"l": "khurak model: recall", "v": ctx["test"]["recall"], "fmt": f"{ctx['test']['recall'] * 100:.0f}%", "us": True},
                         {"l": "spurious brands / Eka question, dictionary", "v": min(1, guard["eka_dictionary_only"]["spurious_brands_per_question"] / 5),
                          "fmt": f"{guard['eka_dictionary_only']['spurious_brands_per_question']:.1f}"},
                         {"l": "spurious brands / Eka question, khurak", "v": min(1, guard["eka_with_model"]["spurious_brands_per_question"] / 5),
                          "fmt": f"{guard['eka_with_model']['spurious_brands_per_question']:.2f}", "us": True}]},
        "bench_note": ("Greedy decoding, 4-bit local models on an Apple M4 (MLX), thinking mode off. MCQA: Eka Care's Indian Drug "
                       "MCQA (all 1,512). Ingredients: 300 brands with verified full compositions. Double dose: 200 question pairs "
                       "where both products share an active ingredient + 60 controls; a pass needs the shared ingredient named and a "
                       "warning. Gemini 3.5 Flash-Lite ran on the free API tier (~500 requests/day), so it covers a subset."),
    }
    os.makedirs(os.path.join(ROOT, "docs", "data"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "docs", "data", "report.json"), "w"), ensure_ascii=False)
    print(json.dumps(rep["kpis"], indent=1), "\nslips", len(rep["slips"]), "dup quotes", len(rep["dup_quotes"]))
    for sl in rep["slips"]:
        print("  ", sl["model"], "|", sl["brand"], "->", sl["answer"], "| truth:", sl["truth"])


if __name__ == "__main__":
    main()
