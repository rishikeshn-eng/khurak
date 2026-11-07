"""Export compact data files for the in-browser guard -> docs/data/

products.tsv   name, brand key, manufacturer index, flags, ingredient indices, strengths, mg
meta.json      ingredient names, manufacturer names, generic-name lookup, rule tables
context.onnx   the trained context model (+ threshold)
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from khurak.find import index  # noqa: E402
import importlib  # noqa: E402
C = importlib.import_module("khurak.check")  # the module, not the package-level function
from khurak.text import CHARS, MAX_CHARS, WINDOW  # noqa: E402
from khurak.normalize import canonical  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "data")


def main():
    idx = index()
    db = idx.db
    ings = db.ingredients
    ing_id = {n: i for i, n in enumerate(ings)}
    mfrs = sorted({p.manufacturer for p in db.products})
    mfr_id = {n: i for i, n in enumerate(mfrs)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "products.tsv"), "w", encoding="utf-8") as f:
        for p in db.products:
            flags = (("d" if p.discontinued else "") + ("" if p.complete else "i") + ("h" if p.missing_hint else "")
                     + ("f" if p.source == "full" else ""))
            f.write("\t".join([p.name, p.brand, str(mfr_id[p.manufacturer]), flags,
                               ",".join(str(ing_id[i]) for i in p.ingredients), "|".join(p.strengths),
                               ",".join("" if m is None else f"{m:g}" for m in p.mg)]) + "\n")
    meta = {
        "ingredients": ings, "manufacturers": mfrs, "generics": {g: canonical(g) for g in idx.generics},
        "max_daily": {k: v[0] for k, v in C.MAX_DAILY_MG.items()},
        "max_note": {k: v[1] for k, v in C.MAX_DAILY_MG.items() if v[1]},
        "classes": {k: sorted(v) for k, v in C.CLASSES.items()},
        "interactions": C.INTERACTIONS, "severe_dup": sorted(C.SEVERE_DUP),
        "chars": CHARS, "max_chars": MAX_CHARS, "window": WINDOW,
        "threshold": json.load(open(os.path.join(ROOT, "khurak", "data", "context.json")))["threshold"],
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    shutil.copy(os.path.join(ROOT, "khurak", "data", "context.onnx"), OUT)
    for f in ("products.tsv", "meta.json", "context.onnx"):
        print(f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")


if __name__ == "__main__":
    main()
