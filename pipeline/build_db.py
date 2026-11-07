"""Build khurak's product database from the open Indian medicine datasets.

Sources
  * A-Z Medicine Dataset of India (junioralive/Indian-Medicine-Dataset, MIT): 253,973 products.
    Its `short_composition1/2` columns keep only the FIRST TWO ingredients of a product.
  * 1mg subset (dmedhi/indian-medicines, 11,825 products) and the A-Z `salt_composition`
    column (7,488 rows): full compositions, used to repair truncated products.

Output: khurak/data/products.tsv.gz  (one row per product)
        khurak/data/meta.json         (provenance + truncation statistics)
"""
import gzip
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from khurak.normalize import split_composition  # noqa: E402

RAW = os.path.expanduser(os.environ.get("KHURAK_RAW", "~/Downloads/claude/rawdata2/meds"))
OUT = os.path.join(os.path.dirname(__file__), "..", "khurak", "data")

FORM_WORDS = {"tablet", "tablets", "tab", "capsule", "capsules", "cap", "syrup", "suspension", "injection",
              "cream", "gel", "ointment", "drop", "drops", "solution", "lotion", "inhaler", "rotacap",
              "rotacaps", "respules", "respule", "spray", "powder", "sachet", "granules", "soap",
              "shampoo", "infusion", "vial", "kit", "paste", "mouthwash", "liquid", "oral", "eye", "ear",
              "nasal", "dusting", "transcap", "transcaps", "softgel", "softgels", "lozenges", "jelly",
              "elixir", "emulsion", "serum", "face", "wash", "pessary", "suppository", "patch", "film",
              "strip", "pfs", "prefilled", "syringe", "redicaps", "spansule", "spansules", "dry", "sugar",
              "free", "sf", "expectorant"}


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def brand_key(name):
    w = re.split(r"[\s/]+", name.strip())[0]
    return re.sub(r"[^\w\-]+$", "", w.lower()).strip("-")


def variant_key(name):
    toks = [t for t in re.split(r"\s+", norm(name)) if t]
    return " ".join(t for t in toks if t not in FORM_WORDS)


def main():
    az = pd.read_csv(os.path.join(RAW, "updated_indian_medicine_data.csv"), low_memory=False)
    mg = pd.read_parquet(os.path.join(RAW, "dmedhi.parquet"))
    full = {norm(n): c for n, c in zip(mg.name, mg.composition) if isinstance(c, str)}
    for n, c in zip(az.name, az.salt_composition):
        if isinstance(c, str) and c.strip():
            full.setdefault(norm(n), c)

    rows, stats = [], {"total": 0, "repaired": 0, "two_listed_unrepaired": 0, "single": 0}
    for r in az.itertuples(index=False):
        name = str(r.name).strip()
        listed = " + ".join(str(x) for x in (r.short_composition1, r.short_composition2) if isinstance(x, str))
        comp_src, complete = listed, True
        if norm(name) in full:
            comp_src, source = full[norm(name)], "full"
            if len(split_composition(full[norm(name)])) > len(split_composition(listed)):
                stats["repaired"] += 1
        else:
            source = "az"
        parts = split_composition(comp_src)
        if not parts:
            continue
        if source == "az" and len(parts) == 2:
            complete = False            # A-Z keeps only two ingredients: a third may be missing
            stats["two_listed_unrepaired"] += 1
        if len(parts) == 1:
            stats["single"] += 1
        # a number in the product name that matches no listed strength hints at a missing ingredient
        nums = set(re.findall(r"(?<![\d.])(\d{2,4})(?![\d.])", name))
        listed_nums = {str(int(p["mg"])) for p in parts if p["mg"]} | set(re.findall(r"\d+", comp_src))
        hint = bool(nums - listed_nums) and not complete
        stats["total"] += 1
        rows.append([
            str(r.id), name, brand_key(name), variant_key(name), str(r.manufacturer_name or ""),
            "" if pd.isna(r.price) else f"{float(r.price):.2f}", "1" if r.Is_discontinued else "0",
            str(r.pack_size_label or ""),
            "|".join(p["name"] for p in parts), "|".join(p["strength"] for p in parts),
            "|".join("" if p["mg"] is None else f"{p['mg']:g}" for p in parts),
            "1" if complete else "0", "1" if hint else "0", source,
        ])

    cols = ["id", "name", "brand", "variant", "manufacturer", "price", "discontinued", "pack",
            "ingredients", "strengths", "mg", "complete", "missing_hint", "source"]
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, "products.tsv.gz"), "wt", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(x.replace("\t", " ").replace("\n", " ") for x in row) + "\n")
    stats["unique_brands"] = len({r[2] for r in rows})
    stats["unique_ingredients"] = len({i for r in rows for i in r[8].split("|")})
    stats["sources"] = {
        "az": "junioralive/Indian-Medicine-Dataset (MIT) - A-Z Medicine Dataset of India, 253,973 products",
        "full": "dmedhi/indian-medicines (1mg, 11,825 products) + A-Z salt_composition column",
    }
    json.dump(stats, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
