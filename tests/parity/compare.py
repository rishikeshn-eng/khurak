"""Python <-> JavaScript parity for the khurak guard.

Runs the same texts through khurak.check (Python, onnxruntime) and docs/khurak.js
(node, onnxruntime-node) and compares medicines found, resolved products,
ingredients, findings and verdict.   python tests/parity/compare.py
"""
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from khurak.check import check  # noqa: E402

EXTRA = [
    "I took Dolo 650 this morning for fever. Can I also take Sinarest New tablet for my cold?",
    "Rx: Tab. Dolo 650 1-1-1-1 x 3 days\nTab. Sinarest New 1-1-1 after food\nCap. Pan 40 OD before breakfast",
    "मैंने बुखार के लिए डोलो 650 ली थी, क्या अब सिर दर्द के लिए क्रोसिन ले सकता हूँ?",
    "Bhaiya ek strip combiflem aur ek dollo650 bhej do, back pain hai",
    "Taking Brufen 400 twice a day and Voveran 50 for my back pain, is that fine?",
    "Pan 40 before breakfast and Allegra 120 at night for my allergy.",
    "I have a bad cold and need some rest, my sugar is fine and life is busy. Should I take Crocin?",
    "Can I take Vicks Action 500 Advanced tablet with paracetamol?",
    "Ultracet and Nexito together? also Alprax at night",
]


def main():
    meta = json.load(open(os.path.join(HERE, "..", "..", "bench", "tasks_meta.json")))
    prompts = [json.loads(l) for l in open(os.path.join(HERE, "..", "..", "bench", "prompts.jsonl"))]
    eka = pd.read_parquet(os.path.expanduser("~/Downloads/claude/rawdata2/meds/eka_mcqa.parquet")).question.tolist()
    texts = EXTRA + [p["prompt"] for p in prompts if meta[p["id"]]["task"] == "dup"] + eka
    js = subprocess.run(["node", os.path.join(HERE, "run.mjs")], input="\n".join(json.dumps(t, ensure_ascii=False) for t in texts),
                        capture_output=True, text=True)
    if js.returncode:
        print(js.stderr[-3000:])
        sys.exit(1)
    jl = js.stdout.strip().split("\n")
    bad = 0
    for t, j in zip(texts, jl):
        r = check(t)
        py = {"items": [[i["label"], i["product"]["name"] if i["product"] else None, sorted(i["ingredients"])] for i in r["items"]],
              "findings": [[f["severity"], f["kind"], f["message"]] for f in r["findings"]], "verdict": r["verdict"]}
        jj = json.loads(j)
        if py != jj:
            bad += 1
            if bad <= 6:
                print("DIFF:", t[:90], "\n  py:", json.dumps(py)[:400], "\n  js:", json.dumps(jj)[:400])
    print(f"parity: {len(texts) - bad}/{len(texts)} identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
