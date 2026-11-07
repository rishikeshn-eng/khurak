"""khurak command line.

  khurak "I took Dolo 650 1-1-1-1, can I also take Sinarest New twice a day?"
  khurak --lookup "Combiflam"          # what is in a brand (all variants)
  khurak --json "..."                  # machine-readable report
  echo "$CHATBOT_REPLY" | khurak -     # check a chatbot's answer before showing it

Exit code: 0 ok, 1 caution, 2 danger.
"""
import argparse
import json
import sys

RED, YEL, GRN, DIM, B, R = "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="khurak", description="Check Indian medicine mentions for dangerous combinations.")
    ap.add_argument("text", nargs="?", help="text to check, or - for stdin")
    ap.add_argument("--lookup", help="show what a brand contains")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    color = sys.stdout.isatty() and not a.json
    c = (lambda code, s: f"{code}{s}{R}") if color else (lambda code, s: s)

    if a.lookup:
        from .find import index
        key = a.lookup.split()[0].lower()
        fam = index().db.family(key)
        if a.json:
            print(json.dumps([p.as_dict() for p in fam], indent=1, ensure_ascii=False))
            return 0
        if not fam:
            print(f"No Indian brand '{a.lookup}' in the database.")
            return 1
        for p in fam[:40]:
            flag = " (discontinued)" if p.discontinued else ""
            warn = c(YEL, "  ⚠ composition may be incomplete") if not p.complete else ""
            print(f"{c(B, p.name)}{c(DIM, flag)}\n   {p.composition()}  {c(DIM, '· ' + p.manufacturer)}{warn}")
        if len(fam) > 40:
            print(c(DIM, f"... {len(fam) - 40} more"))
        return 0

    text = sys.stdin.read() if a.text in (None, "-") else a.text
    from .check import check
    r = check(text)
    if a.json:
        print(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    else:
        print(c(B, "Medicines found"))
        for it in r["items"]:
            p = it["product"]
            what = f"{p['name']}: " + " + ".join(f"{i} {s}" for i, s in zip(p["ingredients"], p["strengths"])) if p \
                else f"(generic) {', '.join(it['ingredients'])}"
            print(f"  • {c(B, it['label'])} → {what}" + (c(DIM, f"  [{it['per_day']}×/day]") if it["per_day"] else ""))
        if not r["items"]:
            print(c(DIM, "  none"))
        if r["findings"]:
            print(c(B, "\nFindings"))
        col = {"high": RED, "medium": YEL, "low": DIM, "info": DIM}
        for f in r["findings"]:
            print(f"  {c(col[f['severity']], f['severity'].upper()):>8}  {f['message']}")
        v = {"danger": c(RED, "DANGER"), "caution": c(YEL, "CAUTION"), "ok": c(GRN, "OK")}[r["verdict"]]
        print(f"\nVerdict: {v}   {c(DIM, 'Not medical advice: confirm with a pharmacist or doctor.')}")
    return {"ok": 0, "caution": 1, "danger": 2}[r["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
