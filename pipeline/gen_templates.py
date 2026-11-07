"""Ask Gemini for realistic Indian medicine-related messages with placeholders.

Each template contains [MED1] (and optionally [MED2]) where a medicine BRAND goes.
Real brand surface forms are substituted later, so labels are exact by construction.
Writes pipeline/templates_gemini.jsonl.
"""
import json
import os
import re
import time

from google import genai

STYLES = [
    "a worried patient texting a family doctor on WhatsApp in English",
    "a patient writing in Hinglish (Hindi in Latin script) on WhatsApp",
    "a patient writing in Hindi in Devanagari script, with medicine names left in English letters",
    "a customer ordering medicines from a local chemist on WhatsApp (mixed English/Hinglish)",
    "a doctor's handwritten-style prescription note (Tab./Syp./Cap., dosing like 1-0-1, x 5 days)",
    "a health chatbot answering a user in English about a medicine",
    "a parent asking about medicine for a child (English or Hinglish)",
    "an elderly person's question typed by a grandchild, informal English",
    "a pharmacist explaining to a customer what a medicine is for",
    "a Reddit/Quora-style question from India about side effects",
    "a Tamil/Telugu/Bengali speaker writing in English with Indian English phrasing",
    "a discharge summary line listing medicines",
]
PROMPT = """Write 40 different short, realistic messages in the voice of {style}.
Each message must mention one or two medicine BRAND names, written ONLY as the placeholders [MED1] and [MED2].
Never write a real medicine name; you may mention generic salts (paracetamol, ibuprofen...) or symptoms.
Vary length (5 to 45 words), tone, spelling quality and context (dose, timing, side effects, price, availability).
Also mix in ordinary words that could look like names (cold, rest, pain, sugar, life, active, plus, fast, care).
Return JSON: a list of strings. No commentary."""


def main():
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out = os.path.join(os.path.dirname(__file__), "templates_gemini.jsonl")
    have = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    with open(out, "a", encoding="utf-8") as f:
        for rep in range(2):
            for i, s in enumerate(STYLES):
                if have >= 900:
                    return
                for attempt in range(4):
                    try:
                        r = c.models.generate_content(model="gemini-3.5-flash", contents=PROMPT.format(style=s),
                                                      config={"temperature": 1.0, "response_mime_type": "application/json"})
                        items = json.loads(r.text)
                        break
                    except Exception as e:
                        print("retry", e.__class__.__name__, str(e)[:120], flush=True)
                        time.sleep(20 * (attempt + 1))
                else:
                    continue
                n = 0
                for t in items:
                    if isinstance(t, str) and "[MED1]" in t:
                        f.write(json.dumps({"style": i, "text": t}, ensure_ascii=False) + "\n")
                        n += 1
                have += n
                print(f"style {i} rep {rep}: +{n} (total {have})", flush=True)
                time.sleep(5)


if __name__ == "__main__":
    main()
