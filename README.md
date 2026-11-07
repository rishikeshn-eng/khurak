# khurak · खुराक

**Ask an AI what's in Dolo 650. One answered “Guggul Hilum”.**

[**Live guard →**](https://rishikeshn-eng.github.io/khurak/) Paste a question, a prescription or a chatbot's reply. 254k Indian medicines and the trained model run in your browser; nothing is uploaded.

---

Indians buy medicine by **brand**: Dolo, Crocin, Calpol, Combiflam, Sinarest, Sumo. Most of them contain the same drug, paracetamol,
and taking two at once is one of the commonest routes to an accidental overdose. A health chatbot that doesn't know Indian brands
can't notice. Real answers from the benchmark (all logged in `bench/out/famous/`):

| Model | “What's in…?” | Model's answer | Truth |
|---|---|---|---|
| Gemma 3 1B | Dolo 650 | Dolo 650 Tablet contains: Guggul Hilum + 100mg Sarkarpaka + 25mg Pulsatilla + 30mg Shatavari + 10mg | paracetamol 650mg |
| Llama 3.2 3B | Combiflam | Ibuprofen 400mg + Paracetamol 500mg + Pantothenic acid 10mg + Vitamin B1 10mg | ibuprofen 400mg + paracetamol 325mg |
| Qwen3 8B | Calpol 500mg | Paracetamol 500mg + Caffeine 60mg | paracetamol 500mg |
| Qwen3 1.7B | Azithral 500 | Azithral 500 Tablet contains Azithromycin + Metronidazole. | azithromycin 500mg |

khurak (1) **measures** how well language models know Indian medicines on three tasks, and (2) ships a **guard** that finds medicine
mentions in free text (English, Hinglish, Devanagari, typos), resolves them against 253,973 products, and flags dangerous combinations.

```python
>>> from khurak import check
>>> r = check("Took Dolo 650 in the morning. Can I take Sinarest New tablet for my cold?")
>>> r["verdict"], r["findings"][0]["message"]
('danger', 'Dolo, Sinarest all contain paracetamol. Taking them together doubles the dose.')
```

## Results

### Language models vs. the guard

| | Eka MCQA (all 1,512) | Eka MCQA (324 covered) | ingredients recalled | answers with an invented ingredient | **double dose caught** | …with compositions given |
|---|---|---|---|---|---|---|
| Gemma 3 1B | 34% | 31% | 2% | 90% | 1% | 2% |
| Qwen3 1.7B | 32% | 34% | 1% | 8% | 1% | 6% |
| Llama 3.2 3B | 44% | 45% | 1% | 1% | 9% | 43% |
| Phi-4 mini 3.8B | 34% | 35% | 4% | 33% | 18% | 42% |
| Qwen3 8B | 52% | 57% | 6% | 16% | 40% | 81% |
| Gemini 3.5 Flash-Lite | 63% | 60% | – | – | – | – |
| **khurak guard** | – | **66%** | – | 0% | **96%** | – |

- **Double dose** = 200 questions where a user has taken product A and asks about product B, both containing the same active
  ingredient (mostly paracetamol), plus 60 control pairs. A pass requires naming the shared ingredient *and* flagging the duplication.
  The guard raised **0 false alarms** on the 60 controls. Even the best local model (Qwen3 8B) catches only 40%; handing it both
  compositions lifts that to 81%, which is the case for giving chatbots khurak as a tool.
- **Composition**: asked what 300 real brands contain, models mostly decline, or invent ingredients (Gemma 3 1B lists Ayurvedic herbs
  for allopathic tablets). Gemini 3.5 Flash-Lite answered MCQA only (free-tier quota).
- **Eka Care's quiz** exposes a data problem: only **21%** of its questions name a brand that exists in the
  largest open dataset, and on those the open data agrees with Eka's answer 66% of the time.

### Household brands (13 brands × 5 models, `bench/prompts_famous.jsonl`)

Real outputs: Gemma 3 1B says Dolo 650 contains *“Guggul Hilum + Sarkarpaka + Pulsatilla + Shatavari”* (it's paracetamol 650 mg);
Llama 3.2 3B adds *“Pantothenic acid + Vitamin B1”* to Combiflam; Qwen3 8B adds caffeine to Calpol and **codeine to Dolo 650**.

### The context model

| | precision | recall | F1 |
|---|---|---|---|
| dictionary only | 3.2% | 100% | 6.1% |
| **khurak** (unseen templates + unseen brands) | **96.7%** | **95.2%** | **96.0%** |

False-alarm rate on real Wikipedia text: 0.015% (English), 0.007% (Hindi).
Two transparent rules sit on top: a brand followed by one of its real strengths/forms/product names (“Dolo 650”, “Liv Joint Tablet”)
is accepted, and once a message certainly mentions a medicine, other well-known brands need less evidence.

### Brand names are reused for unrelated drugs

**8,453 brand names (6%)** in the open data cover products with no ingredient in common, e.g. *Mac* 4 mg is methylprednisolone
but *Mac* 20 mg is rabeprazole; *Bitol* 800 is ethambutol (a TB drug) while *Bitol* 10 is an inositol supplement. khurak resolves
by strength and form, and when the stated product doesn't exist it **refuses to guess** and says the name is ambiguous.

The browser guard (`docs/khurak.js`) is a port of the Python and gives identical output on 1,781 test messages (`tests/parity`).

## What the guard checks

1. **Finds candidates**: exact brand keys, typos (edit distance 1), phonetic and Devanagari spellings
   (`कॉम्बिफ्लेम` → Combiflam, `डोलो` → Dolo), and strengths glued to names (`dolo650`).
2. **Reads the context.** 21% of ordinary English word tokens are also some Indian brand (“cold”, “rest”, “pan”, “life”, even
   “and”, an ondansetron injection). A small trained model (char-CNN + BiGRU, ~100k parameters, 2 MB ONNX) looks at ±10 surrounding
   words and decides whether a match is really a medicine.
3. **Resolves the variant** from the words around it: strength (“650”), suffix (“LC”, “SP”, “New”), dosage form (“tab”, “syrup”, “spray”).
4. **Checks safety:**
   - the same active ingredient in two products (salt-aware: diclofenac sodium = diclofenac potassium)
   - two drugs of the same class (two NSAIDs, two PPIs, opioid + opioid…)
   - a short list of well-established dangerous combinations (NSAID + anticoagulant, opioid + benzodiazepine, tramadol + SSRI, PDE5 + nitrate)
   - daily totals above approximate adult maxima when strength and frequency (`1-0-1`, `TDS`, `every 6 hours`) are stated
   - India's 2011 ban on nimesulide for children under 12
   - discontinued products, and **incomplete source data** (below)

## The data has a hole in it

The largest open list of Indian medicines (the A–Z Medicine Dataset of India, 253,973 products, MIT) stores only the **first two
ingredients** of each product. Where full compositions are available (1mg subset), **25% of two-ingredient listings are actually three
or more**: Ascoril LS loses its guaifenesin, cold syrups lose their dextromethorphan, some cold tablets lose their paracetamol. A
retrieval-augmented chatbot built on this file will miss exactly the duplicates that matter. khurak repairs 1,628 products from full
sources and **flags** the remaining 105,954 two-ingredient listings as possibly incomplete instead of trusting them.

## Install & use

```bash
pip install -e .
khurak "Rx: Tab. Dolo 650 1-1-1-1, Tab. Sinarest New TDS"
khurak --lookup Combiflam
echo "$CHATBOT_REPLY" | khurak -        # exit code 0 ok · 1 caution · 2 danger
```

Put it in front of any health chatbot as an output filter, or give the model khurak's lookup as a tool. The grounded runs in the
benchmark show what happens when a model is given the compositions.

## Reproduce

```
pipeline/build_db.py         merge A–Z (254k) + 1mg full compositions → khurak/data/products.tsv.gz
pipeline/gen_templates.py    753 realistic messages from Gemini with [MED] slots (English, Hinglish, Hindi, prescriptions)
pipeline/make_ctx_data.py    579k training / 200k test windows (unseen templates AND unseen brands; Wikipedia negatives)
pipeline/train_ctx.py        trains + exports the context model
bench/make_tasks.py          Eka MCQA (1,512) + composition (300) + double-dose safety (260) + grounded variant
bench/llm.py · score.py      run and score 7 local models (MLX) + Gemini
bench/eval_guard.py          the guard on the same questions
tests/                       unit tests + Python↔JavaScript parity
```

## Limitations

- **Not medical advice.** khurak flags combinations for a pharmacist or doctor to review. Dose limits are approximate adult maxima;
  children, pregnancy, liver and kidney disease change them.
- The interaction list is deliberately short and conservative. Absence of a finding is not a guarantee of safety.
- Compositions come from open datasets that can be outdated, and a quarter of two-ingredient listings may be missing an ingredient
  (flagged, not fixed).
- The context model was trained on templated messages with real brand names inserted. Real users will produce phrasings it hasn't seen.
- Gemini was evaluated on a subset because of the free API tier's daily request limit.

## Data & credits

A–Z Medicine Dataset of India ([junioralive/Indian-Medicine-Dataset](https://github.com/junioralive/Indian-Medicine-Dataset), MIT) ·
1mg subset ([dmedhi/indian-medicines](https://huggingface.co/datasets/dmedhi/indian-medicines)) ·
[Eka Care Indian Drug MCQA](https://huggingface.co/datasets/ekacare/indian_drug_mcqa) (MIT) · Wikipedia (CC BY-SA) for negative
contexts. MIT licensed.
