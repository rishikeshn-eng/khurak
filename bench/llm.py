"""Resumable LLM runner: local models via MLX (batched, greedy) and Gemini via API.

  python bench/llm.py PROMPTS.jsonl OUT_DIR model [model ...]

PROMPTS.jsonl rows: {"id", "prompt", "max_tokens"}  (optional "system")
Writes OUT_DIR/<model-slug>.jsonl with {"id", "output"}; already-answered ids are skipped,
so a run can be killed and resumed. Decoding is greedy (temperature 0) everywhere, and
Qwen3's thinking mode is switched off so every model answers directly.
"""
import json
import os
import sys
import time

SLUG = lambda m: m.split("/")[-1].replace("-4bit", "").lower()


def load_prompts(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}


def run_mlx(model_id, todo, out, batch=12):
    from mlx_lm import load, batch_generate
    model, tok = load(model_id)

    def toks(p):
        user = (p["system"] + "\n\n" + p["prompt"]) if p.get("system") else p["prompt"]
        return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                       enable_thinking=False)

    # group by max_tokens so a batch shares one budget
    todo = sorted(todo, key=lambda p: (p["max_tokens"], len(p["prompt"])))
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            r = batch_generate(model, tok, [toks(p) for p in chunk], max_tokens=chunk[-1]["max_tokens"])
            for p, text in zip(chunk, r.texts):
                f.write(json.dumps({"id": p["id"], "output": text}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // batch) % 10 == 0:
                print(f"  {SLUG(model_id)} {i + len(chunk)}/{len(todo)}", flush=True)


def run_gemini(model_id, todo, out, rpm=12):
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    gap = 60.0 / rpm
    with open(out, "a", encoding="utf-8") as f:
        for n, p in enumerate(todo):
            for attempt in range(6):
                try:
                    t0 = time.time()
                    r = client.models.generate_content(
                        model=model_id, contents=p["prompt"],
                        config={"temperature": 0, "max_output_tokens": max(1024, p["max_tokens"] * 4),
                                "system_instruction": p.get("system")})
                    text = r.text or ""
                    break
                except Exception as e:
                    msg = str(e)
                    if "PerDay" in msg or "per day" in msg.lower():
                        print("  daily quota exhausted for", model_id, flush=True)
                        return
                    time.sleep(min(90, 10 * (attempt + 1)))
            else:
                continue
            f.write(json.dumps({"id": p["id"], "output": text}, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(max(0.0, gap - (time.time() - t0)))
            if n % 25 == 0:
                print(f"  {model_id} {n}/{len(todo)}", flush=True)


def main():
    prompts, out_dir, models = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    allp = load_prompts(prompts)
    for m in models:
        out = os.path.join(out_dir, SLUG(m) + ".jsonl")
        todo = [p for p in allp if p["id"] not in done_ids(out)]
        print(f"[{m}] {len(todo)} to run", flush=True)
        if not todo:
            continue
        (run_gemini if m.startswith("gemini") else run_mlx)(m, todo, out)
    print("RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
