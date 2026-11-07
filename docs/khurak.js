/* khurak.js - browser port of khurak/text.py, find.py and check.py (parity-tested against Python). */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------ text
  const TOKEN = /[A-Za-z0-9][A-Za-z0-9\-']*|[ऀ-ॣ०-ॿ]+|\S/g;
  const DEVA_DIG = { "०": "0", "१": "1", "२": "2", "३": "3", "४": "4", "५": "5", "६": "6", "७": "7", "८": "8", "९": "9" };
  const CONS = { "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h", "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y" };
  const VOW = { "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऋ": "ri", "ऍ": "e" };
  const MATRA = { "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ृ": "ri", "ॅ": "e" };
  const VIRAMA = "्", NUKTA = "़";
  const isDeva = (s) => /[ऀ-ॿ]/.test(s);

  function romanize(word) {
    const w = [...word.replace(/[०-९]/g, (d) => DEVA_DIG[d])];
    const out = [];
    for (let i = 0; i < w.length; i++) {
      let ch = w[i];
      if (i + 1 < w.length && w[i + 1] === NUKTA && CONS[ch + NUKTA] !== undefined) { ch = ch + NUKTA; i++; }
      if (CONS[ch] !== undefined) {
        out.push(CONS[ch]);
        const nxt = i + 1 < w.length ? w[i + 1] : "";
        if (MATRA[nxt] !== undefined) { out.push(MATRA[nxt]); i++; }
        else if (nxt === VIRAMA) i++;
        else if (nxt && (CONS[nxt] !== undefined || "ंँ".includes(nxt))) out.push("a");
      } else if (VOW[ch] !== undefined) out.push(VOW[ch]);
      else if ("ंँ".includes(ch)) out.push("n");
      else if (ch === "ः") out.push("h");
      else if (ch.charCodeAt(0) < 128) out.push(ch);
    }
    return out.join("");
  }

  const PH = [["chh", "ch"], ["ph", "f"], ["ck", "k"], ["q", "k"], ["c", "k"], ["z", "j"], ["w", "v"], ["x", "ks"], ["y", "i"],
    ["th", "t"], ["dh", "d"], ["kh", "k"], ["gh", "g"], ["bh", "b"], ["sh", "s"], ["ai", "e"], ["ei", "e"], ["au", "o"], ["ou", "o"], ["oo", "u"]];
  function phonetic(word) {
    let w = word.toLowerCase();
    if (isDeva(w)) w = romanize(w);
    w = w.replace(/[^a-z0-9]/g, "").replace(/c(?=[eiy])/g, "s");
    for (const [a, b] of PH) w = w.split(a).join(b);
    return w.replace(/(.)\1+/g, "$1");
  }
  function tokenize(text) { return [...text.matchAll(TOKEN)].map((m) => [m[0], m.index, m.index + m[0].length]); }
  const pyLower = (s) => s.toLowerCase();
  function strip(s, chars) { let a = 0, b = s.length; while (a < b && chars.includes(s[a])) a++; while (b > a && chars.includes(s[b - 1])) b--; return s.slice(a, b); }

  // ------------------------------------------------------------------ data + index
  const FORM_WORDS = new Set(("tablet tablets tab capsule capsules cap syrup suspension injection cream gel ointment drop drops solution lotion " +
    "inhaler rotacap rotacaps respules respule spray powder sachet granules soap shampoo infusion vial kit paste mouthwash liquid oral eye ear " +
    "nasal dusting transcap transcaps softgel softgels lozenges jelly elixir emulsion serum face wash pessary suppository patch film strip pfs " +
    "prefilled syringe redicaps spansule spansules dry sugar free sf expectorant").split(" "));
  let M = null, P = [], BR = new Map(), PHON = new Map(), DEVA = new Map(), DEL = new Map(), GEN = null, MAXGW = 1, CHAR_ID = null, sess = null;
  let POP = new Set();

  const add = (m, k, v) => { let s = m.get(k); if (!s) m.set(k, (s = new Set())); s.add(v); };
  const deletes = (k) => { const o = []; for (let i = 0; i < k.length; i++) o.push(k.slice(0, i) + k.slice(i + 1)); return o; };

  function load(meta, tsv, session) {
    M = meta; sess = session;
    CHAR_ID = new Map(meta.chars.map((c, i) => [c, i + 2]));
    GEN = new Map(Object.entries(meta.generics));
    MAXGW = Math.max(...[...GEN.keys()].map((g) => g.split(" ").length));
    const genFirst = new Set([...GEN.keys()].map((g) => g.split(" ")[0]));
    P = [];
    for (const line of tsv.split("\n")) {
      if (!line) continue;
      const [name, brand, mfr, flags, ings, strs, mg] = line.split("\t");
      const p = { name, brand, manufacturer: meta.manufacturers[+mfr], discontinued: flags.includes("d"), complete: !flags.includes("i"),
        missing_hint: flags.includes("h"), source: flags.includes("f") ? "full" : "az",
        ingredients: ings.split(",").map((i) => meta.ingredients[+i]), strengths: strs.split("|"),
        mg: mg.split(",").map((x) => (x === "" ? null : parseFloat(x))) };
      p.variant = name.toLowerCase().replace(/\s+/g, " ").trim().split(" ").filter((t) => t && !FORM_WORDS.has(t)).join(" ");
      const idx = P.length;
      P.push(p);
      if (p.source === "full") { POP.add(brand); POP.add(brand.split("-")[0]); }
      for (const k of new Set([brand, brand.split("-")[0]]))
        if (k.length >= 3 && !genFirst.has(k) && !/^\d+$/.test(k)) { let a = BR.get(k); if (!a) BR.set(k, (a = [])); a.push(idx); }
    }
    POP = new Set([...POP].filter((k) => BR.has(k)));
    for (const k of BR.keys()) {
      if (k.length >= 5) { for (const d of deletes(k)) add(DEL, d, k); add(DEL, k, k); }
      if (k.length >= 4) {
        const ph = phonetic(k);
        add(PHON, ph, k);
        const k2 = ph.split("a").join("");
        if (k2.length >= 4) for (const v of [k2, ...deletes(k2)]) add(DEVA, v, k);
      }
    }
  }
  const famSize = (k) => BR.get(k).length;
  const byPop = (a, b) => famSize(b) - famSize(a) || (a < b ? -1 : a > b ? 1 : 0);

  function fuzzy(word) {
    if (word.length < 5) return [];
    const out = new Set(DEL.get(word) || []);
    for (const d of deletes(word)) for (const k of DEL.get(d) || []) out.add(k);
    return [...out].filter((k) => Math.abs(k.length - word.length) <= 1);
  }
  const GLUED = /^([a-z][a-z\-]{2,}?)(\d{2,4})$/;
  function edit1(a, b) {
    if (a === b) return true;
    if (Math.abs(a.length - b.length) > 1) return false;
    if (a.length === b.length) {
      const d = [];
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d.push(i);
      return d.length === 1 || (d.length === 2 && d[1] === d[0] + 1 && a[d[0]] === b[d[1]] && a[d[1]] === b[d[0]]);
    }
    if (a.length > b.length) [a, b] = [b, a];
    let i = 0;
    while (i < a.length && a[i] === b[i]) i++;
    return a.slice(i) === b.slice(i + 1);
  }
  const candFeats = (key) => [POP.has(key) ? 1 : 0, Math.min(1, Math.log1p(BR.get(key).length) / Math.log(200))];

  const NOT_BRAND = new Set(["tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps", "syrup", "syp", "injection",
    "inj", "suspension", "cream", "gel", "ointment", "drops", "drop", "softgel", "lotion", "inhaler", "sachet",
    "solution", "spray", "powder", "strip", "dose", "medicine"]);
  function keysFor(tok) {
    const low = strip(tok.toLowerCase(), ".'");
    if (NOT_BRAND.has(low)) return [];
    if (isDeva(low)) {
      const k2 = phonetic(low).split("a").join("");
      if (k2.length < 4) return [];
      const hits = new Set(DEVA.get(k2) || []);
      for (const d of deletes(k2)) for (const k of DEVA.get(d) || []) hits.add(k);
      const scored = [];
      for (const k of hits) { const kk = phonetic(k).split("a").join(""); if (edit1(k2, kk)) scored.push([kk !== k2 ? 1 : 0, -famSize(k), k]); }
      scored.sort((x, y) => x[0] - y[0] || x[1] - y[1] || (x[2] < y[2] ? -1 : x[2] > y[2] ? 1 : 0));
      return scored.slice(0, 3).map((x) => [x[2], "devanagari"]);
    }
    const out = [];
    if (BR.has(low)) out.push([low, "exact"]);
    const m = low.match(GLUED);
    if (m) {
      const base = m[1];
      if (BR.has(base)) out.push([base, "glued"]);
      else if (base.length >= 4) {
        const ph = [...(PHON.get(phonetic(base)) || [])].sort(byPop);
        const near = ph.length ? ph : fuzzy(base).filter((k) => edit1(base, k)).sort(byPop);
        near.slice(0, 1).forEach((k) => out.push([k, "glued"]));
      }
    }
    if (!out.length && low.length >= 4 && !/^\d+$/.test(low)) {
      for (const k of [...(PHON.get(phonetic(low)) || [])].sort(byPop)) out.push([k, "phonetic"]);
      if (!out.length) for (const k of fuzzy(low).filter((k) => edit1(low, k)).sort(byPop).slice(0, 3)) out.push([k, "typo"]);
    }
    return out;
  }

  // window encoding identical to khurak/text.py
  function encodeToken(tok) {
    const ids = [...tok.toLowerCase()].slice(0, M.max_chars).map((c) => CHAR_ID.get(c) || 1);
    while (ids.length < M.max_chars) ids.push(0);
    return ids;
  }
  function shape(t) {
    return [/^[A-Z]/.test(t) || (t[0] && t[0] !== t[0].toLowerCase()) ? 1 : 0, (t === t.toUpperCase() && t !== t.toLowerCase() && t.length > 1) ? 1 : 0,
      /\d/.test(t) ? 1 : 0, isDeva(t) ? 1 : 0, /^[\d.]+$/.test(t) ? 1 : 0];
  }
  function encodeWindow(tokens, ci, cand) {
    const L = M.window;
    const left = tokens.slice(Math.max(0, ci - L), ci), right = tokens.slice(ci + 1, ci + 1 + L);
    const slots = [...Array(L - left.length).fill(""), ...left, tokens[ci], ...right, ...Array(L - right.length).fill("")];
    const shapes = slots.map((t) => [...(t ? shape(t) : [0, 0, 0, 0, 0]), 0, 0]);
    shapes[L][5] = cand[0]; shapes[L][6] = cand[1];
    return { chars: slots.map((t) => (t ? encodeToken(t) : Array(M.max_chars).fill(0))), shapes };
  }
  async function score(windows) {
    if (!windows.length) return [];
    const B = windows.length, T = 2 * M.window + 1, C = M.max_chars;
    const NS = 7;
    const chars = new BigInt64Array(B * T * C), shapes = new Float32Array(B * T * NS), mark = new Float32Array(B * T);
    windows.forEach((w, b) => {
      w.chars.forEach((row, t) => row.forEach((v, c) => (chars[(b * T + t) * C + c] = BigInt(v))));
      w.shapes.forEach((row, t) => row.forEach((v, k) => (shapes[(b * T + t) * NS + k] = v)));
      mark[b * T + M.window] = 1;
    });
    const ort = root.ort;
    const out = await sess.run({ chars: new ort.Tensor("int64", chars, [B, T, C]), shapes: new ort.Tensor("float32", shapes, [B, T, NS]),
      mark: new ort.Tensor("float32", mark, [B, T]) });
    return Array.from(out.prob.data);
  }

  const FORM_STRICT = new Set(["tablet", "tablets", "capsule", "capsules", "syrup", "injection", "suspension", "cream", "gel",
    "ointment", "drops", "drop", "softgel", "lotion", "inhaler", "rotacap", "respules", "sachet"]);
  const FORMS = { tab: "tablet", tabs: "tablet", tablet: "tablet", tablets: "tablet", goli: "tablet", cap: "capsule", caps: "capsule",
    capsule: "capsule", syp: "syrup", syrup: "syrup", suspension: "suspension", susp: "suspension", drops: "drop", drop: "drop",
    spray: "spray", gel: "gel", cream: "cream", ointment: "ointment", inj: "injection", injection: "injection" };
  function resolve(key, follow, before) {
    const fam = (BR.get(key) || []).map((i) => P[i]);
    if (!fam.length) return [];
    follow = follow.slice(0, 3).map((f) => strip(f.toLowerCase(), ".,"));
    const ctx = [...follow, ...before.slice(-2).map((b) => strip(b.toLowerCase(), ".,"))];
    const forms = new Set(ctx.filter((t) => FORMS[t]).map((t) => FORMS[t]));
    const st = follow.slice(0, 2).find((f) => /^\d{1,4}(mg)?$/.test(f));
    if (st) {
      const stated = st.replace(/mg$/, "");
      if (!fam.some((p) => p.mg.some((m) => m && String(Math.trunc(m)) === stated))) {
        const live = fam.filter((p) => !p.discontinued);
        return live.length ? live : fam;
      }
    }
    const sc = (p) => {
      const v = new Set(p.variant.split(/[\s\-/]+/));
      const name = p.name.toLowerCase();
      let s = follow.filter((f) => f && v.has(f) && !FORMS[f]).length * 2;
      s += follow.filter((f) => /^\d+$/.test(f) && p.mg.some((m) => m && String(Math.trunc(m)) === f)).length;
      if (forms.size) s += [...forms].some((f) => name.includes(f)) ? 1.5 : 0;
      else if (/\b(tablet|capsule)s?\b/.test(name)) s += 0.5;
      return s - 0.01 * v.size;
    };
    const scores = fam.map(sc), best = Math.max(...scores);
    const chosen = fam.filter((_, i) => scores[i] === best);
    const live = chosen.filter((p) => !p.discontinued);
    return live.length ? live : chosen;
  }

  async function find(text, threshold) {
    threshold = threshold ?? M.threshold;
    const toks = tokenize(text), words = toks.map((t) => t[0]);
    const cands = [], windows = [];
    words.forEach((w, i) => {
      if (!/^[A-Za-zऀ-ॿ]/.test(w)) return;
      const keys = keysFor(w);
      if (keys.length) { cands.push([i, keys]); windows.push(encodeWindow(words, i, candFeats(keys[0][0]))); }
    });
    const probs = await score(windows);
    // Rule 1 - strength match ("Dolo 650", "Pan 40", "dolo650"); Rule 2 - medical-context propagation.
    cands.forEach(([i, keys], j) => {
      const m = words[i].toLowerCase().match(GLUED);
      const nxt = m ? m[2] : (i + 1 < words.length ? words[i + 1].replace(/mg$/i, "") : "");
      if (/^\d+$/.test(nxt) && (BR.get(keys[0][0]) || []).some((k) => P[k].mg.some((x) => x && String(Math.trunc(x)) === nxt)))
        probs[j] = Math.max(probs[j], 0.99);
      const nxt1 = i + 1 < words.length ? strip(words[i + 1].toLowerCase(), ".") : "";
      const before = i > 0 ? strip(words[i - 1].toLowerCase(), ".") : "";
      const w1 = i + 1 < words.length ? words[i + 1].toLowerCase() : "";
      const completes = /^[a-z][a-z0-9\-]+$/.test(w1) &&
        (BR.get(keys[0][0]) || []).slice(0, 400).some((k) => { const t = P[k].name.toLowerCase().split(/\s+/); return t.length >= 2 && t[1] === w1; });
      if (FORM_STRICT.has(nxt1) || ["tab", "cap", "syp", "inj", "tabs", "caps"].includes(before) || completes)
        probs[j] = Math.max(probs[j], 0.99);
    });
    if (probs.some((p) => p >= threshold))
      cands.forEach(([, keys], j) => { if (POP.has(keys[0][0]) && probs[j] >= 0.05) probs[j] = Math.max(probs[j], threshold); });
    const brands = [], rejected = [], consumed = new Set();
    cands.forEach(([i, keys], j) => {
      const p = probs[j];
      if (consumed.has(i)) return;
      if (p < threshold) {
        rejected.push({ text: words[i], start: toks[i][1], end: toks[i][2], key: keys[0][0], confidence: p,
          example: (BR.get(keys[0][0]) || []).slice(0, 1).map((k) => P[k].name)[0] });
        return;
      }
      const [key, how] = keys[0];
      let follow = words.slice(i + 1, i + 4);
      const m = words[i].toLowerCase().match(GLUED);
      if (m) follow = [m[2], ...follow];
      const prods = resolve(key, follow, words.slice(Math.max(0, i - 2), i));
      if (prods.length) {
        brands.push({ text: words[i], start: toks[i][1], end: toks[i][2], key, match: how, confidence: p, products: prods });
        const pt = prods[0].name.toLowerCase().split(/\s+/);
        for (let k = 1; k < Math.min(pt.length, 5); k++) {
          if (i + k < words.length && words[i + k].toLowerCase() === pt[k]) consumed.add(i + k); else break;
        }
      }
    });
    const generics = [], low = words.map(pyLower), used = new Set();
    for (let n = MAXGW; n >= 1; n--)
      for (let i = 0; i + n <= low.length; i++) {
        let busy = false;
        for (let j = i; j < i + n; j++) if (used.has(j)) busy = true;
        if (busy) continue;
        const phrase = low.slice(i, i + n).join(" ");
        if (GEN.has(phrase) && phrase.length >= 4) {
          generics.push({ text: words.slice(i, i + n).join(" "), start: toks[i][1], end: toks[i + n - 1][2], ingredient: GEN.get(phrase) });
          for (let j = i; j < i + n; j++) used.add(j);
        }
      }
    return { brands, generics, rejected, candidates: cands.length };
  }

  // ------------------------------------------------------------------ check
  const FREQ = [
    [/\b(\d)\s*-\s*(\d)\s*-\s*(\d)(?:\s*-\s*(\d))?\b/, null], [/\b(qid|four times|4 times)\b/i, 4], [/\b(tds|tid|thrice|three times|3 times)\b/i, 3],
    [/\b(bd|bid|twice|two times|2 times|subah[- ]shaam)\b/i, 2], [/\b(od|once|1 time|daily once|hs)\b/i, 1], [/\bevery\s+(4|6|8|12)\s*(?:hours|hrs|h)\b/i, "q"]];
  function dosesPerDay(s) {
    for (const [rx, n] of FREQ) {
      const m = s.match(rx);
      if (!m) continue;
      if (n === null) return m.slice(1).filter(Boolean).reduce((a, g) => a + +g, 0);
      if (n === "q") return Math.floor(24 / +m[1]);
      return n;
    }
    return null;
  }
  const g = (x) => String(parseFloat(x.toPrecision(6)));
  const cmpLower = (a, b) => { const x = a.toLowerCase(), y = b.toLowerCase(); return x < y ? -1 : x > y ? 1 : 0; };
  const classOf = (ing) => Object.entries(M.classes).filter(([, mem]) => mem.includes(ing)).map(([c]) => c);

  async function check(text) {
    const found = await find(text);
    const items = [];
    for (const b of found.brands) {
      const p = b.products[0];
      const common = b.products.map((x) => new Set(x.ingredients)).reduce((a, s) => new Set([...a].filter((v) => s.has(v))));
      const ings = new Map();
      p.ingredients.forEach((ing, k) => { if (common.has(ing)) ings.set(ing, p.mg[k]); });
      items.push({ label: b.text, span: [b.start, b.end], product: p, variants: b.products.length, ingredients: ings,
        per_day: dosesPerDay(text.slice(b.end, b.end + 60)), match: b.match, confidence: b.confidence });
    }
    for (const gn of found.generics)
      items.push({ label: gn.text, span: [gn.start, gn.end], product: null, variants: 0, ingredients: new Map([[gn.ingredient, null]]),
        per_day: null, match: "generic", confidence: 1 });
    const F = [];
    const SEV = new Set(M.severe_dup);
    const byIng = new Map();
    items.forEach((it, k) => { for (const ing of it.ingredients.keys()) { if (!byIng.has(ing)) byIng.set(ing, []); byIng.get(ing).push(k); } });
    for (const [ing, ks] of byIng) {
      const labels = [...new Set(ks.map((k) => items[k].label))].sort(cmpLower);
      if (ks.length >= 2 && labels.length >= 2)
        F.push({ severity: SEV.has(ing) ? "high" : "medium", kind: "duplicate_ingredient", ingredient: ing, items: labels,
          message: `${labels.join(", ")} all contain ${ing}. Taking them together doubles the dose.` });
    }
    for (const [cls, members] of Object.entries(M.classes)) {
      const present = new Map();
      for (const it of items) for (const ing of it.ingredients.keys()) if (members.includes(ing)) { if (!present.has(ing)) present.set(ing, new Set()); present.get(ing).add(it.label); }
      if (present.size >= 2 && !/^(anticoagulant|nitrate|PDE5)/.test(cls)) {
        const ings = [...present.keys()].sort();
        F.push({ severity: ["NSAID", "opioid", "benzodiazepine-type sedative"].includes(cls) ? "high" : "medium", kind: "same_class", class: cls, ingredients: ings,
          message: `Two different ${cls}s together (${ings.join(", ")}): same effect and side effects, stacked.` });
      }
    }
    const cp = new Map();
    for (const it of items) for (const ing of it.ingredients.keys()) for (const c of classOf(ing)) { if (!cp.has(c)) cp.set(c, new Set()); cp.get(c).add(`${ing} (${it.label})`); }
    for (const [a, b, sev, why] of M.interactions)
      if (cp.has(a) && cp.has(b))
        F.push({ severity: sev, kind: "interaction", classes: [a, b], message: `${[...cp.get(a)].sort().join("; ")} with ${[...cp.get(b)].sort().join("; ")}: ${why}.` });
    const totals = new Map(), parts = new Map();
    for (const it of items) {
      if (!it.per_day) continue;
      for (const [ing, mg] of it.ingredients)
        if (mg && M.max_daily[ing] !== undefined) {
          totals.set(ing, (totals.get(ing) || 0) + mg * it.per_day);
          if (!parts.has(ing)) parts.set(ing, []);
          parts.get(ing).push(`${it.label} ${g(mg)} mg x${it.per_day}`);
        }
    }
    for (const [ing, tot] of totals) {
      const lim = M.max_daily[ing], note = M.max_note[ing] || "";
      if (tot > lim)
        F.push({ severity: "high", kind: "daily_dose", ingredient: ing, total_mg: tot, limit_mg: lim,
          message: `About ${g(tot)} mg of ${ing} per day (${parts.get(ing).join(" + ")}) exceeds the usual adult maximum of ${lim} mg${note ? "; " + note : ""}.` });
    }
    for (const it of items) {
      const p = it.product;
      if (!p) continue;
      if (p.ingredients.includes("nimesulide") && /suspension|syrup|drops|paediatric|pediatric|kid/i.test(p.name))
        F.push({ severity: "high", kind: "banned_use", items: [it.label], message: `${p.name} is a liquid nimesulide product; India banned nimesulide for children under 12 in 2011.` });
      if (p.discontinued) F.push({ severity: "low", kind: "discontinued", items: [it.label], message: `${p.name} is listed as discontinued.` });
      if (!p.complete) {
        let msg = `${p.name}: the open dataset lists only two ingredients for this product; a third may be missing.`;
        if (p.missing_hint) msg += " Its name mentions a strength that matches no listed ingredient.";
        F.push({ severity: "info", kind: "incomplete_data", items: [it.label], message: msg });
      }
      if (it.variants > 1) {
        let msg = `'${it.label}' matches ${it.variants} products; only ingredients common to all of them were used.`, sev = "info";
        if (!it.ingredients.size) {
          msg = `'${it.label}' matches ${it.variants} products with no ingredient in common: this brand name is used for different drugs. Check the exact product and strength on the strip.`;
          sev = "medium";
        }
        F.push({ severity: sev, kind: "ambiguous", items: [it.label], message: msg });
      }
    }
    const order = { high: 0, medium: 1, low: 2, info: 3 };
    F.sort((a, b) => order[a.severity] - order[b.severity]);
    return { items, findings: F, candidates: found.candidates, rejected: found.rejected,
      verdict: F.some((f) => f.severity === "high") ? "danger" : F.some((f) => f.severity === "medium") ? "caution" : "ok" };
  }

  function family(key) { return (BR.get(key.toLowerCase()) || []).map((i) => P[i]); }
  root.Khurak = { load, find, check, family, tokenize, phonetic, romanize, keysFor, resolve, get size() { return P.length; } };
})(typeof window !== "undefined" ? window : globalThis);
