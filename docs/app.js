/* khurak site: in-browser guard + results from data/report.json */
(async function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (x, d = 0) => (x == null ? "–" : (x * 100).toFixed(d) + "%");

  const R = await fetch("data/report.json").then((r) => r.json()).catch(() => null);
  if (R) renderResults(R);

  // ---------------------------------------------------------------- guard
  const status = $("#status");
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    const t0 = performance.now();
    const [meta, tsv, sess] = await Promise.all([
      fetch("data/meta.json").then((r) => r.json()),
      fetch("data/products.tsv").then((r) => r.text()),
      ort.InferenceSession.create("data/context.onnx", { executionProviders: ["wasm"] }),
    ]);
    Khurak.load(meta, tsv, sess);
    status.textContent = `ready · ${Khurak.size.toLocaleString()} Indian medicines indexed in ${((performance.now() - t0) / 1000).toFixed(1)}s · nothing leaves your browser`;
  } catch (e) {
    status.textContent = "could not load the guard: " + e.message;
    return;
  }
  const EX = [
    ["Fever + cold", "I took Dolo 650 this morning for fever. Can I also take Sinarest New tablet for my cold?"],
    ["Prescription", "Rx: Tab. Dolo 650 1-1-1-1 x 3 days\nTab. Sinarest New 1-1-1 after food\nCap. Pan 40 OD before breakfast"],
    ["Hindi", "मैंने बुखार के लिए डोलो 650 ली थी, क्या अब सिर दर्द के लिए क्रोसिन ले सकता हूँ?"],
    ["Hinglish order", "Bhaiya ek strip combiflem aur ek dollo650 bhej do, back pain hai"],
    ["Two painkillers", "Taking Brufen 400 twice a day and Voveran 50 for my back pain, is that fine?"],
    ["Safe", "Pan 40 before breakfast and Allegra 120 at night for my allergy."],
    ["Word traps", "I have a bad cold and need some rest, my sugar is fine and life is busy. Should I take Crocin?"],
  ];
  EX.forEach(([l, t]) => { const b = document.createElement("button"); b.textContent = l; b.onclick = () => { $("#input").value = t; run(); }; $("#examples").appendChild(b); });
  let seq = 0;
  async function run() {
    const my = ++seq, text = $("#input").value;
    const r = await Khurak.check(text);
    if (my !== seq) return;
    const marks = [];
    r.items.forEach((it) => marks.push({ s: it.span[0], e: it.span[1], cls: it.product ? "m" : "m gen", tip: it.product ? it.product.name : "generic: " + [...it.ingredients.keys()].join(", ") }));
    if ($("#showRejected").checked)
      r.rejected.forEach((x) => marks.push({ s: x.start, e: x.end, cls: "rej", tip: `A dictionary would call this “${x.example}”. Context model: P(medicine) = ${x.confidence.toFixed(2)}` }));
    marks.sort((a, b) => a.s - b.s);
    let h = "", pos = 0;
    for (const m of marks) { if (m.s < pos) continue; h += esc(text.slice(pos, m.s)) + `<span class="${m.cls}" title="${esc(m.tip)}">${esc(text.slice(m.s, m.e))}</span>`; pos = m.e; }
    $("#annot").innerHTML = h + esc(text.slice(pos));
    const V = { danger: "Dangerous combination found", caution: "Worth checking with a pharmacist", ok: "No problems found" };
    $("#verdict").className = "verdict " + r.verdict;
    $("#verdict").textContent = r.items.length ? V[r.verdict] : "No medicines found";
    $("#findings").innerHTML = r.findings.filter((f) => f.severity !== "info" || f.kind === "incomplete_data")
      .map((f) => `<div class="f"><div class="sev ${f.severity}">${f.severity}</div><div>${esc(f.message)}</div></div>`).join("");
    $("#meds").innerHTML = r.items.filter((i) => i.product).map((it) => {
      const p = it.product;
      return `<div class="med ${p.complete ? "" : "warn"}"><div class="n">${esc(p.name)}</div>
        <div class="c">${p.ingredients.map((x, k) => esc(x + " " + p.strengths[k])).join(" + ")}</div>
        <div class="mf">${esc(p.manufacturer)} · matched “${esc(it.label)}” (${it.match}${it.per_day ? `, ${it.per_day}×/day` : ""})</div></div>`;
    }).join("");
  }
  $("#input").addEventListener("input", run);
  $("#showRejected").addEventListener("change", run);
  $("#input").value = EX[0][1];
  run();
  $("#lk").addEventListener("input", (e) => {
    const fam = Khurak.family(e.target.value.trim().split(/\s+/)[0] || "");
    $("#lkOut").innerHTML = fam.slice(0, 24).map((p) => `<div class="med ${p.complete ? "" : "warn"}"><div class="n">${esc(p.name)}</div>
      <div class="c">${p.ingredients.map((x, k) => esc(x + " " + p.strengths[k])).join(" + ")}</div>
      <div class="mf">${esc(p.manufacturer)}${p.discontinued ? " · discontinued" : ""}${p.complete ? "" : " · composition may be incomplete"}</div></div>`).join("")
      + (fam.length > 24 ? `<div class="note">…and ${fam.length - 24} more</div>` : "");
  });

  // ---------------------------------------------------------------- results
  function renderResults(R) {
    $("#slips").innerHTML = R.slips.map((s) => `<div class="slip"><div class="who">${esc(s.model)}</div><div class="q">What's in ${esc(s.brand)}?</div>
      <div class="a">“${esc(s.answer)}”</div><div class="t">Actually: ${esc(s.truth)}</div></div>`).join("");
    $("#kpis").innerHTML = R.kpis.map((k) => `<div class="kpi"><b>${k.v}</b><span>${k.l}</span></div>`).join("");
    $("#benchNote").innerHTML = R.bench_note;
    const ms = R.models;
    const card = (title, sub, key, guardVal, higherBetter = true) => {
      const rows = ms.filter((m) => m[key] != null).map((m) => ({ l: m.label, v: m[key] }));
      if (guardVal != null) rows.push({ l: "khurak guard", v: guardVal, us: true });
      rows.sort((a, b) => (higherBetter ? b.v - a.v : a.v - b.v));
      return `<div class="card"><div class="t">${title}</div><div class="s">${sub}</div>${rows.map((r) =>
        `<div class="hb ${r.us ? "us" : ""}"><span>${esc(r.l)}</span><div class="tr"><span style="width:${r.v * 100}%"></span></div><span class="v">${pct(r.v)}</span></div>`).join("")}</div>`;
    };
    $("#benchCards").innerHTML =
      card("Eka Care Indian Drug MCQA", `the ${R.guard.eka_covered} questions whose brand is in the open data · accuracy`, "mcqa_covered", R.guard.eka_acc) +
      card("Name the ingredients", "300 real brands · share of ingredients recalled", "compose_recall", null) +
      card("Spot the double dose", "200 risky pairs · warns about the shared ingredient", "dup_caught", R.guard.dup_caught);
    $("#benchTable").innerHTML = `<table><tr><th>Model</th><th>MCQA (all 1,512)</th><th>MCQA (covered)</th><th>Ingredients recalled</th><th>Answers with an invented ingredient</th><th>Double dose caught</th><th>…with composition given</th></tr>` +
      ms.map((m) => `<tr><td>${esc(m.label)}</td><td class="n">${pct(m.mcqa_acc)}</td><td class="n">${pct(m.mcqa_covered)}</td><td class="n">${pct(m.compose_recall)}</td><td class="n">${pct(m.compose_hallucinated)}</td><td class="n">${pct(m.dup_caught)}</td><td class="n">${pct(m.dup_caught_grounded)}</td></tr>`).join("") +
      `<tr class="us"><td>khurak guard</td><td class="n">–</td><td class="n">${pct(R.guard.eka_acc)}</td><td class="n">–</td><td class="n">0%</td><td class="n">${pct(R.guard.dup_caught)}</td><td class="n">–</td></tr></table>`;
    if (R.famous) {
      const F = R.famous, C = { correct: "var(--green)", partial: "var(--amber)", wrong: "var(--red)", unknown: "#cfcabd" };
      const L = { correct: "✓", partial: "½", wrong: "✕", unknown: "?" };
      $("#famous").innerHTML = `<table class="fam"><tr><th>Brand · what it really contains</th>${F.models.map((m) => `<th>${esc(m)}</th>`).join("")}</tr>` +
        F.rows.map((r) => `<tr><td><b>${esc(r.brand)}</b><div class="tr2">${esc(r.truth)}</div></td>${r.cells.map((c) =>
          `<td class="c" title="${esc(c.a)}" style="color:${C[c.s]}">${L[c.s]}</td>`).join("")}</tr>`).join("") + "</table>";
    }
    $("#quotes").innerHTML = (R.dup_quotes || []).map((q) => `<div class="quote"><div class="w">${esc(q.model)} · ${esc(q.q)}</div><blockquote>“${esc(q.a)}”</blockquote><div class="note">Both contain <b>${esc(q.shared)}</b>.</div></div>`).join("");
    // truncation viz: stacked bar
    const T = R.trunc;
    $("#truncViz").innerHTML = `<div class="card"><div class="t">253,973 products in the open dataset</div>
      <div class="s">by how many ingredients are listed</div>
      ${[["one ingredient", T.single, "var(--green)"], ["two listed, repaired from a full source", T.repaired, "var(--blue)"],
         ["two listed, cannot verify", T.unverified, "var(--amber)"]].map(([l, v, c]) =>
        `<div class="hb"><span>${l}</span><div class="tr"><span style="width:${v / T.total * 100}%;background:${c}"></span></div><span class="v">${(v / 1000).toFixed(0)}k</span></div>`).join("")}
      <p class="note" style="margin-bottom:0">Where full compositions exist, <b>${pct(T.truncated_share)}</b> of two-ingredient listings turn out to have three or more:
      an estimated <b>${Math.round(T.unverified * T.truncated_share / 1000)}k</b> products with a silently missing ingredient.</p></div>`;
    const C = R.ctx;
    $("#collide").textContent = C.collide;
    $("#ctxViz").innerHTML = `<div class="card"><div class="t">Finding brand mentions in real messages</div><div class="s">${esc(C.sub)}</div>
      ${C.rows.map((r) => `<div class="hb ${r.us ? "us" : ""}"><span>${esc(r.l)}</span><div class="tr"><span style="width:${r.v * 100}%"></span></div><span class="v">${r.fmt}</span></div>`).join("")}</div>`;
  }
})();
