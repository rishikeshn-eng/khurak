// Parity: run docs/khurak.js (with onnxruntime-node) over JSON-lines texts from stdin.
import fs from "fs";
import * as ort from "onnxruntime-node";
globalThis.ort = ort;
await import("../../docs/khurak.js");
const K = globalThis.Khurak;
const D = new URL("../../docs/data/", import.meta.url);
const meta = JSON.parse(fs.readFileSync(new URL("meta.json", D)));
const tsv = fs.readFileSync(new URL("products.tsv", D), "utf8");
const sess = await ort.InferenceSession.create(new URL("context.onnx", D).pathname);
K.load(meta, tsv, sess);
for (const line of fs.readFileSync(0, "utf8").split("\n").filter(Boolean)) {
  const r = await K.check(JSON.parse(line));
  console.log(JSON.stringify({
    items: r.items.map((i) => [i.label, i.product ? i.product.name : null, [...i.ingredients.keys()].sort()]),
    findings: r.findings.map((f) => [f.severity, f.kind, f.message]), verdict: r.verdict }));
}
