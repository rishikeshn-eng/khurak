"""Train khurak's context model and export it to ONNX (khurak/data/context.onnx).

Input: a window of 21 tokens (candidate in the middle); each token is up to 18
characters (Latin + Devanagari) plus 5 shape features. A char-CNN embeds each token,
a BiGRU reads the window, and attention + the candidate's own state give P(medicine).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from khurak.text import CHARS, MAX_CHARS, WINDOW, N_SHAPE  # noqa: E402

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "khurak", "data", "context.onnx")


class CtxNet(nn.Module):
    def __init__(self, n_chars=len(CHARS) + 2, ce=24, tok=96, hid=64):
        super().__init__()
        self.emb = nn.Embedding(n_chars, ce, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(ce, tok // 3, k, padding=k // 2) for k in (2, 3, 4)])
        self.proj = nn.Linear(tok + N_SHAPE + 1, tok)
        self.gru = nn.GRU(tok, hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hid, 1)
        self.out = nn.Sequential(nn.Linear(4 * hid + tok, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, chars, shapes, mark):
        B, T, C = chars.shape
        e = self.emb(chars.reshape(B * T, C)).transpose(1, 2)            # [B*T, ce, C]
        h = torch.cat([F.relu(c(e)).amax(2) for c in self.convs], 1)      # [B*T, tok]
        h = h.reshape(B, T, -1)
        x = torch.relu(self.proj(torch.cat([h, shapes, mark.unsqueeze(2)], 2)))
        g, _ = self.gru(x)                                               # [B, T, 2h]
        a = torch.softmax(self.att(g).squeeze(2), 1)
        pooled = (g * a.unsqueeze(2)).sum(1)
        center = g[:, WINDOW]
        return self.out(torch.cat([pooled, center, x[:, WINDOW]], 1)).squeeze(1)


class Exportable(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, chars, shapes, mark):
        return torch.sigmoid(self.net(chars, shapes, mark))


def load(split):
    d = np.load(os.path.join(HERE, f"ctx_{split}.npz"))
    return (torch.from_numpy(d["chars"].astype(np.int64)), torch.from_numpy(d["shapes"].astype(np.float32)),
            torch.from_numpy(d["y"].astype(np.float32)), d["src"])


DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def predict(net, chars, shapes, bs=4096):
    net.eval()
    out = []
    with torch.no_grad():
        mark = torch.zeros(1, chars.shape[1], device=DEV)
        mark[0, WINDOW] = 1
        for i in range(0, len(chars), bs):
            c, s = chars[i:i + bs].to(DEV), shapes[i:i + bs].to(DEV)
            out.append(torch.sigmoid(net(c, s, mark.expand(len(c), -1))).cpu())
    return torch.cat(out).numpy()


def main():
    torch.manual_seed(0)
    Xc, Xs, y, _ = load("train")
    Tc, Ts, ty, tsrc = load("test")
    n = len(y)
    perm = torch.randperm(n)
    val = perm[: n // 20]
    tr = perm[n // 20:]
    pos_w = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
    print(f"train {len(tr)} (pos {int(y[tr].sum())}) val {len(val)} test {len(ty)} (pos {int(ty.sum())}) pos_w {pos_w:.2f}")
    net = CtxNet().to(DEV)
    print(sum(p.numel() for p in net.parameters()), "params")
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    mark = torch.zeros(1, Xc.shape[1], device=DEV)
    mark[0, WINDOW] = 1
    best, best_state = -1, None
    for ep in range(int(os.environ.get("EPOCHS", "8"))):
        net.train()
        idx = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]
            logit = net(Xc[b].to(DEV), Xs[b].to(DEV), mark.expand(len(b), -1))
            loss = F.binary_cross_entropy_with_logits(logit, y[b].to(DEV), pos_weight=torch.tensor(min(pos_w, 3.0), device=DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        pv = predict(net, Xc[val], Xs[val])
        ap = average_precision_score(y[val].numpy(), pv)
        print(f"ep {ep + 1} loss {tot / len(tr):.4f} val AP {ap:.4f}", flush=True)
        if ap > best:
            best, best_state = ap, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    # threshold: maximise F1 on validation
    pv = predict(net, Xc[val], Xs[val])
    yv = y[val].numpy()
    ths = np.linspace(0.05, 0.95, 91)
    f1s = [precision_recall_fscore_support(yv, pv >= t, average="binary", zero_division=0)[2] for t in ths]
    th = float(ths[int(np.argmax(f1s))])
    pt = predict(net, Tc, Ts)
    yt = ty.numpy()
    p, r, f, _ = precision_recall_fscore_support(yt, pt >= th, average="binary", zero_division=0)
    res = {"threshold": th, "test": {"precision": p, "recall": r, "f1": f, "ap": average_precision_score(yt, pt),
                                     "n": int(len(yt)), "positives": int(yt.sum())}, "by_source": {}}
    for s in sorted(set(tsrc)):
        m = tsrc == s
        neg = m & (yt == 0)
        posm = m & (yt == 1)
        res["by_source"][s] = {"n": int(m.sum()),
                               "false_positive_rate": float((pt[neg] >= th).mean()) if neg.any() else None,
                               "recall": float((pt[posm] >= th).mean()) if posm.any() else None}
    # baseline: dictionary only (every candidate is a medicine)
    res["dictionary_only"] = {"precision": float(yt.mean()), "recall": 1.0,
                              "f1": float(2 * yt.mean() / (1 + yt.mean()))}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "..", "bench", "out", "context_model.json"), "w"), indent=1)

    net = net.cpu()
    ex = Exportable(net.eval())
    dummy = (Tc[:2], Ts[:2], torch.zeros(2, Tc.shape[1]).index_fill_(1, torch.tensor([WINDOW]), 1.0))
    torch.onnx.export(ex, dummy, OUT, input_names=["chars", "shapes", "mark"], output_names=["prob"],
                      dynamic_axes={"chars": {0: "b"}, "shapes": {0: "b"}, "mark": {0: "b"}, "prob": {0: "b"}},
                      opset_version=17, dynamo=False)
    json.dump({"threshold": th}, open(OUT.replace(".onnx", ".json"), "w"))
    print("exported", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
