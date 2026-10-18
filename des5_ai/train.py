# -*- coding: utf-8 -*-
"""
训练：读取音符区间（&inote_N= → E 的谱面原文）→ 预测 &des_N=（谱师）

纯 numpy 实现的前馈网络：
  EmbeddingBag(音符格 token)   ┐
  EmbeddingBag(字符 n-gram)    ├→ concat 数值特征 → FC256+ReLU+Dropout → FC(C)
类权重 1/sqrt(freq) 平衡训练；推理时叠加 log 先验以最大化 top-1。

用法：python train.py [--epochs 20] [--val-frac 0.12]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import features as F

SEED = 20261017


def load(path):
    """一次性把 npz 读入内存：NpzFile 每次 __getitem__ 都会重新解压，极慢。"""
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    z.close()
    return d


# ---------------------------------------------------------------- 嵌入层
def gather(flat, off, idx, cap):
    lens = (off[idx + 1] - off[idx]).astype(np.int64)
    L = int(min(cap, max(1, lens.max())))
    B = len(idx)
    ids = np.zeros((B, L), dtype=np.int64)
    w = np.zeros((B, L), dtype=np.float32)
    for r, i in enumerate(idx):
        a, b = int(off[i]), int(off[i + 1])
        n = min(b - a, L)
        if n <= 0:
            continue
        ids[r, :n] = flat[a:a + n]
        w[r, :n] = 1.0 / (b - a)
    return ids, w


def embed_forward(emb, ids, w):
    e = emb[ids] * w[:, :, None]
    return e.sum(axis=1)


def embed_backward(nb, d, ids, w, dX):
    """mean-pooling 对嵌入表的梯度（bincount 加速的 scatter-add）。"""
    B, L = ids.shape
    rows = np.repeat(np.arange(B), L)
    contrib = (w.reshape(-1, 1) * dX[rows]).astype(np.float32).reshape(-1)
    col = np.arange(d, dtype=np.int64)
    idx2 = (ids.reshape(-1, 1) * d + col[None, :]).reshape(-1)
    return np.bincount(idx2, weights=contrib, minlength=nb * d).astype(np.float32).reshape(nb, d)


# ---------------------------------------------------------------- 网络
class Net:
    def __init__(self, d1, d2, dn, dh, nc, seed=SEED):
        rng = np.random.default_rng(seed)
        self.E1 = (rng.standard_normal((F.CELL_BUCKETS, d1)) * 0.05).astype(np.float32)
        self.E2 = (rng.standard_normal((F.CHG_BUCKETS, d2)) * 0.05).astype(np.float32)
        din = d1 + d2 + dn
        self.W1 = (rng.standard_normal((din, dh)) * np.sqrt(2.0 / din)).astype(np.float32)
        self.b1 = np.zeros(dh, dtype=np.float32)
        self.W2 = (rng.standard_normal((dh, nc)) * np.sqrt(2.0 / dh)).astype(np.float32)
        self.b2 = np.zeros(nc, dtype=np.float32)

    def forward(self, X1, X2, Xn, training=False, rng=None):
        h = np.concatenate([X1, X2, Xn], axis=1)
        z1 = h @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        mask = None
        if training and rng is not None:
            mask = (rng.random(a1.shape) >= 0.25).astype(np.float32) / 0.75
            a1 = a1 * mask
        logits = a1 @ self.W2 + self.b2
        return logits, (h, z1, a1, mask)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, params, grads, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            p -= lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + self.eps)


def infer(net, d, idx, mean, std, prior, bs=128, cap_cell=None):
    cap_cell = cap_cell or (F.CELL_LEN + 1200)
    out = np.zeros((len(idx), len(prior)), dtype=np.float32)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        i1, w1 = gather(d["cell_flat"], d["cell_off"], b, cap_cell)
        i2, w2 = gather(d["chg_flat"], d["chg_off"], b, F.CHG_LEN)
        X1 = embed_forward(net.E1, i1, w1)
        X2 = embed_forward(net.E2, i2, w2)
        Xn = (d["num"][b] - mean) / std
        logits, _ = net.forward(X1, X2, Xn)
        out[s:s + len(b)] = logits
    return out


def topk_metrics(logits, y, prior, ks=(1, 3, 5), alpha=1.0):
    res = {}
    for mode in ("plain", "prior"):
        z = logits + alpha * prior if mode == "prior" else logits
        order = np.argsort(-z, axis=1)
        rank = np.argmax(order == y[:, None], axis=1)
        for k in ks:
            res["%s@%d" % (mode, k)] = round(float((rank < k).mean()), 4)
    return res


def tune_alpha(logits, y, prior, grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25)):
    best = (0.0, -1.0)
    for a in grid:
        z = logits + a * prior
        acc = float((np.argmax(z, axis=1) == y).mean())
        if acc > best[1]:
            best = (a, acc)
    return best[0], round(best[1], 4)


def macro_f1(logits, y, nc, prior):
    pred = np.argmax(logits + prior, axis=1)
    f1s = []
    for c in range(nc):
        tp = int(((pred == c) & (y == c)).sum())
        if tp == 0:
            continue
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r))
    return round(float(np.mean(f1s)), 4) if f1s else 0.0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(here, "data", "features.npz"))
    ap.add_argument("--out", default=os.path.join(here, "model.npz"))
    ap.add_argument("--metrics", default=os.path.join(here, "metrics.json"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d1", type=int, default=64)
    ap.add_argument("--d2", type=int, default=48)
    ap.add_argument("--dh", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--balanced", type=int, default=1)
    ap.add_argument("--diff-only", type=int, default=0,
                    help="只训练该难度（如 5 = MASTER），0 = 全部难度混合")
    args = ap.parse_args()

    d = load(args.features)
    labels = list(d["labels"])
    y = d["y"].astype(np.int64)
    group, diff = d["group"], d["diff"]
    num = d["num"].astype(np.float32)
    nc, n = len(labels), len(y)

    rng = np.random.default_rng(SEED)
    groups = np.unique(group)
    rng.shuffle(groups)
    n_val_g = int(len(groups) * args.val_frac)
    val_set = set(groups[:n_val_g].tolist())
    is_val = np.array([g in val_set for g in group])
    tr = np.where(~is_val)[0]
    va = np.where(is_val)[0]
    if args.diff_only:
        tr = tr[diff[tr] == args.diff_only]
        va = va[diff[va] == args.diff_only]
    print("train=%d val=%d classes=%d songs=%d/%d%s"
          % (len(tr), len(va), nc, len(groups) - n_val_g, n_val_g,
             ("  仅难度 %d" % args.diff_only) if args.diff_only else ""))

    mean = num[tr].mean(axis=0)
    std = num[tr].std(axis=0)
    # 修复：若某特征在整个训练集中恒定不变（std=0），用很小的 epsilon 会让推理时
    # 出现该特征值变化的样本产生巨大 z 值（如 (0-1)/1e-6 = -1e6），彻底淹没其他
    # 特征。这里将退化特征的 std 置为 1.0，等价于不缩放，只保留均值偏移。
    std = np.where(std < 1e-6, 1.0, std)
    cnt = np.bincount(y[tr], minlength=nc).astype(np.float32)
    prior = np.log((cnt + 1.0) / (cnt.sum() + nc)).astype(np.float32)
    cw = (1.0 / np.sqrt(cnt + 1.0)) if args.balanced else np.ones(nc, dtype=np.float32)
    cw = (cw / cw.mean()).astype(np.float32)

    net = Net(args.d1, args.d2, num.shape[1], args.dh, nc)
    params = [net.E1, net.E2, net.W1, net.b1, net.W2, net.b2]
    opt = Adam(params, lr=args.lr)
    cap_cell = F.CELL_LEN + 1200
    steps = int(np.ceil(len(tr) / args.bs))
    best = {"score": -1.0}
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        order = rng.permutation(tr)
        tot = 0.0
        lr_ep = args.lr * 0.5 * (1 + np.cos(np.pi * (ep - 1) / max(1, args.epochs)))
        for s in range(0, len(order), args.bs):
            b = order[s:s + args.bs]
            i1, w1 = gather(d["cell_flat"], d["cell_off"], b, cap_cell)
            i2, w2 = gather(d["chg_flat"], d["chg_off"], b, F.CHG_LEN)
            X1 = embed_forward(net.E1, i1, w1)
            X2 = embed_forward(net.E2, i2, w2)
            Xn = (num[b] - mean) / std
            yb = y[b]

            logits, (h, z1, a1, mask) = net.forward(X1, X2, Xn, True, rng)
            p = softmax(logits)
            nb = len(b)
            w = cw[yb]
            tot += float(-(w * np.log(p[np.arange(nb), yb] + 1e-12)).mean())
            dlog = p.copy()
            dlog[np.arange(nb), yb] -= 1.0
            dlog *= (w / w.mean())[:, None] / nb

            dz1 = dlog @ net.W2.T
            if mask is not None:
                dz1 = dz1 * mask
            dz1[z1 <= 0] = 0.0
            gW2 = a1.T @ dlog
            gb2 = dlog.sum(axis=0)
            dh = dz1 @ net.W1.T
            gW1 = h.T @ dz1
            gb1 = dz1.sum(axis=0)
            d1 = args.d1
            d2 = args.d2
            gE1 = embed_backward(F.CELL_BUCKETS, d1, i1, w1, dh[:, :d1])
            gE2 = embed_backward(F.CHG_BUCKETS, d2, i2, w2, dh[:, d1:d1 + d2])
            opt.step(params, [gE1, gE2, gW1, gb1, gW2, gb2], lr=lr_ep)

        if ep % 2 == 0 or ep == args.epochs:
            lg = infer(net, d, va, mean, std, prior, cap_cell=cap_cell)
            a, acc = tune_alpha(lg, y[va], prior)
            score = acc
            print("ep %2d loss=%.4f alpha=%.2f top1=%.4f  [%.0fs]"
                  % (ep, tot / max(1, steps), a, acc, time.time() - t0), flush=True)
            if score > best["score"]:
                best = {"score": score, "ep": ep, "alpha": a, "logits": lg}

    lg, alpha = best["logits"], best["alpha"]
    m = topk_metrics(lg, y[va], prior, alpha=alpha)
    res = {
        "classes": nc,
        "samples": int(n),
        "train": int(len(tr)),
        "val": int(len(va)),
        "best_epoch": best["ep"],
        "alpha": alpha,
        "val_metrics": m,
        "val_macro_f1": macro_f1(lg, y[va], nc, prior * alpha),
        "baseline_majority_top1": round(float(cnt.max() / cnt.sum()), 4),
        "label_distribution_top10": [[labels[i], int(cnt[i])] for i in np.argsort(-cnt)[:10]],
    }
    per = {}
    for dd in (2, 3, 4, 5, 6, 7):
        sel = va[diff[va] == dd]
        if len(sel) == 0:
            continue
        sub = infer(net, d, sel, mean, std, prior, cap_cell=cap_cell)
        per[str(dd)] = {"n": int(len(sel)), **topk_metrics(sub, y[sel], prior, alpha=alpha)}
    res["val_by_diff"] = per

    np.savez_compressed(
        args.out,
        E1=net.E1, E2=net.E2, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2,
        num_mean=mean, num_std=std, prior=prior, alpha=np.asarray(alpha, dtype=np.float32),
        labels=np.asarray(labels, dtype=object),
        d1=np.asarray(args.d1), d2=np.asarray(args.d2), dh=np.asarray(args.dh),
        cap_cell=np.asarray(cap_cell), best_epoch=np.asarray(best["ep"]),
    )
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
