# -*- coding: utf-8 -*-
"""
训练 1D-CNN 时序模型（纯 numpy）

用法：
  python train_cnn.py --epochs 25                    # 全难度 → model_cnn.npz
  python train_cnn.py --diff-only 5 --epochs 30 --out model_cnn_diff5.npz \
                      --metrics metrics_cnn_diff5.json

数据来自 data/features.npz（已剔除 ∅ 与 -，只保留真实谱师）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import features as F
from cnn import CnnNet, softmax
from train import SEED, Adam, load, tune_alpha, topk_metrics, macro_f1


def seq_len(d, idx):
    return (d["cellseq_off"][idx + 1] - d["cellseq_off"][idx]).astype(np.int64)


def pad_batch(d, idx, cap, chg_cap=F.CHG_LEN):
    """返回 (ids[B,L], mask[B,L], chg_ids[B,Lc], chg_w[B,Lc])"""
    lens = seq_len(d, idx)
    L = int(min(cap, max(1, lens.max())))
    ids = np.zeros((len(idx), L), dtype=np.int64)
    mask = np.zeros((len(idx), L), dtype=np.float32)
    for r, i in enumerate(idx):
        a, b = int(d["cellseq_off"][i]), int(d["cellseq_off"][i + 1])
        n = min(b - a, L)
        if n:
            ids[r, :n] = d["cellseq_flat"][a:b][:n]
            mask[r, :n] = 1.0
    clens = (d["chg_off"][idx + 1] - d["chg_off"][idx]).astype(np.int64)
    Lc = int(min(chg_cap, max(1, clens.max())))
    cids = np.zeros((len(idx), Lc), dtype=np.int64)
    cw = np.zeros((len(idx), Lc), dtype=np.float32)
    for r, i in enumerate(idx):
        a, b = int(d["chg_off"][i]), int(d["chg_off"][i + 1])
        n = min(b - a, Lc)
        if n:
            cids[r, :n] = d["chg_flat"][a:b][:n]
            cw[r, :n] = 1.0 / (b - a)
    return ids, mask, cids, cw


def infer(net, d, idx, mean, std, prior, cap, bs=64, chg_cap=F.CHG_LEN):
    out = np.zeros((len(idx), len(prior)), dtype=np.float32)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        ids, mask, cids, cw = pad_batch(d, b, cap, chg_cap)
        Xn = (d["num"][b] - mean) / std
        logits, _ = net.forward(ids, mask, cids, cw, Xn)
        out[s:s + len(b)] = logits
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(here, "data", "features.npz"))
    ap.add_argument("--out", default=os.path.join(here, "model_cnn.npz"))
    ap.add_argument("--metrics", default=os.path.join(here, "metrics_cnn.json"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--c", type=int, default=24)
    ap.add_argument("--o", type=int, default=48)
    ap.add_argument("--dh", type=int, default=256)
    ap.add_argument("--d-chg", type=int, default=48)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--chg-cap", type=int, default=800)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--wd", type=float, default=0.0, help="L2 权重衰减（Adam 后再收缩）")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--diff-only", type=int, default=0)
    args = ap.parse_args()

    d = load(args.features)
    labels = [str(x) for x in d["labels"]]
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
    # 同 train.py：训练集中恒定不变的特征（std=0）若只加 epsilon，会在推理时对
    # 该特征取值不同的样本产生巨大的 z 值，淹没其他特征。改用 1.0 避免此问题。
    std = np.where(std < 1e-6, 1.0, std)
    cnt = np.bincount(y[tr], minlength=nc).astype(np.float32)
    prior = np.log((cnt + 1.0) / (cnt.sum() + nc)).astype(np.float32)
    cw = (1.0 / np.sqrt(cnt + 1.0))
    cw = (cw / cw.mean()).astype(np.float32)

    net = CnnNet(nc, d_chg=args.d_chg, dn=num.shape[1], dh=args.dh,
                 c=args.c, o=args.o, seq_len=args.cap, drop=args.dropout)
    params = net.params()
    opt = Adam(params, lr=args.lr)
    steps = int(np.ceil(len(tr) / args.bs))
    best = {"score": -1.0}
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        order = rng.permutation(tr)
        tot = 0.0
        lr_ep = args.lr * 0.5 * (1 + np.cos(np.pi * (ep - 1) / max(1, args.epochs)))
        for s in range(0, len(order), args.bs):
            b = order[s:s + args.bs]
            ids, mask, cids, chw = pad_batch(d, b, args.cap, args.chg_cap)
            Xn = (num[b] - mean) / std
            yb = y[b]
            logits, cache = net.forward(ids, mask, cids, chw, Xn, training=True, rng=rng)
            p = softmax(logits)
            nb = len(b)
            w = cw[yb]
            tot += float(-(w * np.log(p[np.arange(nb), yb] + 1e-12)).mean())
            dlog = p.copy()
            dlog[np.arange(nb), yb] -= 1.0
            dlog *= (w / w.mean())[:, None] / nb
            grads = net.backward(dlog, cache)
            opt.step(params, grads, lr=lr_ep)
            if args.wd:
                for p in params:
                    p -= lr_ep * args.wd * p
        if ep % 2 == 0 or ep == args.epochs:
            lg = infer(net, d, va, mean, std, prior, args.cap, chg_cap=args.chg_cap)
            a, acc = tune_alpha(lg, y[va], prior)
            print("ep %2d loss=%.4f alpha=%.2f top1=%.4f  [%.0fs]"
                  % (ep, tot / max(1, steps), a, acc, time.time() - t0), flush=True)
            if acc > best["score"]:
                best = {"score": acc, "ep": ep, "alpha": a, "logits": lg}

    lg, alpha = best["logits"], best["alpha"]
    m = topk_metrics(lg, y[va], prior, alpha=alpha)
    res = {
        "arch": "cnn",
        "classes": nc,
        "samples": int(n),
        "train": int(len(tr)),
        "val": int(len(va)),
        "best_epoch": best["ep"],
        "alpha": alpha,
        "val_metrics": m,
        "val_macro_f1": macro_f1(lg, y[va], nc, prior * alpha),
        "baseline_majority_top1": round(float(cnt.max() / cnt.sum()), 4),
    }
    per = {}
    for dd in (2, 3, 4, 5, 6, 7):
        sel = va[diff[va] == dd]
        if len(sel) == 0:
            continue
        sub = infer(net, d, sel, mean, std, prior, args.cap, chg_cap=args.chg_cap)
        per[str(dd)] = {"n": int(len(sel)), **topk_metrics(sub, y[sel], prior, alpha=alpha)}
    res["val_by_diff"] = per
    res["label_distribution_top10"] = [[labels[i], int(cnt[i])] for i in np.argsort(-cnt)[:10]]

    np.savez_compressed(
        args.out,
        arch=np.asarray("cnn"),
        E=net.E, E2=net.E2, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2,
        W3=net.W3, b3=net.b3, Wf=net.Wf, bf=net.bf, Wo=net.Wo, bo=net.bo,
        num_mean=mean, num_std=std, prior=prior, alpha=np.asarray(alpha, dtype=np.float32),
        labels=np.asarray(labels, dtype=object),
        c=np.asarray(args.c), o=np.asarray(args.o), dh=np.asarray(args.dh),
        chg_cap=np.asarray(args.chg_cap), dropout=np.asarray(args.dropout),
        d_chg=np.asarray(args.d_chg), cap=np.asarray(args.cap),
        best_epoch=np.asarray(best["ep"]),
    )
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
