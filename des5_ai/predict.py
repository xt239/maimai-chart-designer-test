# -*- coding: utf-8 -*-
"""
推理：给定 maidata.txt / 谱面包（zip），只看 &inote_N= → E 的音符区间，
识别 &des_N= 区间，并预测其中的谱师（仅在「有署名」标签空间内）。

支持两种架构与任意加权集成：
  model_bag.npz  词袋（音符格 + 字符 n-gram + 结构化特征 → MLP）
  model_cnn.npz  1D-CNN 时序（音符格序列 → 三层卷积 + 掩码池化）

用法：
  python predict.py <maidata.txt | xxx.zip | 目录> [--diff 5] [--top 5]
  python predict.py <path> --models "model_bag.npz:1,model_cnn.npz:0.2" --alpha 0.5

输出示例：
  [inote_5 区间] 字符 [329,12410)  第 329~519 行
  [des_5   区间] 字符 [322,333)  值区间 [329,333)  实际值 = Jack
  预测 top5     Jack 0.42 / サファ太 0.17 / ...
"""
import argparse
import json
import os
import re
import sys
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import features as F
from cnn import CnnNet
from extract_dataset import decode, parse_maidata, EMPTY_LABEL

PY = os.path.dirname(os.path.abspath(__file__))
NAMELESS = {EMPTY_LABEL, "-"}


# ------------------------------------------------------------------ 模型
def _pad(ids_list, cap):
    L = max(1, min(cap, max((len(v) for v in ids_list), default=1)))
    ids = np.zeros((len(ids_list), L), dtype=np.int64)
    w = np.zeros((len(ids_list), L), dtype=np.float32)
    for r, v in enumerate(ids_list):
        n = min(len(v), L)
        if n:
            ids[r, :n] = v[:n]
            w[r, :n] = 1.0 / len(v)
    return ids, w


class BaseModel:
    arch = "?"

    def __init__(self, path):
        self.path = path
        self.m = np.load(path, allow_pickle=True)
        self.labels = [str(x) for x in self.m["labels"]]
        self.num_mean = self.m["num_mean"]
        # 修复：训练时 std 用了 `std + 1e-6`，对「训练集中恒定不变」的特征（如
        # has_e_marker，训练集里全部为 1.0）会得到 ~1e-6 的病态极小 std。推理时若
        # 该特征值与训练集不同（如谱面缺少 E 结束标记 → has_e_marker=0），
        # (0-1)/1e-6 = -1e6，会彻底淹没其他特征，导致无论输入什么都输出固定谱师。
        # 这里把近零 std 视为退化特征，改用 1.0 做缩放（训练时这些特征恒为 0，
        # 用 1.0 缩放在正常情况下与原行为一致）。
        std = np.asarray(self.m["num_std"], dtype=np.float32)
        self.num_std = np.where(std < 1e-3, 1.0, std)
        self.prior = self.m["prior"]
        self.alpha = float(self.m["alpha"]) if "alpha" in self.m else 1.0

    def _num(self, nums):
        return (np.stack(nums).astype(np.float32) - self.num_mean) / self.num_std


class BagModel(BaseModel):
    arch = "bag"

    def __init__(self, path):
        super().__init__(path)
        m = self.m
        self.E1, self.E2 = m["E1"], m["E2"]
        self.W1, self.b1, self.W2, self.b2 = m["W1"], m["b1"], m["W2"], m["b2"]
        self.cap = int(m["cap_cell"]) if "cap_cell" in m else F.CELL_LEN + 1200

    def score(self, feats):
        bag_ids, chg_ids, nums = zip(*feats)
        i1, w1 = _pad(bag_ids, self.cap)
        i2, w2 = _pad(chg_ids, F.CHG_LEN)
        X1 = (self.E1[i1] * w1[:, :, None]).sum(axis=1)
        X2 = (self.E2[i2] * w2[:, :, None]).sum(axis=1)
        h = np.concatenate([X1, X2, self._num(nums)], axis=1)
        return np.maximum(h @ self.W1 + self.b1, 0.0) @ self.W2 + self.b2


class CnnModel(BaseModel):
    arch = "cnn"

    def __init__(self, path):
        super().__init__(path)
        m = self.m
        self.c, self.o = int(m["c"]), int(m["o"])
        self.dh, self.d_chg = int(m["dh"]), int(m["d_chg"])
        self.cap = int(m["cap"])
        self.chg_cap = int(m["chg_cap"]) if "chg_cap" in m else F.CHG_LEN
        self.net = CnnNet(len(self.labels), d_chg=self.d_chg, dn=self.num_mean.shape[0],
                          dh=self.dh, c=self.c, o=self.o, seq_len=self.cap, seed=1)
        for k in ("E", "E2", "W1", "b1", "W2", "b2", "W3", "b3", "Wf", "bf", "Wo", "bo"):
            setattr(self.net, k, m[k])

    def score(self, feats):
        seq_ids, chg_ids, nums = zip(*feats)
        ids, mask = _pad_mask(seq_ids, self.cap)
        cids, cw = _pad(chg_ids, self.chg_cap)
        logits, _ = self.net.forward(ids, mask, cids, cw, self._num(nums))
        return logits


def _pad_mask(ids_list, cap):
    L = max(1, min(cap, max((len(v) for v in ids_list), default=1)))
    ids = np.zeros((len(ids_list), L), dtype=np.int64)
    mask = np.zeros((len(ids_list), L), dtype=np.float32)
    for r, v in enumerate(ids_list):
        n = min(len(v), L)
        if n:
            ids[r, :n] = v[:n]
            mask[r, :n] = 1.0
    return ids, mask


def load_model(path):
    m = np.load(path, allow_pickle=True)
    arch = str(m["arch"]) if "arch" in m else "bag"
    m.close()
    return CnnModel(path) if arch == "cnn" else BagModel(path)


class Predictor:
    def __init__(self, specs, alpha=None):
        """specs: [(path, weight), ...]"""
        self.models = [load_model(p) for p, _w in specs]
        w = np.asarray([x for _p, x in specs], dtype=np.float32)
        self.weights = w / w.sum()
        self.alpha = alpha if alpha is not None else self.models[0].alpha
        self.prior = self.models[-1].prior
        self.labels = self.models[0].labels
        self.arches = "+".join(m.arch for m in self.models)

    def predict(self, notes_list, top=5):
        feats = [featurize_all(s) for s in notes_list]
        z = None
        for mdl, wt in zip(self.models, self.weights):
            lg = mdl.score(feats) * wt
            z = lg if z is None else z + lg
        z = z + self.alpha * self.prior
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        order = np.argsort(-p, axis=1)[:, :top]
        return [[(self.labels[j], float(p[i, j])) for j in order[i]] for i in range(p.shape[0])]


def featurize_all(notes):
    """(音符格序列, 字符 n-gram, 结构化数值特征)"""
    bag_ids, chg_ids, num = F.featurize(notes)
    return F.sequence_ids(notes), chg_ids, num


# ------------------------------------------------------------------ 文件解析
def read_maidata(path: str) -> Optional[Tuple[str, str]]:
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if os.path.basename(n).lower() == "maidata.txt"]
            if not names:
                return None
            raw = zf.read(names[0])
    else:
        with open(path, "rb") as f:
            raw = f.read()
    return decode(raw)


def line_of(text: str, off: int) -> int:
    return text.count("\n", 0, off) + 1


def locate_des_region(text: str, diff: int, inote_start: int) -> Optional[Dict[str, Any]]:
    """&des_N= 恒位于 &inote_N= 上一行；据此从音符区间反推 des 区间（取值可含空格）。"""
    tag_len = len("&des_%d=" % diff)
    best = None
    for m in re.finditer(r"(?m)^&des_%d=(.*)$" % diff, text[:inote_start]):
        best = m
    if best is None:
        return None
    value = best.group(1).rstrip("\r")
    vs = best.start() + tag_len
    return {"tag_span": [best.start(), vs + len(value)],
            "value_span": [vs, vs + len(value)], "value": value}


def expand_targets(path: str, limit: int = 20) -> Tuple[List[str], bool]:
    """目录 → 其下的 *.zip 列表（最多 limit 个）；单文件 → ([path], False)。"""
    if not os.path.isdir(path):
        return [path], False
    files: List[str] = []
    for dp, _dirs, fs in os.walk(path):
        files.extend(os.path.join(dp, f) for f in fs if f.lower().endswith(".zip"))
    files.sort()
    return files[:limit], True


def analyze_file(path: str, predictor: "Predictor", diffs: Optional[List[int]], top: int,
                 want_text: bool = True, text_limit: int = 600) -> Optional[Dict[str, Any]]:
    got = read_maidata(path)
    if got is None:
        return None
    text, enc = got
    meta, samples = parse_maidata(text)
    by_diff = {s["diff"]: s for s in samples}
    todo = [d for d in (diffs or sorted(by_diff)) if d in by_diff]
    if not todo:
        return None
    preds = predictor.predict([by_diff[d]["notes"] for d in todo], top=top)
    out: Dict[str, Any] = {"file": path, "encoding": enc, "meta": meta,
                           "arch": predictor.arches, "alpha": predictor.alpha, "results": []}
    for d, cand in zip(todo, preds):
        s = by_diff[d]
        a, b = s["inote_span"]
        region = locate_des_region(text, d, a)
        true_val = s["des_raw"]
        lab = true_val if true_val else EMPTY_LABEL
        in_label_space = lab not in NAMELESS
        rec: Dict[str, Any] = {
            "diff": d,
            "lv": s["lv"],
            "inote_span": [a, b],
            "inote_lines": [line_of(text, a), line_of(text, b)],
            "inote_chars": b - a,
            "notes_lines": s["notes_lines"],
            "has_e": s["has_e"],
            "des_region": region,
            "des_value": lab,
            "des_in_label_space": in_label_space,
            "top": [{"des": c, "p": round(p, 4)} for c, p in cand],
            "hit": bool(cand) and cand[0][0] == lab,
        }
        if want_text:
            rec["notes_preview"] = by_diff[d]["notes"][:text_limit]
        out["results"].append(rec)
    return out


# ------------------------------------------------------------------ CLI
def parse_specs(arg):
    specs = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            p, w = part.rsplit(":", 1)
            specs.append((os.path.abspath(p.strip()), float(w)))
        else:
            specs.append((os.path.abspath(part), 1.0))
    return specs


DEFAULT_MODELS = "model_bag.npz:1,model_cnn.npz:0.2"


def _utf8_stdout() -> None:
    """让 Windows 控制台也能打印日文谱师名。"""
    reconf = getattr(sys.stdout, "reconfigure", None)
    if reconf is not None:
        try:
            reconf(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="maidata.txt / zip / 目录（目录则批量 *.zip）")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="如 model_bag.npz:1,model_cnn.npz:0.2")
    ap.add_argument("--diff", default="5", help="难度：5 / 2,5 / all")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--json", default="")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    diffs = None if args.diff == "all" else [int(x) for x in args.diff.split(",")]
    predictor = Predictor(parse_specs(args.models), alpha=args.alpha)
    print("模型: %s   alpha=%.2f   标签数=%d" % (predictor.arches, predictor.alpha, len(predictor.labels)))

    target = args.target
    if os.path.isdir(target):
        files = []
        for dp, _d, fs in os.walk(target):
            files.extend(os.path.join(dp, f) for f in fs if f.lower().endswith(".zip"))
        files.sort()
        files = files[: args.limit]
    else:
        files = [target]

    out: List[Dict[str, Any]] = []
    for f in files:
        print("=" * 78)
        print("文件: %s" % f)
        rec = analyze_file(f, predictor, diffs, args.top)
        if not rec:
            print("  跳过（无可用谱面）")
            continue
        for r in rec["results"]:
            print("  &inote_%d 区间: 字符 [%d,%d)  第 %d~%d 行  (%d 字符 / %d 行音符, E 收尾=%s)"
                  % (r["diff"], r["inote_span"][0], r["inote_span"][1],
                     r["inote_lines"][0], r["inote_lines"][1], r["inote_chars"],
                     r["notes_lines"], "是" if r["has_e"] else "否"))
            reg = r["des_region"]
            if reg:
                print("  &des_%d   区间: 字符 [%d,%d)  值区间 [%d,%d)  → 实际值 %s"
                      % (r["diff"], reg["tag_span"][0], reg["tag_span"][1],
                         reg["value_span"][0], reg["value_span"][1], repr(r["des_value"])))
            else:
                print("  &des_%d   区间: 文件中缺失" % r["diff"])
            if not r["des_in_label_space"]:
                print("  提示: 该谱无署名（%s），不在模型标签空间内，以下为「谁可能打这首歌」的候选"
                      % r["des_value"])
            print("  预测 top%d: %s" % (args.top, "  /  ".join("%s %.3f" % (c["des"], c["p"]) for c in r["top"])))
            print("  命中: %s" % ("✔" if r["hit"] else "✘"))
        m = rec["meta"]
        print("  曲名: %s  /  作曲: %s  /  版本: %s"
              % (m.get("title", ""), m.get("artist", ""), m.get("version", "")))
        out.append(rec)

    hits = [r["hit"] for rec in out for r in rec["results"] if r["des_in_label_space"]]
    if hits:
        print("=" * 78)
        print("top1 命中率（仅统计有署名的谱）: %d/%d = %.1f%%"
              % (sum(hits), len(hits), 100.0 * sum(hits) / len(hits)))
    if args.json and out:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("json -> %s" % args.json)


if __name__ == "__main__":
    main()
