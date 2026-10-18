# -*- coding: utf-8 -*-
"""
抽取 maimai 谱面数据集。

输入：MaiArchive 全版本目录（每个 *.zip 内含 maidata.txt / track.mp3 / bg.png）
输出：dataset.jsonl  每一行 = 一条样本
      {
        "zip":        相对路径,
        "version":    &version=,
        "title":      &title=,
        "artist":     &artist=,
        "shortid":    &shortid=,
        "diff":       难度号(2..7),
        "lv":         该难度星数,
        "des":        &des_N= 的值（谱师，可能为空）,
        "des_span":   [起始字符偏移, 结束字符偏移]   ← 目标区间
        "inote_span": [起始字符偏移, 结束字符偏移]   ← 输入区间（&inote_N= 之后 ~ E 标记行）
        "has_e":      音符块是否以 E 结束,
        "notes_lines":音符块行数,
        "notes":      音符区间原文
      }
      summary.json   统计信息

用法： python extract_dataset.py [--root DIR] [--out DIR]
"""
import argparse
import json
import os
import re
import sys
import zipfile
from collections import Counter
from typing import Any, Dict, List, Tuple

LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n|$)")
TAG_RE = re.compile(r"^&([A-Za-z_][A-Za-z_0-9]*)=")
EMPTY_LABEL = "\u2205"  # ∅ 表示 &des_N= 为空

META_KEYS = ("title", "artist", "shortid", "genre", "version", "versionid")


def iter_lines(text: str):
    """按行遍历并保留每行起始字符偏移。产出 (行首偏移, 行内容)。"""
    pos = 0
    for m in LINE_RE.finditer(text):
        s = m.group(0)
        if s == "":
            continue
        body = s.rstrip("\r\n")
        yield pos, body
        pos += len(s)


def decode(raw: bytes) -> Tuple[str, str]:
    for enc in ("utf-8-sig", "cp932", "utf-8", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace"), "latin-1/replace"


def parse_maidata(text: str) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """解析 maidata.txt，返回 (meta, samples)。

    meta: {title, artist, shortid, genre, version, versionid}
    samples: 每条含 diff / lv / des / des_raw / des_span / inote_span / has_e /
             notes_lines / notes，其中 des_span 与 inote_span 为 [起始偏移, 结束偏移]。
    """
    lines = list(iter_lines(text))
    tags = []  # (line_idx, tag_name, tag_len, line_start, body)
    for i, (off, body) in enumerate(lines):
        m = TAG_RE.match(body)
        if m:
            tags.append((i, m.group(1), len(m.group(0)), off, body))

    meta: Dict[str, str] = {}
    for _, name, _, _, body in tags:
        if name in META_KEYS and name not in meta:
            meta[name] = body.split("=", 1)[1].strip()

    lv: Dict[int, Tuple[str, int, int]] = {}
    des: Dict[int, Tuple[str, int, int, int]] = {}
    for idx, (i, name, tlen, off, body) in enumerate(tags):
        m = re.fullmatch(r"(lv|des)_(\d+)", name)
        if not m:
            continue
        kind, num = m.group(1), int(m.group(2))
        raw_val = body[tlen:]
        lead = len(raw_val) - len(raw_val.lstrip(" \t"))
        start = off + tlen + lead
        value = raw_val.strip()
        end = start + len(value)
        if kind == "lv":
            lv[num] = (value, start, end)
        else:
            des[num] = (value, start, end, i)

    samples: List[Dict[str, Any]] = []
    for i, name, tlen, off, body in tags:
        m = re.fullmatch(r"inote_(\d+)", name)
        if not m:
            continue
        num = int(m.group(1))
        start = off + tlen
        has_e = False
        end = len(text)
        n_lines = 0
        j = i + 1
        while j < len(lines):
            joff, jbody = lines[j]
            if jbody.strip() == "E":
                has_e = True
                end = joff + len(jbody.rstrip())
                break
            if jbody.startswith("&"):
                end = lines[j - 1][0] + len(lines[j - 1][1]) if j - 1 > i else start
                break

            j += 1
        else:
            end = lines[-1][0] + len(lines[-1][1]) if lines else start
        n_lines = j - i - 1
        notes = text[start:end]
        if not notes.strip():
            continue
        d_val, d_start, d_end, _ = des.get(num, ("", start, start, -1))
        label = d_val if d_val else EMPTY_LABEL
        samples.append(
            {
                "diff": num,
                "lv": lv.get(num, ("", 0, 0))[0],
                "des": label,
                "des_raw": d_val,
                "des_span": [d_start, d_end],
                "inote_span": [start, end],
                "has_e": has_e,
                "notes_lines": n_lines,
                "notes": notes,
            }
        )
    return meta, samples


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)) + "/..")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    zips = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".zip") and not f.startswith("_tmp"):
                zips.append(os.path.join(dirpath, f))
    zips.sort()
    print("found %d zip archives under %s" % (len(zips), root), flush=True)

    out_path = os.path.join(out_dir, "dataset.jsonl")
    stats = Counter()
    label_counter = Counter()
    enc_counter = Counter()
    diff_counter = Counter()
    n_written = 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as fout:
        for zi, zp in enumerate(zips):
            if zi % 200 == 0:
                print("  [%d/%d] %s" % (zi, len(zips), os.path.basename(zp)), flush=True)
            try:
                with zipfile.ZipFile(zp) as zf:
                    names = [n for n in zf.namelist() if os.path.basename(n).lower() == "maidata.txt"]
                    if not names:
                        stats["zip_without_maidata"] += 1
                        continue
                    raw = zf.read(names[0])
            except Exception as exc:  # 坏包
                stats["zip_broken"] += 1
                sys.stderr.write("broken: %s (%s)\n" % (zp, exc))
                continue

            text, enc = decode(raw)
            enc_counter[enc] += 1
            meta, samples = parse_maidata(text)
            stats["zip_ok"] += 1
            if not samples:
                stats["zip_no_samples"] += 1
            for s in samples:
                rec = {
                    "zip": os.path.relpath(zp, root).replace("\\", "/"),
                    "version": meta.get("version", ""),
                    "title": meta.get("title", ""),
                    "artist": meta.get("artist", ""),
                    "shortid": meta.get("shortid", ""),
                }
                rec.update(s)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
                label_counter[s["des"]] += 1
                diff_counter[s["diff"]] += 1
                stats["has_e" if s["has_e"] else "no_e"] += 1
                stats["des_empty" if s["des"] == EMPTY_LABEL else "des_filled"] += 1

    summary = {
        "root": root,
        "zips_found": len(zips),
        "samples": n_written,
        "stats": dict(stats),
        "encodings": dict(enc_counter),
        "samples_by_diff": dict(sorted(diff_counter.items())),
        "distinct_des": len(label_counter),
        "top_des": label_counter.most_common(30),
        "des_singletons": sum(1 for _k, v in label_counter.items() if v == 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
