# -*- coding: utf-8 -*-
"""
本地交互界面：上传 zip（或 maidata.txt）到 ./temp → 点击识别 → 返回区间定位与谱师预测。
每份文件分析完成后即刻从 ./temp 删除，不做基于时间的定时清理。

启动： python server.py [--port 8000] [--host 127.0.0.1]
依赖：仅标准库（http.server）+ numpy；模型文件需已训练好。
"""
import argparse
import json
import os
import posixpath
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

import predict as P  # noqa: E402

PY = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(PY, "static")
TEMP = os.path.join(PY, "temp")          # 上传谱包暂存目录（./temp）
CHUNK = 1 << 16                          # 上传落盘分块大小

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

STATE = {"predictor": None, "specs": P.DEFAULT_MODELS, "alpha": None,
         "max_upload": 200 * 1024 * 1024}


def get_predictor(specs, alpha):
    key = (specs, alpha)
    if STATE["predictor"] is None or STATE["_key"] != key:
        missing = [p for p, _w in P.parse_specs(specs) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("缺少模型文件: %s" % ", ".join(missing))
        STATE["predictor"] = P.Predictor(P.parse_specs(specs), alpha=alpha)
        STATE["_key"] = key
    return STATE["predictor"]


def safe_name(raw):
    """清洗上传/引用的文件名：只取 basename，剔除危险字符，防止路径穿越。"""
    name = (raw or "").strip().strip('"').replace("\\", "/").split("/")[-1].strip()
    name = "".join(c for c in name if c >= " " and c not in '<>:"|?*')
    name = name.strip().strip(".")
    if not name:
        raise ValueError("文件名无效")
    if len(name) > 180:
        root, ext = os.path.splitext(name)
        name = root[:180 - len(ext)] + ext
    return name


def discard(fp):
    """分析完成后即刻删除 temp 中的该文件；失败只告警，不影响结果返回。"""
    try:
        os.remove(fp)
        print("[清理] 分析完成，已删除 %s" % os.path.basename(fp), flush=True)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print("[清理] 删除失败 %s: %s" % (fp, exc), flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "des5ai/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # ---------- 工具 ----------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path):
        rel = path.lstrip("/") or "index.html"
        rel = posixpath.normpath(rel).replace("\\", "/")
        if rel.startswith(".."):
            return self._send(403, {"error": "forbidden"})
        fp = os.path.join(STATIC, rel.replace("/", os.sep))
        if not os.path.isfile(fp):
            return self._send(404, {"error": "not found"})
        ctype = MIME.get(os.path.splitext(fp)[1].lower(), "application/octet-stream")
        with open(fp, "rb") as f:
            self._send(200, f.read(), ctype)

    # ---------- 路由 ----------
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._static("index.html")
        if u.path == "/api/health":
            p = STATE["predictor"]
            return self._send(200, {
                "ok": True,
                "models": STATE["specs"],
                "arch": p.arches if p else None,
                "alpha": p.alpha if p else None,
                "labels": len(p.labels) if p else 0,
                "temp": "temp",
            })
        if u.path.startswith("/static/"):
            return self._static(u.path[len("/static/"):])
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/upload":
            return self._upload(u)
        if u.path == "/api/predict":
            return self._predict()
        return self._send(404, {"error": "not found"})

    # ---------- 上传谱包：原始字节流 + ?name=文件名，落盘到 ./temp ----------
    def _upload(self, u):
        raw = (parse_qs(u.query).get("name") or [""])[0] or self.headers.get("X-Filename") or ""
        try:
            name = safe_name(raw)
        except ValueError as exc:
            self.close_connection = True    # 请求体未被读取，直接断开
            return self._send(400, {"error": str(exc)})
        if os.path.splitext(name)[1].lower() not in (".zip", ".txt"):
            self.close_connection = True
            return self._send(400, {"error": "仅支持 .zip 谱包或 maidata.txt，收到：%s" % name})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            self.close_connection = True
            return self._send(400, {"error": "上传内容为空，请在页面选择文件后再上传"})
        if n > STATE["max_upload"]:
            self.close_connection = True
            mb = 1024.0 * 1024.0
            return self._send(413, {"error": "文件过大（%.1f MB），上限 %.0f MB"
                                    % (n / mb, STATE["max_upload"] / mb)})
        os.makedirs(TEMP, exist_ok=True)
        dest = os.path.join(TEMP, name)
        part = dest + ".part"
        try:
            left = n
            with open(part, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(CHUNK, left))
                    if not chunk:
                        raise IOError("连接被提前关闭")
                    f.write(chunk)
                    left -= len(chunk)
            os.replace(part, dest)      # 同名覆盖，原子替换
        except Exception as exc:
            try:
                os.remove(part)
            except OSError:
                pass
            traceback.print_exc()
            return self._send(500, {"error": "保存失败：%s" % exc})
        size = os.path.getsize(dest)
        print("[上传] %s（%.1f KB）→ temp" % (name, size / 1024.0))
        return self._send(200, {"ok": True, "name": name, "size": size,
                                "message": "上传成功：%s" % name})

    # ---------- 识别：只接受 ./temp 下的文件名 ----------
    def _predict(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            names = req.get("files") or req.get("file") or []
            if isinstance(names, str):
                names = [names]
            if not names and (req.get("path") or "").strip():
                return self._send(400, {"error": "已改为上传模式：请在页面选择谱包上传（文件保存到 ./temp），再点击识别"})
            try:
                names = [safe_name(str(x)) for x in names]
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})
            names = [x for x in names if x][:int(req.get("limit") or 20)]
            if not names:
                return self._send(400, {"error": "请先上传谱包（.zip 或 maidata.txt），再点击识别"})
            specs = req.get("models") or STATE["specs"]
            alpha = req.get("alpha")
            alpha = float(alpha) if alpha not in (None, "") else None
            pred = get_predictor(specs, alpha)
            diffs = req.get("diffs") or None
            if isinstance(diffs, str):
                diffs = [int(x) for x in diffs.replace(" ", "").split(",") if x.isdigit()] or None
            top = int(req.get("top") or 6)
            paths = [os.path.join(TEMP, x) for x in names]
            if not any(os.path.isfile(p) for p in paths):
                return self._send(400, {"error": "上传的文件已不存在（可能已识别过被即刻清理），请重新上传后再识别"})
            results = []
            for fp in paths:
                if not os.path.isfile(fp):
                    results.append({"file": fp, "error": "文件不存在（已清理或未上传），请重新上传"})
                    continue
                rec = None
                try:
                    rec = P.analyze_file(fp, pred, diffs, top,
                                         want_text=bool(req.get("showText", True)))
                except Exception as exc:  # 单个坏包不影响整批
                    results.append({"file": fp, "error": "%s: %s" % (type(exc).__name__, exc)})
                    continue
                finally:
                    discard(fp)     # 该文件分析完成（无论成败）→ 即刻清理
                if rec:
                    results.append(rec)
            if not results:
                return self._send(400, {"error": "该文件下没有可用的谱面（缺少 &inote_N= → E 区间）"})
            is_batch = len(paths) > 1
            payload = {"ok": True, "batch": is_batch, "count": len(results), "results": results}
            if not is_batch and results and "error" not in results[0]:
                payload["result"] = results[0]   # 兼容单文件旧字段
            return self._send(200, payload)
        except FileNotFoundError as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:  # 把后端异常原样带回前端，便于排查
            traceback.print_exc()
            return self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models", default=P.DEFAULT_MODELS)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--max-upload-mb", type=float, default=200,
                    help="单个上传文件大小上限（MB），默认 200")
    args = ap.parse_args()

    STATE["specs"] = args.models
    STATE["alpha"] = args.alpha
    STATE["max_upload"] = int(args.max_upload_mb * 1024 * 1024)
    os.makedirs(TEMP, exist_ok=True)
    try:
        get_predictor(args.models, args.alpha)
        print("模型已加载: %s" % args.models)
    except Exception as exc:
        print("警告：%s（界面仍会启动，但预测会报错）" % exc)

    print("清理策略: 每份文件分析完成后即刻删除（%s）" % TEMP)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("界面已启动: http://%s:%d" % (args.host, args.port))
    print("按 Ctrl+C 停止")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
