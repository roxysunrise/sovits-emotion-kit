"""emotion_server —— 只要你有 GPT-SoVITS，就自动拥有情绪的一个小服务。

纯标准库（http.server + urllib），不装依赖、不改你的 SoVITS。

用法：
    python server/emotion_server.py --config config.json [--listen 9883]

接口：
    GET  /health                      → 配置、ref 库、GSV 连通性
    POST /tts     {"text": "...", "emotion": "开心", "text_lang": "auto", "lang": "zh"}
                                      → audio/wav（带情绪），响应头 X-Emotion / X-Segments
    POST /analyze {"text": "..."}     → JSON（判出的情绪 + 每小句的弧线）
    GET  /refs                        → 当前情绪 ref 的清单与指纹

情绪怎么来的：
    文本 → analyzer（VAD + 标签 + 逐句弧线）
         → refs.pick（换参考音频 ref + temperature/top_k/rep/fragment_interval/speed）
         → 你的 GPT-SoVITS（用那段 ref 直接"演"出来）
         → 拼接 + 去尾巴（不做变调/加气声这类后期）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from emotion import analyzer as A          # noqa: E402
from emotion import dsp, refs, persona as P  # noqa: E402
from emotion import renderer as R          # noqa: E402

CFG: dict = {}
_STATS_CACHE: dict = {"at": 0.0, "rows": []}


def ref_stats(force: bool = False, block: bool = True) -> list[dict] | None:
    """ref 指纹算一次就缓存 —— 10 个 4~5 秒的 ref 逐个算 F0/FFT 要十几秒，
    放在 /health 里每次都算会把健康检查拖死。block=False 时冷启动直接返回 None。"""
    import time as _t
    now = _t.time()
    cold = force or not _STATS_CACHE["rows"] or now - _STATS_CACHE["at"] > 300
    if not cold:
        return _STATS_CACHE["rows"]
    if not block:
        return None
    _STATS_CACHE["rows"] = refs.stats(CFG["refs_dir"])
    _STATS_CACHE["at"] = now
    return _STATS_CACHE["rows"]


# ───────────────────────── 配置 ─────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        cfg = json.load(f)
    cfg.setdefault("gsv_url", "http://127.0.0.1:9880")
    cfg.setdefault("refs_dir", "refs/emotional")
    cfg.setdefault("text_lang", "auto")
    cfg.setdefault("persona_file", "")
    cfg.setdefault("clean_refs", False)
    cfg.setdefault("per_segment_ref", False)
    cfg.setdefault("tail_trim_db", -45.0)
    cfg.setdefault("max_seconds", 58)
    if not os.path.isabs(cfg["refs_dir"]):
        cfg["refs_dir"] = os.path.join(ROOT, cfg["refs_dir"])
    if cfg["persona_file"] and not os.path.isabs(cfg["persona_file"]):
        cfg["persona_file"] = os.path.join(ROOT, cfg["persona_file"])
    return cfg


# ───────────────────────── 合成 ─────────────────────────

def gsv_tts(text: str, params: dict, text_lang: str) -> bytes:
    body = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": params["ref_audio_path"],
        "prompt_text": params["prompt_text"],
        "prompt_lang": params["prompt_lang"],
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": False,
        "use_cuda_graph": False,
        "use_flash_attention": False,
        "seed": int(params.get("seed", 42)),
        "speed_factor": float(params.get("speed_factor", 1.0)),
        "temperature": float(params.get("temperature", 1.0)),
        "top_k": int(params.get("top_k", 15)),
        "top_p": float(params.get("top_p", 1.0)),
        "repetition_penalty": float(params.get("repetition_penalty", 1.35)),
        "fragment_interval": float(params.get("fragment_interval", 0.3)),
    }
    req = urllib.request.Request(
        CFG["gsv_url"].rstrip("/") + "/tts",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=CFG.get("gsv_timeout", 180)) as resp:
        return resp.read()


def synthesize(text: str, hint: str | None = None, text_lang: str | None = None,
               lang: str | None = None) -> tuple[bytes, dict]:
    lang = lang or CFG.get("lang") or A.detect_lang(text) or "zh"
    tlang = text_lang or CFG["text_lang"]
    if tlang in ("auto", ""):
        tlang = {"zh": "zh", "ja": "all_ja", "en": "en"}.get(lang, "auto")
    used: dict = {"ref": "", "labels": []}
    per_seg = bool(CFG.get("per_segment_ref"))

    def synth_fn(seg_text: str, overrides: dict) -> bytes:
        seg_info = overrides.get("segment") or {}
        label = (seg_info.get("label") if per_seg else
                 ((overrides.get("emotion") or {}).get("label"))) or "平静"
        inten = float(seg_info.get("intensity", 0.3) or 0.3)
        speed = float(overrides.get("gpt_sovits_speed", 1.0) or 1.0)
        pick = refs.pick(label, inten, speed=speed, out_dir=CFG["refs_dir"],
                         lang=lang, clean_only=bool(CFG.get("clean_refs")))
        if not pick:
            raise RuntimeError("ref 库为空，先跑 tools/derive_refs.py")
        used["ref"] = os.path.basename(str(pick["gpt_sovits_ref_audio"]))
        used["labels"].append(label)
        params = {
            "ref_audio_path": pick["gpt_sovits_ref_audio"],
            "prompt_text": pick["gpt_sovits_prompt_text"],
            "prompt_lang": pick["gpt_sovits_prompt_lang"],
            "speed_factor": pick["gpt_sovits_speed"],
            "temperature": pick["gpt_sovits_temperature"],
            "top_k": pick["gpt_sovits_top_k"],
            "top_p": pick["gpt_sovits_top_p"],
            "repetition_penalty": pick["gpt_sovits_rep_pen"],
            "fragment_interval": pick["gpt_sovits_fragment_interval"],
            "seed": CFG.get("seed", 42),
        }
        return gsv_tts(seg_text, params, tlang)

    persona = P.load(CFG.get("persona_file") or None)
    res = R.render(
        text, synth_fn,
        persona=persona, lang=lang, hint_label=hint,
        backend_dsp=False,                      # 情绪由模型演，不再叠加后期处理
        tail_trim_db=float(CFG.get("tail_trim_db", -45.0)),
        max_seconds=float(CFG.get("max_seconds", 58)),
        log=lambda *_: None,
    )
    meta = {
        "emotion": res.meta.get("emotion"),
        "segments": len(res.meta.get("segments") or []),
        "duration": res.meta.get("duration"),
        "ref": used["ref"],
        "labels": used["labels"],
        "text_lang": tlang,
        "lang": lang,
    }
    return res.wav, meta


def analyze_only(text: str, hint: str | None = None) -> dict:
    pt = A.analyze(text, hint_label=hint)
    return {
        "label": pt.label,
        "valence": round(pt.valence, 3), "arousal": round(pt.arousal, 3),
        "dominance": round(pt.dominance, 3), "intensity": round(pt.intensity, 3),
        "confidence": round(pt.confidence, 3), "source": pt.source,
        "arc": [{"text": c, "label": (pt.arc[i].label if i < len(pt.arc) else pt.label)}
                for i, c in enumerate(A.split_clauses(text))],
    }


# ───────────────────────── HTTP ─────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "emotion-server/1.0"

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            rows = ref_stats(block=False)
            ok = [r for r in rows if r.get("ok")] if rows else []
            gsv = "?"
            try:
                r = urllib.request.urlopen(CFG["gsv_url"].rstrip("/") + "/docs", timeout=3)
                gsv = "ok" if r.status == 200 else f"http {r.status}"
            except Exception as e:  # noqa: BLE001
                gsv = f"down: {type(e).__name__}"
            payload = {
                "ready": bool(ok) if rows else "warming",
                "gsv": gsv,
                "clean_refs": CFG.get("clean_refs"),
                "per_segment_ref": CFG.get("per_segment_ref"),
            }
            if rows:
                payload["refs"] = {"total": len(rows), "usable": len(ok), "dir": CFG["refs_dir"]}
            else:
                payload["refs"] = {"dir": CFG["refs_dir"], "stats": "warming（首次算指纹要十几秒）"}
            return self._json(200, payload)
        if self.path.startswith("/refs"):
            return self._json(200, {"refs": ref_stats(force=True)})
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"error": f"bad json: {e}"})
        text = (body.get("text") or "").strip()
        if not text:
            return self._json(400, {"error": "text 不能为空"})
        try:
            if self.path.startswith("/analyze"):
                return self._json(200, analyze_only(text, body.get("emotion")))
            if self.path.startswith("/tts"):
                wav, meta = synthesize(text, hint=body.get("emotion"),
                                       text_lang=body.get("text_lang"),
                                       lang=body.get("lang"))
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav)))
                # 头只能是 latin-1，中文标签做 percent 编码
                self.send_header("X-Emotion", urllib.request.quote(
                    ((meta.get("emotion") or {}).get("label") or ""), safe=""))
                self.send_header("X-Ref", urllib.request.quote(str(meta.get("ref")), safe=""))
                self.end_headers()
                self.wfile.write(wav)
                return None
            return self._json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt: str, *args) -> None:  # 安静点
        if os.environ.get("EMOTION_SERVER_VERBOSE"):
            super().log_message(fmt, *args)


def main() -> int:
    ap = argparse.ArgumentParser(description="GPT-SoVITS 情感层服务")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--listen", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9883)
    args = ap.parse_args()

    global CFG
    CFG = load_config(args.config)
    # ref 指纹在后台算，别拖慢启动和健康检查
    threading.Thread(target=ref_stats, kwargs={"force": True}, daemon=True).start()
    srv = ThreadingHTTPServer((args.listen, args.port), Handler)
    print(f"[i] 监听 http://{args.listen}:{args.port}  → POST /tts")
    print(f"[i] ref 库：{CFG['refs_dir']}（指纹后台计算中）")
    print(f"[i] GSV: {CFG['gsv_url']}   clean_refs={CFG.get('clean_refs')}  "
          f"per_segment_ref={CFG.get('per_segment_ref')}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[i] 退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
