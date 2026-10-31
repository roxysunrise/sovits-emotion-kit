"""doctor —— 装完先跑这个：把「能不能出带情感的声音」逐项检查一遍。

    python tools/doctor.py --config config.json

检查项：
  1. 模块与依赖（纯标准库，能 import 就赢）
  2. ref 库（10 个情绪 ref 是否齐、时长/采样率/是否静音）
  3. GPT-SoVITS 连通（/docs 或 /tts）
  4. 端到端 A/B：「平静 vs 难过」两段，看客观指标是不是真的分开了
  5. 结论：直接能用的命令
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from emotion import dsp, metrics, refs  # noqa: E402

OK = "[ OK ]"
NG = "[FAIL]"
_checks: list[tuple[bool, str, str]] = []


def chk(name: str, cond: bool, detail: str = "") -> bool:
    _checks.append((bool(cond), name, detail))
    print(f"{OK if cond else NG} {name}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--skip-tts", action="store_true", help="不做联网合成测试")
    args = ap.parse_args()

    print("=" * 64)
    print("  GPT-SoVITS 情感层 · 自检")
    print("=" * 64)

    cfg: dict = {}
    if os.path.exists(args.config):
        with open(args.config, encoding="utf-8-sig") as f:
            cfg = json.load(f)
        cfg.setdefault("gsv_url", "http://127.0.0.1:9880")
        cfg.setdefault("refs_dir", "refs/emotional")
        if not os.path.isabs(cfg["refs_dir"]):
            cfg["refs_dir"] = os.path.join(ROOT, cfg["refs_dir"])
    chk("找到 config.json", bool(cfg), args.config if cfg else "缺配置，用 --config 指一个")

    # 1) 模块
    try:
        import emotion  # noqa: F401
        from emotion import analyzer, renderer  # noqa: F401
        chk("情感模块可导入（纯标准库）", True)
    except Exception as e:  # noqa: BLE001
        chk("情感模块可导入", False, f"{type(e).__name__}: {e}")
        return 1

    # 2) ref 库
    rows = refs.stats(cfg.get("refs_dir", "refs/emotional"))
    usable = [r for r in rows if r.get("ok")]
    chk("ref 库齐全（≥8 个可用）", len(usable) >= 8,
        f"{len(usable)}/{len(rows)} 可用 @ {cfg.get('refs_dir')}")
    for r in rows:
        if not r.get("ok"):
            print(f"       ✗ {r['label']}: {r.get('problem')}")
    if usable:
        f0s = [r["f0_med"] for r in usable]
        chk("ref 的音高有梯度（max-min ≥ 25Hz）", max(f0s) - min(f0s) >= 25.0,
            f"{min(f0s):.0f}~{max(f0s):.0f}Hz")
        chk("ref 时长都在 3~12s", all(3.0 <= r["sec"] <= 12.5 for r in usable),
            f"{min(r['sec'] for r in usable):.2f}~{max(r['sec'] for r in usable):.2f}s")

    # 3) GSV 连通
    if not args.skip_tts:
        url = cfg.get("gsv_url", "http://127.0.0.1:9880").rstrip("/")
        alive = False
        for probe in ("/docs", "/openapi.json", "/tts"):
            try:
                req = urllib.request.Request(url + probe, method="GET")
                with urllib.request.urlopen(req, timeout=3) as r:
                    alive = r.status in (200, 405, 422)
                    if alive:
                        break
            except Exception:  # noqa: BLE001
                continue
        chk("GPT-SoVITS 在线", alive, url)

        if alive and usable:
            try:
                sys.path.insert(0, os.path.join(ROOT, "server"))
                import emotion_server as ES  # noqa: E402
                ES.CFG = {**cfg}
                wav_sad, meta_sad = ES.synthesize("今日はちょっと寂しいな。", hint="难过")
                wav_calm, meta_calm = ES.synthesize("今日はちょっと寂しいな。", hint="平静")
                xs, rs = dsp.parse_wav(wav_sad)
                xc, rc = dsp.parse_wav(wav_calm)
                fs, fc = metrics.fingerprint(xs, rs), metrics.fingerprint(xc, rc)
                d_f0 = fc["f0_med"] - fs["f0_med"]
                chk("端到端：难过比平静低音高（≥8Hz）", d_f0 >= 8.0,
                    f"难过 {fs['f0_med']:.0f}Hz / 平静 {fc['f0_med']:.0f}Hz → Δ{d_f0:+.1f}Hz")
                chk("端到端：难过比平静慢或轻",
                    fs["rate_syl"] < fc["rate_syl"] or fs["rms_db"] < fc["rms_db"] - 0.5,
                    f"语速 {fs['rate_syl']:.2f} vs {fc['rate_syl']:.2f}，"
                    f"能量 {fs['rms_db']:.1f} vs {fc['rms_db']:.1f}dB")
                out = os.path.join(ROOT, "doctor_ab")
                os.makedirs(out, exist_ok=True)
                open(os.path.join(out, "sad.wav"), "wb").write(wav_sad)
                open(os.path.join(out, "calm.wav"), "wb").write(wav_calm)
                print(f"       试听：{out}\\sad.wav / calm.wav  ref={meta_sad.get('ref')}")
            except Exception as e:  # noqa: BLE001
                chk("端到端合成", False, f"{type(e).__name__}: {e}")

    bad = [c for c in _checks if not c[0]]
    print("-" * 64)
    print(f"  合计 {len(_checks)} 项，通过 {len(_checks) - len(bad)}，失败 {len(bad)}")
    if not bad:
        print("""
  可以用了：
    python server/emotion_server.py --config config.json      # 起一个带情绪的 /tts
    curl -X POST http://127.0.0.1:9883/tts -H "Content-Type: application/json" \\
         -d "{\\"text\\":\\"今天有点累，不过看到你就好啦\\"}" -o out.wav
""")
    else:
        print("  先修上面 FAIL 的项：ref 没生成就跑 tools/derive_refs.py；GSV 不在线先起模型。")
    print("=" * 64)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
