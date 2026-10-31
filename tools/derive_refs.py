"""derive_refs —— 从你的一段参考音频，生成 10 个情绪 ref。

只要你有 GPT-SoVITS 在跑，并且手上有一段**你自己音色的干净录音**（3~8 秒），
就能得到「按情绪换参考音频」所需的一整套 ref。不需要训练、不需要改模型。

    python tools/derive_refs.py --config config.json
    python tools/derive_refs.py --base D:/voice/my-ref.wav --prompt-text "这段音频的逐字转写"
    python tools/derive_refs.py --config config.json --list      # 只看现有 ref 的指纹

参数说明：
    --base        基准参考音频（中性语气、单人、无背景音，3~12 秒）
    --prompt-text 基准音频的**逐字转写**，必须和音频内容完全一致，否则韵律会错位
    --prompt-lang 参考音频的语言：all_zh / all_ja / all_en
    --out         输出目录（默认 refs/emotional）
    --force       覆盖已存在的 ref（会覆盖你手工放进去的真人 ref，慎用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from emotion import refs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="生成情绪参考音频（ref）")
    ap.add_argument("--config", default="", help="读 config.json 里的 base_ref/prompt_text")
    ap.add_argument("--base", default="", help="基准 ref 音频路径")
    ap.add_argument("--prompt-text", default="", help="基准 ref 的逐字转写")
    ap.add_argument("--prompt-lang", default="all_zh", help="all_zh / all_ja / all_en")
    ap.add_argument("--out", default="", help="输出目录，默认 refs/emotional")
    ap.add_argument("--force", action="store_true", help="覆盖已有 ref")
    ap.add_argument("--list", action="store_true", help="只列出现有 ref 的指纹")
    ap.add_argument("--validate", default="", help="只校验某个 ref 文件")
    args = ap.parse_args()

    cfg: dict = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8-sig") as f:
            cfg = json.load(f)

    out_dir = args.out or cfg.get("refs_dir") or "refs/emotional"
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(ROOT, out_dir)

    if args.validate:
        ok, info = refs.validate(args.validate)
        print(json.dumps({"ok": ok, **info}, ensure_ascii=False, indent=1))
        return 0 if ok else 1

    if args.list:
        rows = refs.stats(out_dir)
        if not rows:
            print("没有 ref。先跑一次 derive。")
            return 1
        print(f"{'标签':<6}{'时长':>7}{'F0':>9}{'能量':>9}{'气声':>8}  状态")
        for r in rows:
            state = "OK" if r["ok"] else r["problem"]
            print(f"{r['label']:<6}{r['sec']:>6.2f}s{r['f0_med']:>8.1f}Hz"
                  f"{r['rms_db']:>8.1f}dB{r['hf_ratio']:>8.3f}  {state}")
        return 0

    base = args.base or cfg.get("base_ref") or ""
    prompt_text = args.prompt_text or cfg.get("prompt_text") or ""
    prompt_lang = args.prompt_lang or cfg.get("prompt_lang") or "all_zh"

    if not base or not os.path.exists(base):
        print("[!] 缺少 --base（基准参考音频）。它应该是你自己的音色、中性语气、3~12 秒。")
        return 2
    if not prompt_text:
        print("[!] 缺少 --prompt-text（基准音频的逐字转写）。")
        print("    不做这一步，GPT-SoVITS 会把文字和韵律对错位，念出来会飘。")
        return 2

    print(f"[i] base      : {base}")
    print(f"[i] prompt    : {prompt_text}")
    print(f"[i] lang      : {prompt_lang}")
    print(f"[i] 输出       : {out_dir}")
    made = refs.derive_all(base, out_dir, prompt_text=prompt_text,
                           prompt_lang=prompt_lang, force=args.force)
    print(f"\n已生成 {len(made)} 个 ref。指纹：")
    for r in refs.stats(out_dir):
        state = "OK" if r["ok"] else r["problem"]
        print(f"  {r['label']:<6}{r['sec']:>6.2f}s  F0={r['f0_med']:>6.1f}Hz  "
              f"rms={r['rms_db']:>6.1f}dB  hf={r['hf_ratio']:.3f}  {state}")
    print("\n下一步：python tools/doctor.py --config config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
